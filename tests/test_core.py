from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_rag.benchmark import evaluate, percentile, run_benchmark, compare_reports
from knowledge_rag.config import Config, Source
from knowledge_rag.index import connect, ingest, manifest, parse_markdown
from knowledge_rag.retrieval import Retriever, tokenize


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("a", "b"):
            (self.root / name).mkdir()
        (self.root / "config.toml").write_text("# unit test configuration", encoding="utf-8")
        self.config = Config(self.root / "config.toml", self.root / "index.db", self.root / "traces.jsonl",
                             (Source("a", self.root / "a"), Source("b", self.root / "b")))

    def write(self, source, name, text):
        path = self.root / source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_incremental_update_delete_and_version_guard(self):
        path = self.write("a", "note.md", "# Alpha\nold value")
        self.assertEqual(ingest(self.config)["added"], 1)
        old = Retriever(self.config).search("Alpha")["hits"][0]
        self.assertEqual(ingest(self.config)["unchanged"], 1)
        path.write_text("# Beta\nnew value", encoding="utf-8")
        self.assertEqual(ingest(self.config)["updated"], 1)
        self.assertFalse(Retriever(self.config).search("Alpha")["hits"])
        with self.assertRaisesRegex(ValueError, "version changed"):
            Retriever(self.config).read_document(old["doc_id"], old["source_hash"])
        path.unlink()
        self.assertEqual(ingest(self.config)["deleted"], 1)
        self.assertFalse(Retriever(self.config).search("Beta")["hits"])

    def test_missing_root_does_not_delete_index(self):
        self.write("a", "note.md", "survival")
        ingest(self.config)
        missing = replace(self.config, sources=(Source("a", self.root / "missing"),))
        with self.assertRaises(ValueError):
            ingest(missing)
        self.assertEqual(len(manifest(self.config.database)), 1)

    def test_bad_document_rolls_back_whole_batch(self):
        self.write("a", "a.md", "original")
        ingest(self.config)
        self.write("a", "a.md", "changed")
        self.write("a", "z.md", "---\ntitle: invalid\n")
        with self.assertRaises(ValueError):
            ingest(self.config)
        self.assertEqual(Retriever(self.config).read_document("a:a.md")["text"], "original")

    def test_cross_kb_and_status_filter(self):
        self.write("a", "note.md", "---\nstatus: current\n---\n# Atlas\nowner Lin")
        self.write("b", "note.md", "---\nstatus: old\n---\n# Atlas\nowner Zhou")
        ingest(self.config)
        retriever = Retriever(self.config)
        self.assertEqual({h["kb_id"] for h in retriever.search("Atlas", ["a"])["hits"]}, {"a"})
        self.assertEqual([h["doc_id"] for h in retriever.search("Atlas", filters={"status": "current"})["hits"]], ["a:note.md"])
        with self.assertRaises(ValueError):
            retriever.search("Atlas", ["missing"])
        with self.assertRaises(ValueError):
            retriever.search("Atlas", [])

    def test_removed_source_not_exposed_before_resync(self):
        self.write("b", "note.md", "Atlas")
        ingest(self.config)
        narrowed = replace(self.config, sources=(self.config.sources[0],))
        self.assertFalse(Retriever(narrowed).search("Atlas")["hits"])
        with self.assertRaises(ValueError):
            Retriever(narrowed).read_document("b:note.md")

    def test_budget_and_citation_lines(self):
        content = "---\nstatus: verified\n---\n# Hello\nEvidence is here.\n## Next\nOther text.\n"
        path = self.write("a", "note.md", content)
        ingest(self.config)
        hits = Retriever(self.config).search("Evidence", max_chars=9)["hits"]
        self.assertEqual(sum(len(h["text"]) for h in hits), 9)
        self.assertTrue(hits[0]["truncated"])
        full = Retriever(self.config).search("Evidence")["hits"][0]
        original = "".join(path.read_bytes().decode("utf-8").splitlines(keepends=True)[full["start_line"] - 1:full["end_line"]])
        self.assertEqual(full["text"], original)

    def test_pipeline_change_rechunks_unchanged_file(self):
        self.write("a", "note.md", "# Text\n" + ("one long line\n" * 30))
        ingest(self.config)
        changed = ingest(replace(self.config, chunk_chars=100))
        self.assertEqual(changed["updated"], 1)
        self.assertGreater(changed["chunks_written"], 1)

    def test_frontmatter_and_code_fence(self):
        title, metadata, chunks = parse_markdown("---\ndate: 2026-10-07\ntags: [a, b]\n---\n# Real\n```python\n# fake heading\n```\n", "fallback", 100)
        self.assertEqual(title, "Real")
        self.assertEqual(metadata["date"], "2026-10-07")
        self.assertEqual(metadata["tags"], ["a", "b"])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["heading"], "Real")

    def test_trace_has_stages_but_no_query_or_text(self):
        self.write("a", "note.md", "secretphrase sourcecontent")
        ingest(self.config)
        result = Retriever(self.config).search("secretphrase")
        trace_text = self.config.traces.read_text(encoding="utf-8")
        self.assertNotIn("secretphrase", trace_text)
        self.assertNotIn("sourcecontent", trace_text)
        record = json.loads(trace_text.splitlines()[-1])
        self.assertEqual(record["trace_id"], result["trace_id"])
        self.assertEqual(set(record["stages_ms"]), {"load_filter", "tokenize", "rank", "pack"})

    def test_error_is_traced(self):
        with self.assertRaises(ValueError):
            Retriever(self.config).search("")
        record = json.loads(self.config.traces.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["status"], "error")

    def test_hidden_and_excluded_files(self):
        self.write("a", ".hidden.md", "secret")
        self.write("a", "secret.md", "secret")
        self.write("a", "allowed.md", "public")
        sources = (replace(self.config.sources[0], exclude=("secret.md",)),)
        self.assertEqual(ingest(replace(self.config, sources=sources))["added"], 1)

    def test_tokenizer_mixed_language(self):
        self.assertEqual(tokenize("RRF 语义检索 source_hash"), ["rrf", "语义", "义检", "检索", "source_hash"])

    def test_metrics_require_visible_evidence(self):
        case = {"relevant": [{"doc_id": "a", "contains": "evidence"}, {"doc_id": "b", "contains": "proof"}]}
        hits = [{"doc_id": "a", "text": "wrong section"}, {"doc_id": "b", "text": "proof"},
                {"doc_id": "b", "text": "proof"}]
        score = evaluate(case, hits, 3)
        self.assertEqual(score["recall_at_k"], 0.5)
        self.assertEqual(score["mrr_at_k"], 0.5)
        self.assertEqual(score["precision_at_k"], 1 / 3)

    def test_unanswerable_is_not_counted_as_perfect_recall(self):
        score = evaluate({"relevant": []}, [], 5)
        self.assertIsNone(score["recall_at_k"])
        self.assertEqual(score["empty_on_unanswerable"], 1)
        self.assertEqual(percentile([1, 2, 3], 0.95), 2.9)

    def test_benchmark_isolated_and_compare_rejects_drift(self):
        self.write("a", "note.md", "# target\nevidence")
        ingest(self.config)
        before = manifest(self.config.database)
        dataset = self.root / "cases.jsonl"
        dataset.write_text(json.dumps({"id": "q", "category": "exact", "query": "target", "relevant": [
            {"doc_id": "a:note.md", "contains": "evidence"}]}), encoding="utf-8")
        report = run_benchmark(self.config, dataset, self.root / "runs", repeats=2, warmup=0)
        self.assertEqual(manifest(self.config.database), before)
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["recall_at_k"], 1)
        comparison = compare_reports(report, report)
        self.assertEqual(comparison["delta_candidate_minus_baseline"]["recall_at_k"], 0)
        data["corpus_hash"] = "changed"
        different = self.root / "different.json"
        different.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corpus_hash"):
            compare_reports(report, different)


if __name__ == "__main__":
    unittest.main()
