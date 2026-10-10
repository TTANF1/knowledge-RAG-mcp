from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.check_publication import audit, forbidden_path, inspect_file
from scripts.export_benchmark_summary import export as export_summary, fingerprint
from scripts.export_public import export as export_snapshot


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Publication Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.write(".gitignore", ".state/\n.local/\n*.db\n")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=True).stdout

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_private_artifact_patterns_and_public_fixture(self):
        for name in (".state/index.db", ".local/learning.md", "other.sqlite3-wal", "logs/run.txt",
                     "benchmark/traces.jsonl", "benchmarks/baselines/report.json", "config.user.local.toml", ".env.production"):
            self.assertTrue(forbidden_path(name), name)
        self.assertFalse(forbidden_path("benchmarks/datasets/smoke-v1.jsonl"))
        self.assertFalse(forbidden_path(".env.example"))

    def test_paths_and_key_are_detected_without_echoing_secret(self):
        home = "C:" + "/Users/" + "private-person/vault"
        secret = "sk-" + "x" * 30
        issues = inspect_file("notes.md", (home + "\n" + secret).encode(), [])
        self.assertTrue(any("path" in item for item in issues))
        self.assertTrue(any("api-key" in item for item in issues))
        self.assertFalse(any(secret in item or home in item for item in issues))
        self.assertFalse(inspect_file("notes.md", b"C:/path/to/your/knowledge", []))

    def test_forced_ignored_staged_file_is_blocked(self):
        self.write(".local/private.db", "private body")
        self.git("add", "-f", ".local/private.db")
        self.assertTrue(any(item["reason"] == "private-artifact-path" for item in audit(self.root, staged=True)))

    def test_staged_content_checked_even_if_working_copy_fixed(self):
        private_path = str(self.root / "vault")
        self.write("notes.md", private_path)
        self.git("add", "notes.md")
        self.write("notes.md", "generic public note")
        self.assertTrue(audit(self.root, staged=True))
        self.assertFalse(audit(self.root))

    def test_history_detects_removed_private_path(self):
        self.write("notes.md", str(self.root / "vault"))
        self.git("add", "notes.md")
        self.git("commit", "-qm", "initial")
        self.write("notes.md", "public note")
        self.git("add", "notes.md")
        self.git("commit", "-qm", "clean current contents")
        self.assertFalse(audit(self.root))
        self.assertTrue(any("commit" in item for item in audit(self.root, history=True)))

    def test_export_excludes_git_private_state_and_keeps_source(self):
        self.write("README.md", "public source")
        self.write(".state/index.db", "private body")
        self.write(".local/notes.md", "private notes")
        self.git("add", ".gitignore", "README.md")
        target = export_snapshot(self.root)
        self.assertEqual((target / "README.md").read_text(encoding="utf-8"), "public source")
        for name in (".git", ".state", ".local"):
            self.assertFalse((target / name).exists())
        self.assertTrue((self.root / ".state/index.db").exists())

    def test_public_summary_proves_synthetic_corpus_and_omits_raw_rows(self):
        self.write("examples/knowledge/learning/note.md", "Public learning")
        self.write("examples/knowledge/product/note.md", "Public product")
        cases = [{"id": "q", "query": "public", "category": "exact", "relevant": []}]
        self.write("benchmarks/datasets/smoke-v1.jsonl", json.dumps(cases[0]) + "\n")
        corpus = [{"id": f"{kb}:note.md", "source_hash": sha256((self.root / f"examples/knowledge/{kb}/note.md").read_bytes()).hexdigest()}
                  for kb in ("learning", "product")]
        settings = {"strategy": "bm25", "top_k": 5, "max_chars": 4000, "repeats": 5, "warmup": 1,
                    "chunk_chars": 1200, "tokenizer": "latin-word-cjk-bigram-v1", "bm25_k1": 1.5, "bm25_b": 0.75,
                    "timing_scope": "in-process search including trace append; excluding MCP transport"}
        report = {"dataset_hash": fingerprint(cases), "corpus_hash": fingerprint(corpus), "corpus_manifest": corpus,
                  "environment": {"code_hash": "a" * 64, "python": "3.12", "private_note": "private environment"},
                  "settings": settings, "metrics": {"recall_at_k": 1}, "by_category": {},
                  "per_query": [{"query": "private query should not publish"}]}
        source = self.write("report.json", json.dumps(report))
        result = export_summary(self.root, source, self.root / "public/first")
        public = result.read_text(encoding="utf-8")
        self.assertNotIn("private query", public)
        self.assertNotIn("private environment", public)
        self.assertNotIn("corpus_manifest", public)
        report["corpus_manifest"][0]["id"] = "private:note.md"
        source.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "approved public"):
            export_summary(self.root, source, self.root / "public/rejected")

    def test_unreviewed_public_report_blocked(self):
        issues = inspect_file("benchmarks/public/example/report.json", b'{"per_query":[]}', [])
        self.assertIn("unreviewed-public-benchmark-report", issues)

    def test_dense_summary_accepts_only_numeric_telemetry(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "benchmarks/public/2026-10-10-embedding-dense/report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(inspect_file("benchmarks/public/test/report.json", json.dumps(report).encode(), []), [])
        for location, key, value in (("vector_build", "source_ids", ["private:note"]),
                                     ("embedding_queries", "api_key", "synthetic-credential")):
            candidate = json.loads(json.dumps(report))
            candidate[location][key] = value
            self.assertIn("unreviewed-public-benchmark-report", inspect_file("benchmarks/public/test/report.json", json.dumps(candidate).encode(), []))
        report["vector_build"]["encoding"]["misses"] = "private content"
        self.assertIn("unreviewed-public-benchmark-report", inspect_file("benchmarks/public/test/report.json", json.dumps(report).encode(), []))


if __name__ == "__main__":
    unittest.main()
