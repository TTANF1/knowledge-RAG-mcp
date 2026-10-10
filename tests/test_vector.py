from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from knowledge_rag.embeddings import EmbeddingSpec
from knowledge_rag.telemetry import digest
from knowledge_rag.vector_check import fixture, run_vector_check
from knowledge_rag.vector_index import VectorManifest
from knowledge_rag.vector_store import QdrantVectorStore


class ContractTests(unittest.TestCase):
    def test_default_cli_does_not_import_optional_qdrant(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import knowledge_rag.cli; assert 'qdrant_client' not in sys.modules"],
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_failed_storage_check_keeps_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("knowledge_rag.vector_check.QdrantVectorStore", side_effect=ValueError("test failure")):
                with self.assertRaises(ValueError):
                    run_vector_check(Path(directory))
            reports = list(Path(directory).glob("*/report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error_type"], "ValueError")
            self.assertEqual(report["passed_count"], 0)
    def test_spec_identity_includes_all_parameters(self):
        spec, _ = fixture()
        for field, value in {"provider": "other", "model_id": "other", "revision": "other", "dimension": 4,
                             "max_tokens": 2, "template_version": "other", "normalize": False, "distance": "dot"}.items():
            with self.subTest(field=field):
                self.assertNotEqual(spec.spec.fingerprint, replace(spec.spec, **{field: value}).fingerprint)

    def test_invalid_spec_and_vectors(self):
        manifest, _ = fixture()
        for changes in ({"dimension": True}, {"dimension": 0}, {"max_tokens": 0}, {"revision": ""},
                        {"normalize": 1}, {"distance": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(manifest.spec, **changes)
        for vector in ((1, 0), (0, 0, 0), (2, 0, 0), (True, 0, 0), (float("nan"), 0, 0),
                       (float("inf"), 0, 0), ("1", 0, 0)):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                manifest.spec.validate_vector(vector)
        self.assertEqual(manifest.spec.validate_vector((0.8, 0.6, 0)), [0.8, 0.6, 0.0])

    def test_manifest_order_and_contract_isolation(self):
        manifest, _ = fixture()
        self.assertEqual(manifest, replace(manifest, source_versions=tuple(reversed(manifest.source_versions))))
        for candidate in (replace(manifest, generation="next"), replace(manifest, spec=replace(manifest.spec, revision="v2")),
                          replace(manifest, source_versions=(("alpha:first", digest("changed")),))):
            self.assertNotEqual(manifest.collection, candidate.collection)
            self.assertNotEqual(manifest.vector_name, candidate.vector_name)

    def test_manifest_invalid_sources(self):
        manifest, _ = fixture()
        for sources in ((("invalid", digest("a")),), (("alpha:a", "short"),),
                        (("alpha:a", digest("a")), ("alpha:a", digest("b")))):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                replace(manifest, source_versions=sources)

    def test_manifest_roundtrip_no_overwrite_and_tampering(self):
        manifest, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.save(path)
            self.assertEqual(VectorManifest.load(path), manifest)
            with self.assertRaises(FileExistsError):
                replace(manifest, generation="next").save(path)
            self.assertEqual(VectorManifest.load(path), manifest)
            self.assertFalse(list(Path(directory).glob(".vector-manifest-*")))
            for field in ("contract_id", "collection", "corpus_hash", "vector_name"):
                data = manifest.to_dict()
                data[field] = "tampered"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    VectorManifest.load(path)


@unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "requires vector extra")
class QdrantTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.points = fixture()
        self.store = QdrantVectorStore(self.manifest)
        self.addCleanup(self.store.close)
        self.store.create()
        self.store.upsert(self.points)

    def test_scope_and_metadata_before_ranking(self):
        self.assertEqual(self.store.search((1, 0, 0), top_k=1)[0].kb_id, "beta")
        self.assertEqual(self.store.search((1, 0, 0), ["alpha"], top_k=1)[0].doc_id, "alpha:first")
        self.assertEqual(self.store.search((1, 0, 0), ["alpha"], {"status": "draft"}, 1)[0].doc_id, "alpha:second")
        self.assertEqual(self.store.search((1, 0, 0), filters={"status": "missing"}), [])
        self.assertEqual(self.store.count(["alpha"]), 2)

    def test_idempotence_and_delete(self):
        before = self.store.search((1, 0, 0))
        self.store.upsert(self.points)
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(before, self.store.search((1, 0, 0)))
        self.assertEqual(self.store.delete_documents(["alpha:first"]), 1)
        self.assertEqual(self.store.delete_documents(["alpha:first"]), 0)
        self.assertEqual(self.store.count(), 2)

    def test_invalid_batch_does_not_write(self):
        for invalid in (replace(self.points[1], vector=(2, 0, 0)), replace(self.points[1], kb_id="beta"),
                        replace(self.points[1], source_hash=digest("stale")), replace(self.points[1], metadata={"bad": float("nan")})):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.store.upsert([replace(self.points[0], embedding_unit_id="new"), invalid])
            self.assertEqual(self.store.count(), 3)
        with self.assertRaises(ValueError):
            self.store.upsert([self.points[0], self.points[0]])

    def test_invalid_query_filters(self):
        for scope in ([], ["unknown"], [{}], "alpha"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.store.search((1, 0, 0), scope)
        for filters in ({"status.x": "draft"}, {"status": []}, {"status": None}, {"status": float("nan")}, []):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                self.store.search((1, 0, 0), filters=filters)
        for top_k in (0, 51, True):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.store.search((1, 0, 0), top_k=top_k)
        with self.assertRaises(ValueError):
            self.store.search((1, 0, 0), exact=False)

    def test_wrong_schema_not_recreated(self):
        self.store.client.delete_collection(self.manifest.collection)
        self.store.client.create_collection(self.manifest.collection, vectors_config={
            self.manifest.vector_name: self.store.models.VectorParams(size=4, distance=self.store.models.Distance.COSINE)})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.store.create()
        self.assertEqual(self.store.client.get_collection(self.manifest.collection).config.params.vectors[self.manifest.vector_name].size, 4)

    def test_corrupt_source_payload_rejected(self):
        hit = self.store.search((1, 0, 0))[0]
        self.store.client.set_payload(self.manifest.collection, payload={"source_hash": digest("stale")}, points=[hit.point_id])
        with self.assertRaises(ValueError):
            self.store.search((1, 0, 0))

    def test_numeric_metadata(self):
        self.store.upsert([replace(self.points[0], metadata={"rank": 2, "ratio": 0.5, "done": True})])
        for filters in ({"rank": 2}, {"ratio": 0.5}, {"done": True}):
            self.assertEqual([h.doc_id for h in self.store.search((1, 0, 0), filters=filters)], ["alpha:first"])

    def test_local_persistence_and_storage_report(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_vector_check(Path(directory))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["passed_count"], 18)
            run = Path(result["report"]).parent
            traces = [json.loads(line) for line in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(t["status"] == "error" for t in traces))
            serialized = json.dumps(traces)
            for forbidden in ("alpha:first", "reviewed", "source_hash", "vector\":"):
                self.assertNotIn(forbidden, serialized)

    def test_local_storage_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with QdrantVectorStore(self.manifest, mode="local", path=Path(directory)) as first:
                first.create()
                script = ("import sys; from pathlib import Path; from knowledge_rag.vector_check import fixture; "
                          "from knowledge_rag.vector_store import QdrantVectorStore\n"
                          "try:\n QdrantVectorStore(fixture()[0], mode='local', path=Path(sys.argv[1]))\n"
                          "except RuntimeError:\n sys.exit(0)\nelse:\n sys.exit(1)\n")
                result = subprocess.run([sys.executable, "-c", script, directory], capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            with QdrantVectorStore(self.manifest, mode="local", path=Path(directory)) as reopened:
                self.assertFalse(reopened.create())

    def test_close_and_mode_validation(self):
        self.store.close()
        self.store.close()
        with self.assertRaises(ValueError):
            self.store.search((1, 0, 0))
        for kwargs in ({"mode": "invalid"}, {"mode": "local"}, {"mode": "memory", "url": "http://localhost"},
                       {"mode": "server", "url": "file:///tmp/db"}, {"mode": "server", "url": "http://user:pass@localhost"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QdrantVectorStore(self.manifest, **kwargs)
