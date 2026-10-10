from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge_rag.config import Config, Source, EmbeddingConfig, VectorConfig, load_config
from knowledge_rag.embedding_providers import APIProvider, CachedProvider, TextProvider, document_windows
from knowledge_rag.embeddings import EmbeddingSpec
from knowledge_rag.dense_index import build_vector, vector_status, load_active
from knowledge_rag.onboarding import config_text
from knowledge_rag.retrieval import Retriever


class FakeProvider(TextProvider):
    def __init__(self, config=None):
        self.config = config or EmbeddingConfig(dimension=3, max_tokens=32, template="plain-v1")
        self.spec = EmbeddingSpec("fake", "fake", "v1", 3, self.config.max_tokens, "plain-v1")
        self.load_ms = 0.0
        self.calls = []
        self.fail = False

    def _count(self, text):
        return len(text) + 2

    def _encode(self, texts):
        if self.fail:
            raise ValueError("simulated embedding failure")
        self.calls.extend(texts)
        return [[0.0, 1.0, 0.0] if "second" in text else [1.0, 0.0, 0.0] for text in texts]


class EmbeddingTests(unittest.TestCase):
    def test_windows_cover_every_character_and_full_template(self):
        provider = FakeProvider()
        for text in ("中文🙂" * 100, "x" * 2048, "a b\n" * 100):
            units = document_windows(provider, text, "title")
            self.assertGreater(len(units), 1)
            self.assertEqual("".join(unit.text[len("title\n"):] for unit in units), text)
            self.assertEqual(units[0].start, 0)
            self.assertEqual(units[-1].end, len(text))
            self.assertTrue(all(unit.tokens <= provider.spec.max_tokens for unit in units))
            self.assertTrue(all(a.end == b.start for a, b in zip(units, units[1:])))
        with self.assertRaises(ValueError):
            document_windows(provider, "a", "x" * 32)
        with self.assertRaises(ValueError):
            provider.encode_query("x" * 32)

    def test_cache_reuse_role_and_spec_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.db"
            raw = FakeProvider()
            provider = CachedProvider(raw, path, query_cache=True)
            vectors = provider.encode_documents(["same", "same", "other"])
            self.assertEqual(len(raw.calls), 2)
            self.assertEqual(provider.encode_documents(["same", "other"]), vectors[::2])
            self.assertEqual(len(raw.calls), 2)
            provider.encode_query("same")
            self.assertEqual(len(raw.calls), 3)
            provider.encode_query("same")
            self.assertEqual(len(raw.calls), 3)
            raw.spec = replace(raw.spec, revision="v2")
            changed = CachedProvider(raw, path)
            changed.encode_documents(["same"])
            self.assertEqual(len(raw.calls), 4)

    def test_failed_encoding_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = FakeProvider()
            provider = CachedProvider(raw, Path(directory) / "cache.db")
            raw.fail = True
            with self.assertRaises(ValueError):
                provider.encode_documents(["a"])
            raw.fail = False
            provider.encode_documents(["a"])
            self.assertEqual(len(raw.calls), 1)

    def test_embedding_check_is_repeatable_and_retains_report(self):
        from knowledge_rag.embedding_check import run_embedding_check
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = FakeProvider()
            provider = CachedProvider(raw, root / "cache.db")
            config = Config(root / "config.toml", root / "index.db", root / "traces", (Source("alpha", root),),
                            embedding=replace(raw.config, cache=root / "cache.db"))
            with patch("knowledge_rag.embedding_check.get_provider", return_value=provider):
                first, second = run_embedding_check(config), run_embedding_check(config)
            self.assertEqual(first["passed_count"], 6)
            self.assertEqual(second["status"], "passed")
            self.assertNotEqual(first["report"], second["report"])
            with patch("knowledge_rag.embedding_check.get_provider", side_effect=ValueError("failure")), self.assertRaises(ValueError):
                run_embedding_check(config)
            self.assertTrue(any(json.loads(path.read_text())["status"] == "failed" for path in root.glob("embedding-checks/*/report.json")))

    def test_configuration_roundtrip_preserves_vector_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(root / "config.toml", root / "db", root / "trace", (Source("alpha", root),),
                            embedding=EmbeddingConfig(model_dir=root / "model", cache=root / "cache.db"),
                            vector=VectorConfig(root / "vectors"))
            config.path.write_text(config_text(config), encoding="utf-8")
            self.assertEqual(config, load_config(config.path))
            with self.assertRaises(ValueError):
                EmbeddingConfig(batch_size=0)

    def test_invalid_api_address_or_missing_key(self):
        for url in ("file:///tmp", "http://remote.example/v1", "https://user:password@example/v1", "https://example/v1?key=example"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                APIProvider(EmbeddingConfig(provider="api", base_url=url))
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "not set"):
            APIProvider(EmbeddingConfig(provider="api", base_url="https://example/v1"))


@unittest.skipUnless(importlib.util.find_spec("httpx") and importlib.util.find_spec("tiktoken"), "requires embedding-api")
class APITests(unittest.TestCase):
    def make_provider(self):
        import tiktoken
        config = EmbeddingConfig(provider="api", base_url="https://example.test/v1", model="test", revision="v1",
                                 dimension=3, max_tokens=32, template="plain-v1")
        with patch.dict(os.environ, {config.api_key_env: "synthetic-test-credential"}), patch.object(tiktoken, "get_encoding") as getter:
            getter.return_value.encode.side_effect = lambda text, **kwargs: list(text)
            return APIProvider(config)

    def test_api_order_normalization_and_key_header(self):
        import httpx
        provider = self.make_provider()
        response = httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2, 0]}, {"index": 0, "embedding": [2, 0, 0]}]})
        with patch.dict(os.environ, {provider.config.api_key_env: "synthetic-test-credential"}), patch.object(httpx.Client, "post", return_value=response) as post:
            self.assertEqual(provider.encode_documents(["a", "b"]), [[1, 0, 0], [0, 1, 0]])
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer synthetic-test-credential")
            self.assertEqual(post.call_args.kwargs["json"]["input"], ["a", "b"])

    def test_api_errors_do_not_echo_credentials_or_body(self):
        import httpx
        provider = self.make_provider()
        for response in (httpx.Response(401, text="synthetic-test-credential private text"),
                         httpx.Response(200, json={"data": [{"index": 1, "embedding": [1, 0, 0]}]}),
                         httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]}),
                         httpx.Response(200, json={"data": None}),
                         httpx.Response(200, json={"data": [{"index": 0, "embedding": None}]}),
                         httpx.Response(200, text=json.dumps({"data": [{"index": 0, "embedding": [float("nan"), 0, 0]}]}))):
            with patch.dict(os.environ, {provider.config.api_key_env: "synthetic-test-credential"}), patch.object(httpx.Client, "post", return_value=response):
                with self.assertRaises(ValueError) as error:
                    provider.encode_query("a")
                self.assertNotIn("synthetic-test-credential", str(error.exception))
                self.assertNotIn("private text", str(error.exception))


@unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "requires vector extra")
class DenseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for kb in ("alpha", "beta"):
            (self.root / kb).mkdir()
        (self.root / "alpha/first.md").write_text("---\nstatus: reviewed\n---\n# first\n" + "x" * 1400, encoding="utf-8")
        (self.root / "alpha/second.md").write_text("---\nstatus: draft\n---\n# second\nsecond", encoding="utf-8")
        (self.root / "beta/foreign.md").write_text("# first\nforeign", encoding="utf-8")
        self.config = Config(self.root / "config.toml", self.root / "index.db", self.root / "traces.jsonl",
            (Source("alpha", self.root / "alpha"), Source("beta", self.root / "beta")),
            embedding=EmbeddingConfig(dimension=3, max_tokens=32, template="plain-v1", cache=self.root / "cache.db"),
            vector=VectorConfig(self.root / "vectors"))
        self.raw = FakeProvider(self.config.embedding)
        self.provider = CachedProvider(self.raw, self.config.embedding.cache)
        patcher = patch("knowledge_rag.embedding_providers.get_provider", return_value=self.provider)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_long_windows_scope_parent_dedup_and_read(self):
        report = build_vector(self.config)
        self.assertGreater(report["points"], 50)
        self.assertEqual(report["split_chunks"], 1)
        result = Retriever(self.config, "dense").search("first", ["alpha"], top_k=2, max_chars=1500)
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual({h["doc_id"] for h in result["hits"]}, {"alpha:first.md", "alpha:second.md"})
        hit = result["hits"][0]
        self.assertIn("first", Retriever(self.config).read_document(hit["doc_id"], hit["source_hash"])["text"])
        filtered = Retriever(self.config, "dense").search("first", ["alpha"], filters={"status": "draft"})
        self.assertEqual([h["doc_id"] for h in filtered["hits"]], ["alpha:second.md"])
        self.assertEqual(Retriever(self.config, "dense").search("first", filters={"status": "missing"})["hits"], [])

    def test_no_change_encoding_zero_update_and_delete(self):
        build_vector(self.config)
        report = build_vector(self.config)
        self.assertEqual(report["encoding"]["misses"], 0)
        self.assertEqual(report["encoding"]["encode_calls"], 0)
        (self.root / "alpha/first.md").write_text("# first\nchanged", encoding="utf-8")
        (self.root / "alpha/second.md").unlink()
        self.assertEqual(vector_status(self.config)["status"], "stale")
        report = build_vector(self.config)
        self.assertGreater(report["encoding"]["misses"], 0)
        self.assertEqual(vector_status(self.config)["status"], "ready")
        hits = Retriever(self.config, "dense").search("first", ["alpha"])["hits"]
        self.assertEqual(len(hits), 1)
        self.assertIn("changed", hits[0]["text"])

    def test_failed_build_preserves_both_snapshots(self):
        build_vector(self.config)
        pointer = (self.config.vector.root / "active.json").read_bytes()
        (self.root / "alpha/first.md").write_text("# first\nnew", encoding="utf-8")
        self.raw.fail = True
        with self.assertRaises(ValueError):
            build_vector(self.config)
        self.assertEqual(pointer, (self.config.vector.root / "active.json").read_bytes())
        self.assertEqual(vector_status(self.config)["status"], "failed")
        self.raw.fail = False
        hit = Retriever(self.config, "dense").search("first", ["alpha"], top_k=1)["hits"][0]
        self.assertNotIn("new", hit["text"])
        self.assertNotIn("new", Retriever(self.config).read_document(hit["doc_id"], hit["source_hash"])["text"])
        self.assertTrue(any(json.loads(path.read_text())["status"] == "failed" for path in self.config.vector.root.glob("generations/*/build-report.json")))

    def test_model_or_scope_change_requires_rebuild(self):
        build_vector(self.config)
        self.provider.spec = replace(self.provider.spec, revision="new")
        with self.assertRaisesRegex(ValueError, "model changed"):
            Retriever(self.config, "dense").search("first")
        changed = replace(self.config, sources=self.config.sources[:1])
        with self.assertRaisesRegex(ValueError, "not built"):
            Retriever(changed, "dense").search("first")

    def test_source_changes_during_build_not_published(self):
        with patch("knowledge_rag.dense_index.source_versions", return_value=[]), self.assertRaises(ValueError):
            build_vector(self.config)
        self.assertIsNone(load_active(self.config))

    def test_publish_failure_keeps_old_generation_and_records_failure(self):
        build_vector(self.config)
        pointer = (self.config.vector.root / "active.json").read_bytes()
        with patch("knowledge_rag.dense_index.os.replace", side_effect=OSError("simulated publication failure")):
            with self.assertRaises(OSError):
                build_vector(self.config)
        self.assertEqual(pointer, (self.config.vector.root / "active.json").read_bytes())
        self.assertEqual(vector_status(self.config)["last_build_status"], "failed")

    def test_delete_all_documents_publishes_empty_generation(self):
        build_vector(self.config)
        for source in self.config.sources:
            for path in source.root.glob("*.md"):
                path.unlink()
        report = build_vector(self.config)
        self.assertEqual(report["points"], 0)
        self.assertEqual(Retriever(self.config, "dense").search("first")["hits"], [])
        self.assertEqual(Retriever(self.config).list_knowledge_bases(), [])
