from dataclasses import replace
from pathlib import Path
import tempfile
import os
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from knowledge_rag.config import Config, Source, load_config
from knowledge_rag.index import ingest, manifest
from knowledge_rag.onboarding import Onboarding, active_config, saved_config_path
from knowledge_rag.retrieval import Retriever


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "base").mkdir()
        (self.root / "incoming").mkdir()
        (self.root / "base" / "old.md").write_text("# Old\nOriginal evidence", encoding="utf-8")
        self.note = self.root / "incoming" / "note.md"
        self.note.write_text("# New\nOnboarding evidence", encoding="utf-8")
        self.config = Config(self.root / "config.toml", self.root / "state" / "index.db",
                             self.root / "state" / "trace.jsonl", (Source("base", self.root / "base"),))
        self.onboarding = Onboarding(self.config)
        ingest(self.config)

    def preview(self):
        return self.onboarding.preview("new", str(self.root / "incoming"))

    def test_example_libraries_require_user_setup(self):
        project = Path(__file__).resolve().parents[1]
        demo = replace(load_config(project / "config.example.toml"), database=self.root / "demo.db",
                       traces=self.root / "demo-traces.jsonl")
        result = Onboarding(demo).status()
        self.assertEqual(result["status"], "needs_knowledge_base")
        self.assertEqual(result["next_action"], "ask_user_in_current_conversation")
        self.assertIn("完整路径", result["question"])

    def test_cli_question_is_utf8_with_legacy_windows_encoding(self):
        project = Path(__file__).resolve().parents[1]
        example = project / "examples" / "knowledge" / "learning"
        path = self.root / "demo.toml"
        path.write_text('[[sources]]\nid = "learning"\nroot = ' + json.dumps(str(example)) + '\n', encoding="utf-8")
        result = subprocess.run([sys.executable, "-m", "knowledge_rag.cli", "--config", str(path), "setup-status"],
                                env={**os.environ, "PYTHONIOENCODING": "cp936"},
                                capture_output=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout.decode("utf-8"))["question"],
                         "你要接入的知识库在哪个目录？请提供完整路径，也可以选择稍后配置。")

    def test_preview_does_not_read_text_or_publish(self):
        self.note.write_bytes(b"\xff\xfe binary invalid utf8")
        snapshot = manifest(self.config.database)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("preview must not read text")):
            result = self.preview()
        self.assertEqual(result["file_count"], 1)
        self.assertFalse(saved_config_path(self.config).exists())
        self.assertEqual(manifest(self.config.database), snapshot)

    def test_connect_immediate_and_restart_with_preserved_base(self):
        preview = self.preview()
        result = self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], True)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["documents"], 1)
        current = active_config(self.config)
        self.assertEqual({s.id for s in current.sources}, {"base", "new"})
        self.assertEqual(Onboarding(self.config).status()["status"], "ready")
        self.assertEqual(Retriever(current).search("Onboarding", ["new"])["hits"][0]["doc_id"], "new:note.md")
        self.assertEqual(Retriever(current).search("Original", ["base"])["hits"][0]["doc_id"], "base:old.md")
        self.assertEqual(len(manifest(self.config.database)), 1)

    def test_confirmation_and_changed_preview_required(self):
        preview = self.preview()
        with self.assertRaisesRegex(ValueError, "confirmation"):
            self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], False)
        self.note.write_text("changed scope or content metadata", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "preview changed"):
            self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], True)
        self.assertFalse(saved_config_path(self.config).exists())

    def test_index_failure_preserves_previous_active_config(self):
        preview = self.preview()
        self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], True)
        old = active_config(self.config)
        saved = saved_config_path(self.config).read_bytes()
        self.note.write_text("---\nunterminated: true\n", encoding="utf-8")
        preview = self.preview()
        with self.assertRaises(ValueError):
            self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], True)
        self.assertEqual(saved_config_path(self.config).read_bytes(), saved)
        self.assertEqual(Retriever(old).search("Onboarding", ["new"])["hits"][0]["doc_id"], "new:note.md")

    def test_scope_changes_during_index_not_published(self):
        original = ingest
        def changing(config):
            result = original(config)
            (self.root / "incoming" / "extra.md").write_text("unexpected", encoding="utf-8")
            return result
        preview = self.preview()
        with patch("knowledge_rag.onboarding.ingest", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "directory changed"):
                self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], True)
        self.assertFalse(saved_config_path(self.config).exists())

    def test_invalid_yaml_reports_actionable_error_without_publishing(self):
        self.note.write_text("---\ntags: [broken\n---\n# Text\n", encoding="utf-8")
        preview = self.preview()
        with self.assertRaisesRegex(ValueError, "invalid YAML frontmatter"):
            self.onboarding.connect("new", str(self.root / "incoming"), preview["preview_id"], True)
        self.assertFalse(saved_config_path(self.config).exists())

    def test_conflicts_empty_oversized_and_missing_paths(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.onboarding.preview("new", "relative")
        with self.assertRaises(ValueError):
            self.onboarding.preview("new", str(self.root / "missing"))
        with self.assertRaisesRegex(ValueError, "id already exists"):
            self.onboarding.preview("base", str(self.root / "incoming"))
        with self.assertRaisesRegex(ValueError, "another knowledge"):
            self.onboarding.preview("alias", str(self.root / "base"))
        (self.root / "empty").mkdir()
        self.assertFalse(self.onboarding.preview("empty", str(self.root / "empty"))["can_connect"])
        limited = Onboarding(replace(self.config, max_file_bytes=1))
        self.assertFalse(limited.preview("new", str(self.root / "incoming"))["can_connect"])
