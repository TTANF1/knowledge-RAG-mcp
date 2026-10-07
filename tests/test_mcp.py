import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from knowledge_rag.config import load_config
from knowledge_rag.index import ingest


@unittest.skipUnless(importlib.util.find_spec("mcp"), "install the mcp extra for protocol integration tests")
class MCPIntegrationTests(unittest.TestCase):
    def test_stdio_initialize_discover_search_read_and_validation(self):
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "note.md").write_text("# Atlas\nA verified demonstration.\n", encoding="utf-8")
            config_path = root / "config.toml"
            config_path.write_text('[[sources]]\nid = "demo"\nroot = "docs"\n', encoding="utf-8")
            ingest(load_config(config_path))

            async def exercise():
                params = StdioServerParameters(command=sys.executable,
                    args=["-m", "knowledge_rag.cli", "--config", str(config_path), "serve"])
                with anyio.fail_after(30):
                    async with stdio_client(params) as (read, write):
                        async with ClientSession(read, write) as session:
                            initialized = await session.initialize()
                            self.assertIn("THIS conversation", initialized.instructions)
                            names = {tool.name for tool in (await session.list_tools()).tools}
                            self.assertEqual(names, {"list_knowledge_bases", "search_knowledge", "read_document",
                                                     "get_setup_status", "preview_knowledge_base", "connect_knowledge_base"})
                            response = await session.call_tool("search_knowledge", {"query": "Atlas", "kb_ids": ["demo"]})
                            self.assertFalse(response.is_error)
                            payload = json.loads(response.content[0].text)
                            hit = payload["hits"][0]
                            self.assertEqual(hit["doc_id"], "demo:note.md")
                            response = await session.call_tool("read_document", {"doc_id": hit["doc_id"], "source_hash": hit["source_hash"]})
                            self.assertFalse(response.is_error)
                            self.assertIn("verified demonstration", json.loads(response.content[0].text)["text"])
                            invalid = await session.call_tool("search_knowledge", {"query": "Atlas", "top_k": 0})
                            self.assertTrue(invalid.is_error)
                            self.assertIn("top_k must be", invalid.content[0].text)
                            incoming = root / "incoming"
                            incoming.mkdir()
                            (incoming / "new.md").write_text("# Fresh\nNew connected evidence.\n", encoding="utf-8")
                            previewed = await session.call_tool("preview_knowledge_base", {"kb_id": "new", "root": str(incoming)})
                            self.assertFalse(previewed.is_error)
                            preview = json.loads(previewed.content[0].text)
                            rejected = await session.call_tool("connect_knowledge_base", {
                                "kb_id": "new", "root": str(incoming), "preview_id": preview["preview_id"], "confirmed": False})
                            self.assertTrue(rejected.is_error)
                            connected = await session.call_tool("connect_knowledge_base", {
                                "kb_id": "new", "root": str(incoming), "preview_id": preview["preview_id"], "confirmed": True})
                            self.assertFalse(connected.is_error)
                            searched = await session.call_tool("search_knowledge", {"query": "Fresh", "kb_ids": ["new"]})
                            self.assertFalse(searched.is_error)
                            self.assertEqual(json.loads(searched.content[0].text)["hits"][0]["doc_id"], "new:new.md")

            anyio.run(exercise)
