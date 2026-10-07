from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .config import Config
from .telemetry import Trace


def install_codex(config: Config) -> dict:
    """Register the local service, verify its protocol, and return the current-chat next action."""
    try:
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as error:
        raise ValueError("install the MCP extra first: uv sync --locked --extra mcp") from error

    command = str(Path(sys.executable).resolve())
    args = ["-m", "knowledge_rag.cli", "--config", str(config.path), "serve"]
    with Trace(config.traces, "install_codex") as trace:
        with trace.stage("register"):
            existing = subprocess.run(["codex", "mcp", "get", "knowledge-rag-mcp", "--json"],
                                      capture_output=True, text=True, timeout=20)
            if existing.returncode == 0:
                entry = json.loads(existing.stdout)
                transport = entry["transport"]
                if (transport["type"] != "stdio" or Path(transport["command"]).resolve() != Path(command)
                        or transport["args"] != args or not entry["enabled"]):
                    raise ValueError("existing knowledge-rag-mcp registration differs; resolve it before installing")
                registration = "already_registered"
            else:
                if "No MCP server named" not in existing.stderr:
                    raise ValueError("cannot inspect Codex MCP configuration: " + existing.stderr.strip())
                result = subprocess.run(["codex", "mcp", "add", "knowledge-rag-mcp", "--", command, *args],
                                        capture_output=True, text=True, timeout=20)
                if result.returncode:
                    raise ValueError("Codex MCP registration failed: " + result.stderr.strip())
                registration = "registered"

        async def verify():
            with anyio.fail_after(30):
                async with stdio_client(StdioServerParameters(command=command, args=args)) as (read, write):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        tools = sorted(t.name for t in (await session.list_tools()).tools)
                        required = {"get_setup_status", "preview_knowledge_base", "connect_knowledge_base",
                                    "list_knowledge_bases", "search_knowledge", "read_document"}
                        if set(tools) != required:
                            raise ValueError("installed MCP tool list does not match this project version")
                        status = await session.call_tool("get_setup_status", {})
                        if status.is_error:
                            raise ValueError("MCP setup-status verification failed")
                        return json.loads(status.content[0].text), tools, initialized.instructions

        with trace.stage("verify_mcp"):
            status, tools, instructions = anyio.run(verify)
        return {"installation": registration, "protocol_verification": "passed", "tools": tools,
                "setup": status, "agent_instruction": (
                    "Ask setup.question in the current conversation now and wait for the user's directory; "
                    "then preview, show scope, confirm, connect and test a scoped search. "
                    "Do not end this task at installation success. Respect a request to configure later."
                    if status["status"] == "needs_knowledge_base" else "Follow setup.next_action; no onboarding question is needed."),
                "server_instructions": instructions, "trace_id": trace.record["trace_id"]}
