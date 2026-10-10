from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .benchmark import compare_reports, run_benchmark
from .config import load_config
from .index import ingest
from .retrieval import Retriever
from .onboarding import Onboarding, active_config


def main():
    # CLI JSON must be UTF-8 even when Windows inherits a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="knowledge-rag")
    parser.add_argument("--config", type=Path, default=Path("config.example.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    sub.add_parser("list")
    sub.add_parser("serve")
    sub.add_parser("setup-status")
    sub.add_parser("install")
    vector_check = sub.add_parser("vector-check")
    vector_check.add_argument("--output", type=Path)
    preview = sub.add_parser("preview")
    preview.add_argument("kb_id")
    preview.add_argument("root")
    preview.add_argument("--include", action="append")
    preview.add_argument("--exclude", action="append")
    setup = sub.add_parser("connect")
    setup.add_argument("kb_id")
    setup.add_argument("root")
    setup.add_argument("--preview-id", required=True)
    setup.add_argument("--confirmed", action="store_true")
    setup.add_argument("--include", action="append")
    setup.add_argument("--exclude", action="append")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--kb", action="append")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--max-chars", type=int, default=4000)
    search.add_argument("--strategy", choices=["bm25", "overlap"], default="bm25")
    search.add_argument("--filters", type=json.loads)
    read = sub.add_parser("read")
    read.add_argument("doc_id")
    read.add_argument("--source-hash")
    read.add_argument("--start-line", type=int, default=1)
    read.add_argument("--max-chars", type=int, default=4000)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/smoke-v1.jsonl"))
    bench.add_argument("--output", type=Path, default=Path("benchmarks/runs"))
    bench.add_argument("--strategy", choices=["bm25", "overlap"], default="bm25")
    bench.add_argument("--top-k", type=int, default=5)
    bench.add_argument("--max-chars", type=int, default=4000)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--warmup", type=int, default=1)
    compare = sub.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            result = compare_reports(args.baseline, args.candidate)
            if args.output:
                with args.output.open("x", encoding="utf-8") as file:
                    json.dump(result, file, ensure_ascii=False, indent=2)
        else:
            config = load_config(args.config)
            # Benchmarks intentionally use the explicit source configuration, not saved personal connections.
            if args.command in ("index", "list", "search", "read"):
                config = active_config(config)
            if args.command == "vector-check":
                from .vector_check import run_vector_check
                result = run_vector_check(args.output or config.database.parent / "vector-checks")
            elif args.command == "install":
                from .installation import install_codex
                result = install_codex(config)
            elif args.command == "setup-status":
                result = Onboarding(config).status()
            elif args.command == "preview":
                result = Onboarding(config).preview(args.kb_id, args.root, args.include, args.exclude)
            elif args.command == "connect":
                result = Onboarding(config).connect(args.kb_id, args.root, args.preview_id,
                                                     args.confirmed, args.include, args.exclude)
            elif args.command == "index":
                result = ingest(config)
            elif args.command == "list":
                result = Retriever(config).list_knowledge_bases()
            elif args.command == "search":
                result = Retriever(config, args.strategy).search(args.query, args.kb, args.top_k, args.max_chars, args.filters)
            elif args.command == "read":
                result = Retriever(config).read_document(args.doc_id, args.source_hash, args.start_line, args.max_chars)
            elif args.command == "benchmark":
                result = {"report": str(run_benchmark(config, args.dataset, args.output, args.strategy,
                                                      args.top_k, args.max_chars, args.repeats, args.warmup))}
            else:
                from .server import create_server
                create_server(config).run(transport="stdio")
                return
    except (ValueError, OSError, subprocess.SubprocessError, TimeoutError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
