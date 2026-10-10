"""Export aggregate results only after verifying the exact public synthetic corpus."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def export(root: Path, source: Path, destination: Path) -> Path:
    report = json.loads(source.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (root / "benchmarks/datasets/smoke-v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = []
    for kb in ("learning", "product"):
        base = root / "examples/knowledge" / kb
        for path in sorted(base.rglob("*.md")):
            expected.append({"id": f"{kb}:{path.relative_to(base).as_posix()}",
                             "source_hash": sha256(path.read_bytes()).hexdigest()})
    expected.sort(key=lambda item: item["id"])
    if (report.get("dataset_hash") != fingerprint(cases) or report.get("corpus_manifest") != expected
            or report.get("corpus_hash") != fingerprint(expected)):
        raise ValueError("only the exact approved public smoke-v1 corpus can be exported")
    allowed_settings = ("strategy", "top_k", "max_chars", "repeats", "warmup", "chunk_chars",
                        "tokenizer", "bm25_k1", "bm25_b", "timing_scope")
    public = {"schema_version": 1, "kind": "synthetic-benchmark-summary",
              "provenance": {"dataset": "smoke-v1", "dataset_hash": report["dataset_hash"],
                             "corpus_hash": report["corpus_hash"], "code_hash": report["environment"]["code_hash"],
                             "python": report["environment"]["python"]},
              "settings": {key: report["settings"][key] for key in allowed_settings},
              "metrics": report["metrics"], "by_category": report["by_category"],
              "stages_ms": report.get("stages_ms", {}),
              "limitations": ["Approved synthetic development corpus only.",
                              "No private source ids, paths, queries, citations, per-query rows or traces are exported.",
                              "Small warm in-process measurements are not production performance evidence."]}
    if report["settings"].get("strategy") == "dense":
        spec_keys = {"provider", "model_id", "revision", "dimension", "max_tokens", "template_version", "normalize", "distance"}
        spec = report["settings"].get("embedding_spec", {})
        if set(spec) != spec_keys:
            raise ValueError("dense report must have an embedding specification without extra fields")
        for key in ("embedding_spec", "vector_mode", "query_cache", "exact", "batch_size", "threads", "token_window", "score_tolerance"):
            public["settings"][key] = report["settings"][key]
        build = report.get("vector_build", {})
        public["vector_build"] = {key: build[key] for key in (
            "documents", "chunks", "points", "split_chunks", "input_tokens", "encoding", "model_load_ms",
            "operation_ms", "stages_ms", "storage_bytes", "process_memory") if key in build}
        public["embedding_queries"] = {key: report.get("embedding_queries", {}).get(key, 0)
                                       for key in ("hits", "misses", "encoded_tokens", "encode_calls", "encode_ms")}
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "report.json"
    path.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = ["# Public synthetic benchmark summary", "", "| Metric | Value |", "| --- | ---: |"]
    summary += [f"| {key} | {value} |" for key, value in public["metrics"].items()]
    (destination / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(export(Path(__file__).resolve().parents[1], args.source, args.destination))
