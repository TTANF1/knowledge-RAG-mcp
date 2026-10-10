from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

from .config import Config
from .index import ingest, manifest
from .retrieval import Retriever, TOKENIZER_VERSION
from .telemetry import canonical, digest

BENCHMARK_VERSION = "evidence-visible-v1"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def evaluate(case: dict, hits: list[dict], k: int) -> dict:
    """Credit only an expected document with its evidence visible in the returned excerpt."""
    expected = {q["doc_id"]: q["contains"] for q in case["relevant"]}
    found, gains = set(), []
    for hit in hits[:k]:
        doc_id = hit["doc_id"]
        match = doc_id in expected and doc_id not in found and expected[doc_id] in hit["text"]
        gains.append(int(match))
        if match:
            found.add(doc_id)
    ranks = [i + 1 for i, gain in enumerate(gains) if gain]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return {"recall_at_k": len(found) / len(expected) if expected else None,
            "hit_at_k": float(bool(found)) if expected else None,
            "mrr_at_k": 1 / ranks[0] if ranks else (0.0 if expected else None),
            "ndcg_at_k": dcg / ideal if ideal else None,
            "precision_at_k": len(found) / k if expected else None,
            "empty_on_unanswerable": float(not hits) if not expected else None,
            "empty_on_answerable": float(not hits) if expected else None,
            "kb_leakage": sum(h["kb_id"] not in case["kb_ids"] for h in hits) if case.get("kb_ids") else 0}


def load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("benchmark cases must have unique ids and cannot be empty")
    for case in cases:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("case requires a query")
        if "category" not in case or not isinstance(case.get("relevant"), list):
            raise ValueError("case requires category and relevant list (empty for unanswerable)")
        ids = [q["doc_id"] for q in case["relevant"]]
        if len(ids) != len(set(ids)) or any(not q.get("contains") for q in case["relevant"]):
            raise ValueError("relevance must have unique document ids and nonempty evidence")
    return cases


def aggregate(rows: list[dict]) -> dict:
    metrics = {}
    for key in rows[0]["metrics"]:
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        metrics[key] = statistics.mean(values) if values else None
    latencies = [value for r in rows for value in r["latencies_ms"]]
    metrics.update({"queries": len(rows), "latency_p50_ms": percentile(latencies, 0.5),
                    "latency_p95_ms": percentile(latencies, 0.95),
                    "mean_content_chars": statistics.mean(r["content_chars"] for r in rows),
                    "mean_response_json_bytes": statistics.mean(r["response_json_bytes"] for r in rows)})
    return metrics


def provenance(root: Path) -> dict:
    files = sorted([*root.glob("src/**/*.py"), root / "pyproject.toml", root / "uv.lock"])
    code = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in files if p.is_file()}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                  capture_output=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True,
                                    capture_output=True, timeout=5).stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        revision, dirty = None, None
    return {"git_commit": revision, "git_dirty": dirty, "code_hash": digest(canonical(code)),
            "file_hashes": code, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"]))}


def run_benchmark(config: Config, dataset: Path, output: Path, strategy: str = "bm25",
                  top_k: int = 5, max_chars: int = 4000, repeats: int = 5, warmup: int = 1) -> Path:
    if repeats < 1 or warmup < 0 or not 1 <= top_k <= 50 or not 1 <= max_chars <= 20000:
        raise ValueError("invalid benchmark parameters")
    cases = load_cases(dataset)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = replace(config, traces=run_dir / "traces.jsonl")
    if config.vector:
        run_config = replace(run_config, vector=replace(config.vector, root=run_dir / "vectors"))
    rows = []
    vector_build = None
    config.database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="knowledge-rag-benchmark-", dir=config.database.parent) as temp:
        run_config = replace(run_config, database=Path(temp) / "index.db")
        indexing = ingest(run_config)
        snapshot = manifest(run_config.database)
        from .index import connect
        from contextlib import closing
        with closing(connect(run_config.database)) as db:
            corpus = {r["id"]: r for r in db.execute("SELECT id,kb_id,content,metadata FROM documents")}
        for case in cases:
            for label in case["relevant"]:
                doc = corpus.get(label["doc_id"])
                if doc is None or label["contains"] not in doc["content"]:
                    raise ValueError(f"invalid evidence label: {case['id']}")
                if case.get("kb_ids") and doc["kb_id"] not in case["kb_ids"]:
                    raise ValueError(f"label outside selected knowledge bases: {case['id']}")
                metadata = json.loads(doc["metadata"])
                if any(metadata.get(k) != v for k, v in case.get("filters", {}).items()):
                    raise ValueError(f"label conflicts with metadata filter: {case['id']}")
        if strategy == "dense":
            from .dense_index import build_vector
            vector_build = build_vector(run_config)
        retriever = Retriever(run_config, strategy)
        # Seeded order reduces systematic order bias. Each case has warmups and repeated timings.
        order = list(cases)
        random.Random(42).shuffle(order)
        for case in order:
            kwargs = {"query": case["query"], "kb_ids": case.get("kb_ids"), "top_k": top_k,
                      "max_chars": max_chars, "filters": case.get("filters")}
            for _ in range(warmup):
                retriever.search(**kwargs)
            timings, trace_ids, outputs = [], [], []
            for _ in range(repeats):
                start = perf_counter()
                result = retriever.search(**kwargs)
                timings.append((perf_counter() - start) * 1000)
                trace_ids.append(result["trace_id"])
                outputs.append([(h["chunk_id"], h["score"], h["text"]) for h in result["hits"]])
            if any([(x[0], x[2]) for x in value] != [(x[0], x[2]) for x in outputs[0]]
                   or any(not math.isclose(x[1], y[1], rel_tol=1e-6, abs_tol=1e-7) for x, y in zip(value, outputs[0]))
                   for value in outputs[1:]):
                raise ValueError("nondeterministic retrieval; benchmark needs repeated quality aggregation")
            rows.append({"id": case["id"], "category": case["category"], "trace_ids": trace_ids,
                         "metrics": evaluate(case, result["hits"], top_k), "latencies_ms": timings,
                         "content_chars": result["content_chars"],
                         "response_json_bytes": len(canonical(result).encode("utf-8")),
                         "hits": [{k: v for k, v in hit.items() if k not in ("text", "metadata")}
                                  for hit in result["hits"]]})
    rows.sort(key=lambda row: row["id"])
    measured_ids = {tid for row in rows for tid in row["trace_ids"]}
    records = [json.loads(line) for line in run_config.traces.read_text(encoding="utf-8").splitlines()]
    measured = [record for record in records if record["trace_id"] in measured_ids]
    stages = {name: [record["stages_ms"].get(name, 0.0) for record in measured]
              for name in sorted({name for record in measured for name in record["stages_ms"]})}
    report = {"schema_version": 1, "benchmark_version": BENCHMARK_VERSION,
              "run_id": run_id, "dataset_hash": digest(canonical(cases)),
              "corpus_hash": digest(canonical(snapshot)), "corpus_manifest": snapshot,
              "config_hash": digest(config.path.read_bytes()),
              "settings": {"strategy": strategy, "top_k": top_k, "max_chars": max_chars,
                           "repeats": repeats, "warmup": warmup, "chunk_chars": config.chunk_chars,
                           "tokenizer": TOKENIZER_VERSION, "bm25_k1": 1.5, "bm25_b": 0.75,
                           "timing_scope": "in-process search including trace append; excluding MCP transport"},
              "environment": provenance(Path(__file__).resolve().parents[2]),
              "indexing": indexing, "metrics": aggregate(rows),
              "stages_ms": {name: {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}
                            for name, values in stages.items()},
              "by_category": {category: aggregate([r for r in rows if r["category"] == category])
                              for category in sorted({r["category"] for r in rows})},
              "per_query": rows,
              "limitations": ["Synthetic development set; not representative of personal knowledge quality.",
                              "No answer generation: faithfulness and hallucination are not measured.",
                              "Character/byte counts are not model tokens or monetary cost.",
                              "Small warm-cache in-process timings are not production capacity measurements."]}
    if vector_build:
        report["vector_build"] = vector_build
        report["settings"]["embedding_spec"] = vector_build["embedding_spec"]
        report["settings"]["vector_mode"] = config.vector.mode
        report["settings"]["query_cache"] = config.embedding.query_cache
        report["settings"]["exact"] = True
        report["settings"]["batch_size"] = config.embedding.batch_size
        report["settings"]["threads"] = config.embedding.threads
        report["settings"]["token_window"] = "complete-character-coverage-v1"
        report["settings"]["score_tolerance"] = {"relative": 1e-6, "absolute": 1e-7}
        report["embedding_queries"] = {key: sum(record.get("embedding", {}).get(key, 0) for record in measured)
                                       for key in ("hits", "misses", "encoded_tokens", "encode_calls", "encode_ms")}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "dataset.jsonl").write_text("".join(canonical(c) + "\n" for c in cases), encoding="utf-8")
    summary = [f"# Benchmark {run_id}", "", f"Strategy: {strategy}; corpus: {report['corpus_hash']}", "",
               "| Metric | Value |", "| --- | ---: |"]
    summary += [f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |"
                for key, value in report["metrics"].items()]
    summary += ["", "## Limits", *[f"- {s}" for s in report["limitations"]]]
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return run_dir / "report.json"


def compare_reports(baseline: Path, candidate: Path) -> dict:
    before, after = [json.loads(p.read_text(encoding="utf-8")) for p in (baseline, candidate)]
    for key in ("schema_version", "benchmark_version", "dataset_hash", "corpus_hash"):
        if before[key] != after[key]:
            raise ValueError(f"incomparable reports: {key} changed")
    for key in ("top_k", "max_chars", "repeats", "warmup", "timing_scope"):
        if before["settings"][key] != after["settings"][key]:
            raise ValueError(f"incomparable reports: {key} changed")
    deltas = {key: after["metrics"][key] - value for key, value in before["metrics"].items()
              if isinstance(value, (float, int)) and isinstance(after["metrics"].get(key), (float, int))}
    previous = {r["id"]: r for r in before["per_query"]}
    if set(previous) != {r["id"] for r in after["per_query"]}:
        raise ValueError("incomparable query ids")
    changes, paired = [], []
    for row in after["per_query"]:
        old = previous[row["id"]]
        if row["metrics"]["recall_at_k"] is not None:
            difference = row["metrics"]["recall_at_k"] - old["metrics"]["recall_at_k"]
            paired.append(difference)
        metric_changes = {key: value - old["metrics"][key] for key, value in row["metrics"].items()
                          if value is not None and old["metrics"][key] is not None and value != old["metrics"][key]}
        if metric_changes:
            changes.append({"id": row["id"], "deltas": metric_changes})
    rng = random.Random(42)
    samples = [statistics.mean(rng.choices(paired, k=len(paired))) for _ in range(2000)] if paired else []
    warnings = ["Development-set deltas are descriptive; a small paired bootstrap is not proof of generalization."]
    for key in ("python", "platform", "machine", "processor"):
        if before["environment"][key] != after["environment"][key]:
            warnings.append(f"Environment {key} changed; latency comparison is not controlled.")
    return {"baseline": before["run_id"], "candidate": after["run_id"],
            "delta_candidate_minus_baseline": deltas, "query_changes": changes,
            "recall_delta_bootstrap_95pct": [percentile(samples, 0.025), percentile(samples, 0.975)] if samples else None,
            "warnings": warnings}
