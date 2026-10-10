"""Repeatable storage checks with synthetic unit vectors, not an embedding benchmark."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
import platform
from uuid import uuid4

from .embeddings import EmbeddingSpec
from .telemetry import canonical, digest
from .benchmark import percentile
from .vector_index import VectorManifest
from .vector_store import QdrantVectorStore, VectorPoint


def fixture():
    spec = EmbeddingSpec("fixture", "fixed-3d", "v1", 3, 1, "v1")
    versions = tuple((doc, digest(doc)) for doc in ("alpha:first", "alpha:second", "beta:first"))
    manifest = VectorManifest(spec, "fixture-v1", versions)
    points = [VectorPoint(doc.split(":")[0], doc, "chunk-" + str(i), "unit-" + str(i),
                          source_hash, vector, {"status": status})
              for i, ((doc, source_hash), vector, status) in enumerate(zip(versions,
                  ((0.8, 0.6, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                  ("reviewed", "draft", "reviewed")))]
    return manifest, points


def run_vector_check(output: Path) -> dict:
    run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    manifest, points = fixture()
    manifest.save(run / "manifest.json")
    checks = []
    started = perf_counter()

    def check(name, condition):
        if not condition:
            raise ValueError("vector storage check failed: " + name)
        checks.append(name)

    def rejected(name, action):
        try:
            action()
        except ValueError:
            checks.append(name)
        else:
            raise ValueError("vector storage check accepted invalid input: " + name)

    try:
        with QdrantVectorStore(manifest, mode="local", path=run / "qdrant", traces=run / "traces.jsonl") as store:
            check("create", store.create())
            check("idempotent_create", not store.create())
            check("insert", store.upsert(points) == 3 and store.count() == 3)
            check("idempotent_upsert", store.upsert(points) == 3 and store.count() == 3)
            check("global_nearest", store.search((1, 0, 0), top_k=1)[0].kb_id == "beta")
            check("scope_before_top_k", store.search((1, 0, 0), ["alpha"], top_k=1)[0].doc_id == "alpha:first")
            hits = store.search((1, 0, 0), ["alpha"], {"status": "draft"}, top_k=1)
            check("metadata_before_top_k", len(hits) == 1 and hits[0].doc_id == "alpha:second")
            rejected("dimension_rejected", lambda: store.search((1, 0)))
            rejected("source_version_rejected", lambda: store.upsert([replace(points[0], source_hash=digest("stale"))]))
            rejected("invalid_batch_rejected", lambda: store.upsert([
                replace(points[0], embedding_unit_id="new-unit"), replace(points[1], vector=(float("nan"), 0, 0))]))
            check("invalid_batch_no_partial_write", store.count() == 3)
            rejected("unknown_scope_rejected", lambda: store.search((1, 0, 0), ["unknown"]))
        loaded = VectorManifest.load(run / "manifest.json")
        with QdrantVectorStore(loaded, mode="local", path=run / "qdrant", traces=run / "traces.jsonl") as store:
            check("restart_persistence", store.count() == 3 and store.search((1, 0, 0), ["alpha"], top_k=1)[0].doc_id == "alpha:first")
            check("update_same_identity", store.upsert([replace(points[0], vector=(0, 0, 1))]) == 1 and store.count() == 3)
            check("update_visible", store.search((0, 0, 1), ["alpha"], top_k=1)[0].doc_id == "alpha:first")
            check("delete_document", store.delete_documents(["alpha:first"]) == 1 and store.count() == 2)
            check("idempotent_delete", store.delete_documents(["alpha:first"]) == 0)
            check("delete_visible", all(hit.doc_id != "alpha:first" for hit in store.search((1, 0, 0))))
        status = "passed"
    except Exception as error:
        status = "failed"
        failure = type(error).__name__
        raise
    finally:
        try:
            client_version = version("qdrant-client")
        except PackageNotFoundError:
            client_version = None
        traces = [json.loads(line) for line in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()] if (run / "traces.jsonl").exists() else []
        timings = {}
        for trace in traces:
            if trace["status"] != "ok":
                continue
            for stage, duration in {"operation": trace["operation_ms"], **trace["stages_ms"]}.items():
                timings.setdefault(trace["operation"] + "." + stage, []).append(duration)
        report = {"schema_version": 1, "kind": "vector-storage-check", "status": status,
                  "fixture": "fixed-3d-v1", "contract_id": manifest.contract_id,
                  "fixture_hash": digest(canonical([asdict(point) for point in points])),
                  "embedding_spec": manifest.spec.to_dict(), "mode": "local", "exact": True,
                  "versions": {"qdrant_client": client_version, "python": platform.python_version()},
                  "code_hashes": {name: digest(Path(__file__).with_name(name).read_text(encoding="utf-8"))
                                  for name in ("embeddings.py", "vector_index.py", "vector_store.py", "vector_check.py")},
                  "passed_count": len(checks), "checks": checks,
                  "elapsed_ms": (perf_counter() - started) * 1000,
                  "operation_count": len(traces), "expected_rejection_count": sum(t["status"] == "error" for t in traces),
                  "stage_timings_ms": {name: {"samples": len(values), "p50": percentile(values, 0.5),
                                               "p95": percentile(values, 0.95)} for name, values in timings.items()},
                  "storage_bytes": sum(p.stat().st_size for p in (run / "qdrant").rglob("*") if p.is_file()),
                  "limitations": ["fixed vectors; no semantic quality measurement", "local mode only; no server or concurrency benchmark"]}
        if status == "failed":
            report["error_type"] = failure
        (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report": str(run / "report.json"), "status": status, "passed_count": len(checks)}
