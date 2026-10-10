"""Repeatable real-provider contract checks with public synthetic text only."""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .embedding_providers import CachedProvider, document_windows, get_provider
from .resources import process_memory
from .telemetry import digest


def run_embedding_check(config) -> dict:
    if not config.embedding:
        raise ValueError("embedding configuration is required")
    folder = config.embedding.cache.parent / "embedding-checks" / uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = {"schema_version": 1, "kind": "embedding-contract-check", "status": "failed", "checks": []}
    try:
        loaded = get_provider(config.embedding)
        provider = CachedProvider(loaded.provider, folder / "cache.db", query_cache=True)
        text = "中文计划：摄影与学习。\n" * provider.spec.max_tokens
        units = document_windows(provider, text, "合成测试")
        if (len(units) < 2 or units[-1].end != len(text)
                or "".join(unit.text[len("合成测试\n"):] for unit in units) != text
                or any(unit.tokens > provider.spec.max_tokens for unit in units)):
            raise ValueError("embedding window coverage check failed")
        report["checks"] += ["long_text_complete_coverage", "full_template_token_limit"]
        vectors = provider.encode_documents([unit.text for unit in units])
        for vector in vectors:
            provider.spec.validate_vector(vector)
        report["checks"] += ["dimension_finite_and_normalized"]
        before = provider.stats.copy()
        provider.encode_documents([unit.text for unit in units])
        if provider.stats["encode_calls"] != before["encode_calls"]:
            raise ValueError("warm document cache check failed")
        report["checks"] += ["warm_documents_zero_encoding"]
        provider.encode_query("摄影计划")
        before_query = provider.stats["encode_calls"]
        provider.encode_query("摄影计划")
        if provider.stats["encode_calls"] != before_query:
            raise ValueError("warm query cache check failed")
        report["checks"] += ["warm_query_zero_encoding"]
        try:
            provider.encode_query(text)
        except ValueError:
            report["checks"].append("overlong_query_rejected")
        else:
            raise ValueError("overlong query was accepted")
        report.update(status="passed", fixture_hash=digest(text), units=len(units),
                      maximum_tokens=max(unit.tokens for unit in units), embedding_spec=provider.spec.to_dict(),
                      encoding=provider.stats, model_load_ms=provider.load_ms)
    except Exception as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["code_hashes"] = {name: digest(Path(__file__).with_name(name).read_bytes()) for name in (
            "embeddings.py", "embedding_providers.py", "embedding_check.py", "config.py")}
        report["settings"] = {"provider": config.embedding.provider, "batch_size": config.embedding.batch_size,
                              "threads": config.embedding.threads, "query_cache": True}
        report["versions"] = {}
        for package in ("onnxruntime", "tokenizers", "httpx", "tiktoken"):
            try:
                report["versions"][package] = version(package)
            except PackageNotFoundError:
                pass
        report["elapsed_ms"] = (perf_counter() - started) * 1000
        report["process_memory"] = process_memory()
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"status": report["status"], "passed_count": len(report["checks"]), "report": str(folder / "report.json")}
