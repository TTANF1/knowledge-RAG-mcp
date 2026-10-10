"""Build immutable SQLite/vector generations and publish one shared pointer."""
from contextlib import closing
from dataclasses import replace
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .config import Config
from .index import connect, ingest, manifest, source_files, PARSER_VERSION
from .telemetry import Trace, canonical, digest
from .vector_index import VectorManifest
from .vector_store import QdrantVectorStore, VectorPoint
from .resources import process_memory

QUERY_LOCK = RLock()


def scope_hash(config):
    return digest(canonical([str(config.database), config.chunk_chars, config.max_file_bytes,
        [(s.id, str(s.root), s.include, s.exclude) for s in config.sources],
        [config.vector.mode, config.vector.url] if config.vector else None,
        PARSER_VERSION, "complete-character-coverage-v1"]))


def source_versions(config):
    return sorted((f"{s.id}:{path.relative_to(s.root).as_posix()}", digest(path.read_bytes()))
                  for s in config.sources for path in source_files(s))


def load_active(config: Config) -> dict | None:
    if not config.vector:
        return None
    pointer = config.vector.root / "active.json"
    if not pointer.exists():
        return None
    data = json.loads(pointer.read_text(encoding="utf-8"))
    if data.get("scope_hash") != scope_hash(config):
        return None
    generation = data.get("generation")
    if not isinstance(generation, str) or len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
        raise ValueError("invalid active vector generation")
    folder = config.vector.root / "generations" / generation
    vector_manifest = VectorManifest.load(folder / "manifest.json")
    if (data.get("contract_id") != vector_manifest.contract_id or generation != vector_manifest.generation
            or not (folder / "index.db").is_file()):
        raise ValueError("active vector generation is incomplete")
    return {**data, "folder": folder, "manifest": vector_manifest, "database": folder / "index.db"}


def snapshot_config(config: Config) -> Config:
    active = load_active(config)
    return replace(config, database=active["database"]) if active else config


def store_for(config, active):
    if config.vector.mode == "local":
        return QdrantVectorStore(active["manifest"], mode="local", path=active["folder"] / "qdrant", traces=config.traces)
    return QdrantVectorStore(active["manifest"], mode="server", url=config.vector.url,
                            api_key=os.environ.get(config.vector.api_key_env), traces=config.traces)


def require_embedding(config):
    if not config.embedding or not config.vector:
        raise ValueError("embedding and vector configuration are required")
    from .embedding_providers import get_provider
    return get_provider(config.embedding)


def vector_status(config: Config) -> dict:
    active = load_active(config)
    result = {"status": "not_built", "has_ready_generation": bool(active)}
    if active:
        spec = active["manifest"].spec
        settings = embedding_settings_hash(config) if config.embedding else None
        fresh = tuple(source_versions(config)) == active["manifest"].source_versions
        result.update(status="ready" if fresh and settings == active["embedding_settings_hash"] else "stale",
                      generation=active["generation"], contract_id=active["contract_id"],
                      model=spec.model_id, revision=spec.revision, dimension=spec.dimension,
                      documents=len(active["manifest"].source_versions), points=active["points"],
                      mode=config.vector.mode)
    if config.vector and (config.vector.root / "last-build.json").exists():
        attempt = json.loads((config.vector.root / "last-build.json").read_text(encoding="utf-8"))
        result["last_build_status"] = attempt["status"]
        if attempt["status"] == "failed":
            result["status"] = "failed"
    return result


def embedding_settings_hash(config):
    values = config.embedding.__dict__ | {"model_dir": str(config.embedding.model_dir), "cache": str(config.embedding.cache)}
    return digest(canonical(values))


def build_vector(config: Config) -> dict:
    from .embedding_providers import CachedProvider, document_windows
    if not config.embedding or not config.vector:
        raise ValueError("embedding and vector configuration are required")
    import portalocker
    root = config.vector.root
    root.mkdir(parents=True, exist_ok=True)
    try:
        lock = portalocker.Lock(str(root / "build.lock"), timeout=0)
        lock.acquire()
    except portalocker.exceptions.LockException:
        raise ValueError("another vector build is running") from None
    generation = uuid4().hex
    folder = root / "generations" / generation
    folder.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "generation": generation, "status": "failed"}
    try:
        with Trace(config.traces, "vector_build", generation=generation) as trace:
            with trace.stage("model_load"):
                provider = require_embedding(config)
                # Reuse the loaded model, but keep this build's counters separate from serving queries.
                provider = CachedProvider(provider.provider, config.embedding.cache, config.embedding.query_cache)
            stats_before = provider.stats.copy()
            with trace.stage("snapshot"):
                candidate = replace(config, database=folder / "index.db")
                report["indexing"] = ingest(candidate)
                sources = tuple((row["id"], row["source_hash"]) for row in manifest(candidate.database))
                vector_manifest = VectorManifest(provider.spec, generation, sources)
                with closing(connect(candidate.database)) as db:
                    rows = [dict(row) for row in db.execute("""SELECT c.*, d.kb_id,d.title,d.metadata,d.source_hash
                        FROM chunks c JOIN documents d ON c.doc_id=d.id ORDER BY c.id""")]
            active = {"manifest": vector_manifest, "folder": folder}
            points_count, tokens, split_chunks = 0, 0, 0
            with store_for(config, active) as store:
                store.create()
                for row in rows:
                    with trace.stage("token_windows"):
                        units = document_windows(provider, row["text"], row["title"] + "\n" + row["heading"])
                        split_chunks += len(units) > 1
                        tokens += sum(unit.tokens for unit in units)
                    with trace.stage("encode"):
                        vectors = provider.encode_documents([unit.text for unit in units])
                    with trace.stage("upsert"):
                        points = [VectorPoint(row["kb_id"], row["doc_id"], row["id"],
                                  digest(canonical([row["id"], unit.start, unit.end, provider.spec.fingerprint])),
                                  row["source_hash"], vector, json.loads(row["metadata"]))
                                  for unit, vector in zip(units, vectors)]
                        points_count += store.upsert(points)
                with trace.stage("verify"):
                    if store.count() != points_count or tuple(source_versions(config)) != sources:
                        raise ValueError("vector coverage or source versions changed during build")
            vector_manifest.save(folder / "manifest.json")
            stats = {key: provider.stats[key] - stats_before[key] for key in stats_before}
            report.update(status="ready", documents=len(sources), chunks=len(rows), points=points_count,
                          split_chunks=split_chunks, input_tokens=tokens, encoding=stats,
                          model_load_ms=provider.load_ms, embedding_spec=provider.spec.to_dict(),
                          mode=config.vector.mode, contract_id=vector_manifest.contract_id)
            pointer = {"schema_version": 1, "generation": generation, "contract_id": vector_manifest.contract_id,
                       "scope_hash": scope_hash(config), "embedding_settings_hash": embedding_settings_hash(config),
                       "points": points_count}
            with trace.stage("publish"):
                temporary = root / (".active-" + generation + ".json")
                try:
                    with temporary.open("x", encoding="utf-8") as file:
                        file.write(canonical(pointer))
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary, root / "active.json")
                finally:
                    temporary.unlink(missing_ok=True)
            trace.record["counts"] = {"documents": len(sources), "chunks": len(rows), "points": points_count,
                                      "input_tokens": tokens, **stats}
        report["stages_ms"] = trace.record["stages_ms"]
        report["operation_ms"] = trace.record["operation_ms"]
        report["storage_bytes"] = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        report["process_memory"] = process_memory()
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        if "trace" in locals():
            report["stages_ms"] = trace.record["stages_ms"]
            report["operation_ms"] = trace.record["operation_ms"]
        raise
    finally:
        try:
            (folder / "build-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (root / "last-build.json").write_text(canonical({"generation": generation, "status": report["status"]}), encoding="utf-8")
        finally:
            lock.release()
    return report


def dense_rank(config, rows, query, kb_ids, filters, top_k, trace, active):
    with QUERY_LOCK:
        with trace.stage("vector_state"):
            if not active:
                raise ValueError("vector index is not built for this source configuration; run vector-build")
            provider = require_embedding(config)
            if active["manifest"].spec != provider.spec:
                raise ValueError("embedding model changed; rebuild vector index before dense search")
            if not rows:
                return []
        with trace.stage("query_embed"):
            before = provider.stats.copy()
            vector = provider.encode_query(query)
            trace.record["embedding"] = {key: provider.stats[key] - before[key] for key in before}
        with trace.stage("vector_query"):
            with store_for(config, active) as store:
                # Each long parent can have multiple units; fetch enough to return top_k distinct parents.
                limit = min(50, max(top_k, top_k * 4))
                hits = store.search(vector, kb_ids, filters, top_k=limit)
                page = hits
                while len(page) == limit and len({h.parent_chunk_id for h in hits}) < top_k:
                    page = store.search(vector, kb_ids, filters, top_k=limit, offset=len(hits))
                    hits.extend(page)
        with trace.stage("source_resolve"):
            parents = {row["id"]: row for row in rows}
            selected = {}
            for hit in hits:
                row = parents.get(hit.parent_chunk_id)
                if (row is None or row["doc_id"] != hit.doc_id or row["source_hash"] != hit.source_hash
                        or row["kb_id"] != hit.kb_id):
                    raise ValueError("vector citation does not match the active SQLite snapshot")
                selected[hit.parent_chunk_id] = max(selected.get(hit.parent_chunk_id, float("-inf")), hit.score)
            trace.record["query_model_spec_hash"] = provider.spec.fingerprint
            return sorted(((score, parents[key]) for key, score in selected.items()), key=lambda item: (-item[0], item[1]["id"]))
