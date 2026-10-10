"""Qdrant persistence adapter with explicit source and generation boundaries."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from .telemetry import Trace, canonical
from .vector_index import VectorManifest


@dataclass(frozen=True)
class VectorPoint:
    kb_id: str
    doc_id: str
    parent_chunk_id: str
    embedding_unit_id: str
    source_hash: str
    vector: Sequence[float]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VectorHit:
    point_id: str
    score: float
    kb_id: str
    doc_id: str
    parent_chunk_id: str
    embedding_unit_id: str
    source_hash: str
    metadata: dict


class VectorStore(Protocol):
    def create(self) -> bool: ...
    def upsert(self, points: Sequence[VectorPoint]) -> int: ...
    def search(self, vector: Sequence[float], kb_ids: list[str] | None = None,
               filters: dict | None = None, top_k: int = 5, exact: bool = True) -> list[VectorHit]: ...
    def count(self, kb_ids: list[str] | None = None) -> int: ...
    def delete_documents(self, doc_ids: Sequence[str]) -> int: ...
    def close(self): ...


class QdrantVectorStore:
    def __init__(self, manifest: VectorManifest, *, mode: str = "memory", path: Path | None = None,
                 url: str | None = None, api_key: str | None = None, traces: Path | None = None):
        if mode not in ("memory", "local", "server"):
            raise ValueError("Qdrant mode must be memory, local or server")
        if (mode == "memory" and (path is not None or url is not None or api_key is not None)
                or mode == "local" and (path is None or url is not None or api_key is not None)
                or mode == "server" and (url is None or path is not None)):
            raise ValueError("specify exactly the connection fields required by Qdrant mode")
        if mode == "server":
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Qdrant URL must be HTTP(S), with credentials supplied separately")
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise ValueError("Qdrant is optional; install with uv sync --extra vector") from error
        self.manifest, self.mode, self.traces, self.models = manifest, mode, traces, models
        self._closed = False
        if mode == "local":
            self.client = QdrantClient(path=str(Path(path).resolve()))
        elif mode == "server":
            self.client = QdrantClient(url=url, api_key=api_key, timeout=10)
        else:
            self.client = QdrantClient(location=":memory:")

    def _trace(self, operation: str, **attributes) -> Trace:
        return Trace(self.traces, "vector_" + operation, mode=self.mode,
                     contract_id=self.manifest.contract_id, **attributes)

    def _check_collection(self):
        if self._closed:
            raise ValueError("vector store is closed")
        if not self.client.collection_exists(self.manifest.collection):
            raise ValueError("vector collection has not been created")
        vectors = self.client.get_collection(self.manifest.collection).config.params.vectors
        expected = self.manifest.spec
        if (not isinstance(vectors, dict) or set(vectors) != {self.manifest.vector_name}
                or vectors[self.manifest.vector_name].size != expected.dimension
                or vectors[self.manifest.vector_name].distance.value.casefold() != expected.distance):
            raise ValueError("Qdrant collection is incompatible with this manifest")

    def create(self) -> bool:
        with self._trace("create") as trace:
            if self._closed:
                raise ValueError("vector store is closed")
            with trace.stage("schema"):
                created = not self.client.collection_exists(self.manifest.collection)
                if created:
                    distance = {"cosine": self.models.Distance.COSINE,
                                "dot": self.models.Distance.DOT, "euclid": self.models.Distance.EUCLID}[self.manifest.spec.distance]
                    self.client.create_collection(self.manifest.collection, vectors_config={
                        self.manifest.vector_name: self.models.VectorParams(size=self.manifest.spec.dimension, distance=distance)})
                self._check_collection()
            trace.record["created"] = created
        return created

    def _payload(self, point: VectorPoint) -> dict:
        if not isinstance(point, VectorPoint):
            raise ValueError("upsert requires VectorPoint records")
        for value in (point.kb_id, point.doc_id, point.parent_chunk_id, point.embedding_unit_id, point.source_hash):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("point identity fields must be nonempty strings")
        self.manifest.validate_source(point.kb_id, point.doc_id, point.source_hash)
        if not isinstance(point.metadata, dict) or any(not isinstance(k, str) for k in point.metadata):
            raise ValueError("point metadata must be a JSON object with string keys")
        try:
            metadata = json.loads(json.dumps(point.metadata, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError("metadata must contain finite JSON values") from None
        return {"record_type": "chunk-vector-v1", "kb_id": point.kb_id, "doc_id": point.doc_id,
                "parent_chunk_id": point.parent_chunk_id, "embedding_unit_id": point.embedding_unit_id,
                "source_hash": point.source_hash, "generation": self.manifest.generation,
                "model_spec_hash": self.manifest.spec.fingerprint, "contract_id": self.manifest.contract_id,
                "metadata": metadata}

    def _point_id(self, payload: dict) -> str:
        return str(uuid5(NAMESPACE_URL, canonical([self.manifest.contract_id, payload["kb_id"],
                          payload["doc_id"], payload["parent_chunk_id"], payload["embedding_unit_id"]])))

    def upsert(self, points: Sequence[VectorPoint]) -> int:
        with self._trace("upsert", requested=len(points)) as trace:
            with trace.stage("validate"):
                records = []
                ids = set()
                for point in points:
                    payload = self._payload(point)
                    vector = self.manifest.spec.validate_vector(point.vector)
                    point_id = self._point_id(payload)
                    if point_id in ids:
                        raise ValueError("duplicate point identity within one batch")
                    ids.add(point_id)
                    records.append(self.models.PointStruct(id=point_id, vector={self.manifest.vector_name: vector}, payload=payload))
            with trace.stage("persist"):
                self._check_collection()
                if records:
                    self.client.upsert(self.manifest.collection, points=records, wait=True)
            trace.record["written"] = len(records)
        return len(records)

    def _filter(self, kb_ids: list[str] | None = None, filters: dict | None = None,
                doc_ids: Sequence[str] | None = None):
        selected = sorted(self.manifest.knowledge_bases) if kb_ids is None else kb_ids
        if (not isinstance(selected, (list, tuple)) or not selected
                or any(not isinstance(kb, str) for kb in selected)
                or not set(selected) <= self.manifest.knowledge_bases):
            raise ValueError("unknown or empty vector knowledge-base selection")
        conditions = [self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
                      for key, value in (("record_type", "chunk-vector-v1"), ("contract_id", self.manifest.contract_id),
                                         ("generation", self.manifest.generation), ("model_spec_hash", self.manifest.spec.fingerprint))]
        conditions.append(self.models.FieldCondition(key="kb_id", match=self.models.MatchAny(any=selected)))
        if doc_ids is not None:
            if not doc_ids or not set(doc_ids) <= {doc for doc, _ in self.manifest.source_versions}:
                raise ValueError("unknown or empty document selection")
            conditions.append(self.models.FieldCondition(key="doc_id", match=self.models.MatchAny(any=list(doc_ids))))
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("metadata filters must be an object")
        for key, value in (filters or {}).items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError("metadata filter keys must be identifiers without dots")
            if type(value) in (str, bool, int):
                condition = self.models.FieldCondition(key="metadata." + key, match=self.models.MatchValue(value=value))
            elif type(value) is float and math.isfinite(value):
                condition = self.models.FieldCondition(key="metadata." + key, range=self.models.Range(gte=value, lte=value))
            else:
                raise ValueError("V1 metadata filters support only finite scalar values")
            conditions.append(condition)
        return self.models.Filter(must=conditions)

    def search(self, vector: Sequence[float], kb_ids: list[str] | None = None, filters: dict | None = None,
               top_k: int = 5, exact: bool = True) -> list[VectorHit]:
        with self._trace("search", top_k=top_k, exact=exact) as trace:
            with trace.stage("validate"):
                if type(top_k) is not int or not 1 <= top_k <= 50 or type(exact) is not bool:
                    raise ValueError("top_k must be 1..50 and exact must be boolean")
                if self.mode != "server" and not exact:
                    raise ValueError("local and memory modes support exact search only")
                values = self.manifest.spec.validate_vector(vector)
                scope = self._filter(kb_ids, filters)
                self._check_collection()
            with trace.stage("query"):
                result = self.client.query_points(self.manifest.collection, query=values,
                    using=self.manifest.vector_name, query_filter=scope, limit=top_k,
                    search_params=self.models.SearchParams(exact=exact) if self.mode == "server" else None,
                    with_payload=True, with_vectors=False)
            with trace.stage("verify_results"):
                hits = []
                for record in result.points:
                    payload = record.payload or {}
                    if (payload.get("contract_id") != self.manifest.contract_id
                            or payload.get("generation") != self.manifest.generation
                            or payload.get("model_spec_hash") != self.manifest.spec.fingerprint):
                        raise ValueError("query returned a foreign vector contract")
                    try:
                        point = VectorPoint(payload["kb_id"], payload["doc_id"], payload["parent_chunk_id"],
                                            payload["embedding_unit_id"], payload["source_hash"], (), payload["metadata"])
                        if payload != self._payload(point) or str(record.id) != self._point_id(payload):
                            raise ValueError("query returned an invalid point identity or payload")
                    except KeyError:
                        raise ValueError("query returned an incomplete vector payload") from None
                    if point.kb_id not in (kb_ids or self.manifest.knowledge_bases) or not math.isfinite(record.score):
                        raise ValueError("query returned an out-of-scope or non-finite result")
                    hits.append(VectorHit(str(record.id), float(record.score), point.kb_id, point.doc_id,
                                          point.parent_chunk_id, point.embedding_unit_id, point.source_hash, point.metadata))
                hits.sort(key=lambda hit: ((hit.score if self.manifest.spec.distance == "euclid" else -hit.score), hit.point_id))
            trace.record["returned_count"] = len(hits)
        return hits

    def count(self, kb_ids: list[str] | None = None) -> int:
        with self._trace("count") as trace:
            with trace.stage("query"):
                self._check_collection()
                return self.client.count(self.manifest.collection, count_filter=self._filter(kb_ids), exact=True).count

    def delete_documents(self, doc_ids: Sequence[str]) -> int:
        with self._trace("delete", requested_documents=len(doc_ids)) as trace:
            with trace.stage("persist"):
                scope = self._filter(doc_ids=doc_ids)
                self._check_collection()
                before = self.client.count(self.manifest.collection, count_filter=scope, exact=True).count
                self.client.delete(self.manifest.collection, points_selector=self.models.FilterSelector(filter=scope), wait=True)
                after = self.client.count(self.manifest.collection, count_filter=scope, exact=True).count
            trace.record["deleted_count"] = before - after
        return before - after

    def close(self):
        if not self._closed:
            self.client.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        self.close()
