"""Immutable source/model manifest for a vector storage generation (V1)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

from .embeddings import EmbeddingSpec
from .telemetry import canonical, digest


@dataclass(frozen=True)
class VectorManifest:
    spec: EmbeddingSpec
    generation: str
    source_versions: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.spec, EmbeddingSpec):
            raise ValueError("manifest requires an embedding specification")
        if not isinstance(self.generation, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.generation):
            raise ValueError("invalid vector generation")
        try:
            sources = tuple(sorted((doc_id, source_hash) for doc_id, source_hash in self.source_versions))
        except (ValueError, TypeError):
            raise ValueError("invalid source version manifest") from None
        if len({doc_id for doc_id, _ in sources}) != len(sources):
            raise ValueError("manifest requires unique source document ids")
        for doc_id, source_hash in sources:
            if (not isinstance(doc_id, str) or not re.fullmatch(r"[a-z0-9_-]+:.+", doc_id)
                    or not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash)):
                raise ValueError("invalid document id or source hash")
        object.__setattr__(self, "source_versions", sources)

    @property
    def corpus_hash(self) -> str:
        return digest(canonical([{"id": doc, "source_hash": version} for doc, version in self.source_versions]))

    @property
    def contract_id(self) -> str:
        return digest(canonical([1, self.spec.fingerprint, self.generation, self.corpus_hash]))

    @property
    def collection(self) -> str:
        return "krag_v1_" + self.contract_id[:32]

    @property
    def vector_name(self) -> str:
        return "embedding_" + self.contract_id

    @property
    def knowledge_bases(self) -> set[str]:
        return {doc.split(":", 1)[0] for doc, _ in self.source_versions}

    def validate_source(self, kb_id: str, doc_id: str, source_hash: str):
        if not doc_id.startswith(kb_id + ":") or dict(self.source_versions).get(doc_id) != source_hash:
            raise ValueError("point source does not match this manifest generation")

    def to_dict(self) -> dict:
        return {"schema_version": 1, "kind": "vector-storage-contract", "embedding_spec": self.spec.to_dict(),
                "generation": self.generation, "source_versions": [list(pair) for pair in self.source_versions],
                "corpus_hash": self.corpus_hash, "contract_id": self.contract_id,
                "collection": self.collection, "vector_name": self.vector_name}

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Same-directory hard link publishes complete bytes and cannot overwrite an existing manifest.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".vector-manifest-", delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")
                file.flush()
                os.fsync(file.fileno())
            except BaseException:
                file.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "VectorManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = {"schema_version", "kind", "embedding_spec", "generation", "source_versions", "corpus_hash",
                "contract_id", "collection", "vector_name"}
        if not isinstance(data, dict) or set(data) != keys or data["schema_version"] != 1 or data["kind"] != "vector-storage-contract":
            raise ValueError("unsupported vector manifest schema")
        try:
            result = cls(EmbeddingSpec(**data["embedding_spec"]), data["generation"], tuple(data["source_versions"]))
        except (TypeError, KeyError):
            raise ValueError("invalid embedding specification in manifest") from None
        if result.to_dict() != data:
            raise ValueError("vector manifest fingerprint or derived fields do not match")
        return result
