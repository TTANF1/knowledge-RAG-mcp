from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
import re

from .model_download import E5_MODEL, E5_REVISION


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "local"
    model: str = E5_MODEL
    revision: str = E5_REVISION
    dimension: int = 384
    max_tokens: int = 512
    template: str = "e5-v1"
    model_dir: Path = Path(".state/models/multilingual-e5-small")
    cache: Path = Path(".state/embedding-cache.db")
    tokenizer: str = "cl100k_base"
    base_url: str = ""
    api_key_env: str = "KNOWLEDGE_RAG_EMBEDDING_KEY"
    batch_size: int = 2
    threads: int = 2
    query_cache: bool = False

    def __post_init__(self):
        if self.provider not in ("local", "api") or self.template not in ("e5-v1", "plain-v1"):
            raise ValueError("embedding provider must be local/api, template e5-v1/plain-v1")
        if not self.model or not self.revision or not self.tokenizer:
            raise ValueError("embedding model, revision and tokenizer are required")
        if (type(self.dimension) is not int or not 1 <= self.dimension <= 65536
                or type(self.max_tokens) is not int or self.max_tokens < 8
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= 128
                or type(self.threads) is not int or not 1 <= self.threads <= 64
                or type(self.query_cache) is not bool):
            raise ValueError("invalid embedding limits")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must name an environment variable")


@dataclass(frozen=True)
class VectorConfig:
    root: Path
    mode: str = "local"
    url: str = ""
    api_key_env: str = "QDRANT_API_KEY"

    def __post_init__(self):
        if self.mode not in ("local", "server"):
            raise ValueError("vector mode must be local or server")


@dataclass(frozen=True)
class Source:
    id: str
    root: Path
    include: tuple[str, ...] = ("*.md", "**/*.md")
    exclude: tuple[str, ...] = (".*/**", "**/.*/**")


@dataclass(frozen=True)
class Config:
    path: Path
    database: Path
    traces: Path
    sources: tuple[Source, ...]
    chunk_chars: int = 1200
    max_file_bytes: int = 2_000_000
    embedding: EmbeddingConfig | None = None
    vector: VectorConfig | None = None


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    base = path.parent
    sources = tuple(Source(
        id=s["id"], root=(base / s["root"]).resolve(),
        include=tuple(s.get("include", ["*.md", "**/*.md"])),
        exclude=tuple(s.get("exclude", [".*/**", "**/.*/**"])),
    ) for s in data.get("sources", []))
    if not sources or len({s.id for s in sources}) != len(sources):
        raise ValueError("sources must have unique, nonempty ids")
    if any(not s.id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in s.id) for s in sources):
        raise ValueError("source ids must use lowercase letters, digits, - or _")
    chunk_chars = data.get("index", {}).get("chunk_chars", 1200)
    max_bytes = data.get("index", {}).get("max_file_bytes", 2_000_000)
    if not isinstance(chunk_chars, int) or chunk_chars < 100 or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("invalid index limits")
    storage = data.get("storage", {})
    embedding_data = data.get("embedding")
    vector_data = data.get("vector")
    embedding = None
    if embedding_data is not None:
        try:
            embedding = EmbeddingConfig(**{**embedding_data,
                "model_dir": (base / embedding_data.get("model_dir", ".state/models/multilingual-e5-small")).resolve(),
                "cache": (base / embedding_data.get("cache", ".state/embedding-cache.db")).resolve()})
        except TypeError:
            raise ValueError("unknown embedding configuration field; API keys belong in environment variables") from None
    vector = None
    if vector_data is not None:
        try:
            vector = VectorConfig(**{**vector_data, "root": (base / vector_data.get("root", ".state/vectors")).resolve()})
        except TypeError:
            raise ValueError("unknown vector configuration field") from None
    return Config(path, (base / storage.get("database", ".state/index.db")).resolve(),
                  (base / storage.get("traces", ".state/traces.jsonl")).resolve(),
                  sources, chunk_chars, max_bytes, embedding, vector)
