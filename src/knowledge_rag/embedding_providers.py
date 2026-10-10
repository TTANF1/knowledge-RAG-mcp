"""Local ONNX and HTTP embeddings share one validated, versioned contract."""
from dataclasses import dataclass
from contextlib import closing
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sqlite3
from time import perf_counter
from urllib.parse import urlparse

from .config import EmbeddingConfig
from .embeddings import EmbeddingSpec
from .model_download import file_hash
from .telemetry import canonical, digest


class TextProvider:
    def prepare(self, text: str, role: str) -> str:
        return (("query: " if role == "query" else "passage: ") if self.config.template == "e5-v1" else "") + text

    def count_tokens(self, text: str) -> int:
        return self._count(text)

    def document_tokens(self, text: str) -> int:
        return self._count(self.prepare(text, "document"))

    def _encode_role(self, texts, role):
        prepared = [self.prepare(text, role) for text in texts]
        if any(self._count(text) > self.spec.max_tokens for text in prepared):
            raise ValueError("embedding input exceeds token limit; split documents or shorten query")
        vectors = self._encode(prepared)
        if len(vectors) != len(texts):
            raise ValueError("embedding result count mismatch")
        output = []
        for vector in vectors:
            if (not isinstance(vector, (list, tuple)) or len(vector) != self.spec.dimension
                    or any(type(x) is bool or not isinstance(x, (int, float)) for x in vector)):
                raise ValueError("invalid embedding response dimension or numeric values")
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("invalid embedding response values")
            output.append(self.spec.validate_vector([value / norm for value in vector]))
        return output

    def encode_documents(self, texts):
        return self._encode_role(texts, "document")

    def encode_query(self, text):
        return self._encode_role([text], "query")[0]


class LocalONNXProvider(TextProvider):
    def __init__(self, config: EmbeddingConfig):
        started = perf_counter()
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            raise ValueError("local embedding requires the embedding-local extra") from None
        self.config = config
        model, tokenizer = config.model_dir / "onnx/model.onnx", config.model_dir / "tokenizer.json"
        if not model.is_file() or not tokenizer.is_file():
            raise ValueError("local model not installed; run model-download or configure a compatible ONNX directory")
        artifact_hash = digest(canonical([file_hash(model), file_hash(tokenizer)]))
        self.spec = EmbeddingSpec("onnx-cpu", config.model, config.revision + ":" + artifact_hash,
                                  config.dimension, config.max_tokens, config.template + ":mean-pool-pad-v1")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.no_truncation()
        pad_token = "<pad>" if self.tokenizer.token_to_id("<pad>") is not None else "[PAD]"
        pad_id = self.tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError("local tokenizer requires a padding token")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.load_ms = (perf_counter() - started) * 1000

    def _count(self, text):
        return len(self.tokenizer.encode(text).ids)

    def _encode(self, texts):
        import numpy as np
        output = []
        names = {field.name for field in self.session.get_inputs()}
        for start in range(0, len(texts), self.config.batch_size):
            encodings = self.tokenizer.encode_batch(texts[start:start + self.config.batch_size])
            values = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                      "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                      "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64)}
            hidden = self.session.run(None, {name: values[name] for name in names})[0]
            if hidden.ndim != 3 or hidden.shape[-1] != self.spec.dimension:
                raise ValueError("local ONNX model must output token states with configured dimension")
            mask = values["attention_mask"][..., None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            output.extend(pooled.tolist())
        return output


class APIProvider(TextProvider):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        parsed = urlparse(config.base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))):
            raise ValueError("embedding API URL must use HTTPS, or HTTP on localhost, without credentials/query")
        if not os.environ.get(config.api_key_env):
            raise ValueError("embedding API key environment variable is not set: " + config.api_key_env)
        try:
            import tiktoken
        except ImportError:
            raise ValueError("API embedding requires the embedding-api extra") from None
        os.environ["TIKTOKEN_CACHE_DIR"] = str(config.cache.parent / "tokenizers")
        self.tokenizer = tiktoken.get_encoding(config.tokenizer)
        self.spec = EmbeddingSpec("api-" + digest(config.base_url.rstrip("/"))[:16], config.model,
            config.revision, config.dimension, config.max_tokens, config.template + ":" + config.tokenizer)
        self.load_ms = 0.0

    def _count(self, text):
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _encode(self, texts):
        import httpx
        output = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]
            try:
                with httpx.Client(timeout=60, follow_redirects=False) as client:
                    response = client.post(self.config.base_url.rstrip("/") + "/embeddings",
                        headers={"Authorization": "Bearer " + os.environ[self.config.api_key_env]},
                        json={"model": self.config.model, "input": batch, "encoding_format": "float"})
                if response.status_code != 200:
                    raise ValueError("embedding API returned HTTP " + str(response.status_code))
                data = response.json()["data"]
                if (not isinstance(data, list) or len(data) != len(batch)
                        or any(not isinstance(row, dict) or type(row.get("index")) is not int for row in data)
                        or {row["index"] for row in data} != set(range(len(batch)))):
                    raise ValueError("embedding API returned inconsistent result indexes")
                output.extend(row["embedding"] for row in sorted(data, key=lambda row: row["index"]))
            except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError):
                # Do not echo URLs, request text, response bodies or credentials.
                raise ValueError("embedding API transport or response format error") from None
        return output


@dataclass(frozen=True)
class EncodingUnit:
    text: str
    start: int
    end: int
    tokens: int


def document_windows(provider, text: str, title: str = "") -> list[EncodingUnit]:
    """Split original characters, retaining complete coverage and counting the full template."""
    context = title + "\n" if title else ""
    if provider.document_tokens(context) >= provider.spec.max_tokens:
        raise ValueError("document title exceeds embedding token budget")
    units, start = [], 0
    while start < len(text):
        low, high, accepted = start + 1, len(text), start
        while low <= high:
            end = (low + high) // 2
            if provider.document_tokens(context + text[start:end]) <= provider.spec.max_tokens:
                accepted, low = end, end + 1
            else:
                high = end - 1
        if accepted == start:
            raise ValueError("embedding budget cannot fit a document character")
        value = context + text[start:accepted]
        units.append(EncodingUnit(value, start, accepted, provider.document_tokens(value)))
        start = accepted
    return units


class CachedProvider:
    def __init__(self, provider, path: Path, query_cache=False):
        self.provider, self.spec, self.path = provider, provider.spec, Path(path)
        self.query_cache = query_cache
        self.load_ms = provider.load_ms
        self.stats = {"hits": 0, "misses": 0, "encoded_tokens": 0, "encode_ms": 0.0, "encode_calls": 0}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (id TEXT PRIMARY KEY, vector TEXT NOT NULL)")

    def document_tokens(self, text):
        return self.provider.document_tokens(text)

    def count_tokens(self, text):
        return self.provider.count_tokens(text)

    def _encode(self, texts, role, cached=True):
        keys = [digest(canonical([self.spec.fingerprint, role, self.provider.prepare(text, role)])) for text in texts]
        known = {}
        if cached:
            with closing(sqlite3.connect(self.path)) as db:
                for key in set(keys):
                    row = db.execute("SELECT vector FROM embeddings WHERE id=?", (key,)).fetchone()
                    if row:
                        known[key] = self.spec.validate_vector(json.loads(row[0]))
        missing = {}
        for key, text in zip(keys, texts):
            if key not in known:
                missing.setdefault(key, text)
        self.stats["hits"] += sum(key in known for key in keys)
        self.stats["misses"] += len(missing)
        if missing:
            started = perf_counter()
            values = list(missing.values())
            vectors = self.provider.encode_documents(values) if role == "document" else [self.provider.encode_query(v) for v in values]
            if len(vectors) != len(missing):
                raise ValueError("embedding result count mismatch")
            validated = [self.spec.validate_vector(vector) for vector in vectors]
            self.stats["encode_ms"] += (perf_counter() - started) * 1000
            self.stats["encode_calls"] += 1
            self.stats["encoded_tokens"] += sum(self.provider.count_tokens(self.provider.prepare(text, role)) for text in values)
            known.update(zip(missing, validated))
            if cached:
                with closing(sqlite3.connect(self.path)) as db, db:
                    db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?)", [(key, canonical(known[key])) for key in missing])
        return [known[key] for key in keys]

    def encode_documents(self, texts):
        return self._encode(texts, "document")

    def encode_query(self, text):
        return self._encode([text], "query", self.query_cache)[0]


@lru_cache(maxsize=1)
def get_provider(config: EmbeddingConfig) -> CachedProvider:
    provider = LocalONNXProvider(config) if config.provider == "local" else APIProvider(config)
    return CachedProvider(provider, config.cache, config.query_cache)
