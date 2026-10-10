"""Versioned embedding contracts; real model providers are the next milestone."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Protocol, Sequence

from .telemetry import canonical, digest


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model_id: str
    revision: str
    dimension: int
    max_tokens: int
    template_version: str
    normalize: bool = True
    distance: str = "cosine"

    def __post_init__(self):
        for value in (self.provider, self.model_id, self.revision, self.template_version):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("embedding provider, model, revision and template must be specified")
        if type(self.dimension) is not int or not 1 <= self.dimension <= 65536:
            raise ValueError("embedding dimension must be an integer in 1..65536")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.normalize) is not bool or self.distance not in ("cosine", "dot", "euclid"):
            raise ValueError("invalid normalization or distance contract")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return digest(canonical(self.to_dict()))

    def validate_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.dimension:
            raise ValueError("vector dimension does not match embedding contract")
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in vector):
            raise ValueError("vector values must be real numbers")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite")
        norm = math.hypot(*values)
        if self.distance == "cosine" and norm == 0:
            raise ValueError("cosine vectors cannot be zero")
        if self.normalize and not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("vector must be normalized according to embedding contract")
        return values


class EmbeddingProvider(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def count_tokens(self, text: str) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...
