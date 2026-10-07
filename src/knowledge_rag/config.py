from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


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
    return Config(path, (base / storage.get("database", ".state/index.db")).resolve(),
                  (base / storage.get("traces", ".state/traces.jsonl")).resolve(),
                  sources, chunk_chars, max_bytes)
