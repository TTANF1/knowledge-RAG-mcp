from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

_lock = Lock()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str | bytes) -> str:
    return sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


class Trace:
    """One local append-only operation record; query/document text is omitted."""

    def __init__(self, path: Path | None, operation: str, **attributes):
        self.path = path
        self.record = {"schema_version": 1, "trace_id": uuid4().hex,
                       "timestamp": datetime.now(timezone.utc).isoformat(),
                       "operation": operation, "attributes": attributes, "stages_ms": {}}

    def __enter__(self):
        self.start = perf_counter()
        return self

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            stages = self.record["stages_ms"]
            stages[name] = stages.get(name, 0) + (perf_counter() - start) * 1000

    def __exit__(self, kind, error, tb):
        self.record["operation_ms"] = (perf_counter() - self.start) * 1000
        self.record["status"] = "error" if error else "ok"
        if error:
            self.record["error_type"] = kind.__name__
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _lock, self.path.open("a", encoding="utf-8") as file:
                file.write(canonical(self.record) + "\n")
        return False
