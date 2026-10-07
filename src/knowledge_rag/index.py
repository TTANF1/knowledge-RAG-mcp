from __future__ import annotations

from contextlib import closing
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import sqlite3

import yaml

from .config import Config, Source
from .telemetry import Trace, canonical, digest

PARSER_VERSION = "markdown-lines-v1"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, path TEXT NOT NULL,
            source_hash TEXT NOT NULL, pipeline_hash TEXT NOT NULL,
            title TEXT NOT NULL, metadata TEXT NOT NULL, content TEXT NOT NULL,
            UNIQUE(kb_id, path));
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            heading TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            text TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS documents_kb ON documents(kb_id);
    """)
    return db


def source_files(source: Source) -> list[Path]:
    if not source.root.is_dir():
        raise ValueError(f"source root is missing: {source.id}")
    found = []
    for path in source.root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source.root).as_posix()
        if any(part.startswith(".") for part in path.relative_to(source.root).parts):
            continue
        if not any(fnmatchcase(relative, pattern) for pattern in source.include):
            continue
        if any(fnmatchcase(relative, pattern) for pattern in source.exclude):
            continue
        if not path.resolve().is_relative_to(source.root):
            raise ValueError(f"source link escapes configured root: {source.id}")
        found.append(path)
    return sorted(found)


def parse_markdown(content: str, fallback_title: str, max_chars: int) -> tuple[str, dict, list[dict]]:
    lines = content.splitlines(keepends=True)
    metadata, offset = {}, 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            raise ValueError("unterminated YAML frontmatter")
        metadata = yaml.safe_load("".join(lines[1:end])) or {}
        if not isinstance(metadata, dict):
            raise ValueError("frontmatter must be a mapping")
        # YAML dates and other scalar types become portable JSON values.
        metadata = json.loads(json.dumps(metadata, ensure_ascii=False, default=str))
        offset = end + 1
    title = str(metadata.get("title") or fallback_title)
    chunks, buffer, headings = [], [], []
    start, size, fence = offset + 1, 0, None

    def flush():
        nonlocal buffer, size
        text = "".join(buffer)
        if text.strip():
            chunks.append({"heading": " > ".join(h[1] for h in headings),
                           "start_line": start, "end_line": start + len(buffer) - 1,
                           "text": text})
        buffer, size = [], 0

    for number, line in enumerate(lines[offset:], offset + 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if heading:
            flush()
            level, label = len(heading[1]), heading[2]
            headings = [h for h in headings if h[0] < level] + [(level, label)]
            if level == 1 and title == fallback_title:
                title = label
        elif size >= max_chars and fence is None:
            flush()
        if not buffer:
            start = number
        buffer.append(line)
        size += len(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
    flush()
    return title, metadata, chunks


def ingest(config: Config) -> dict:
    pipeline_hash = digest(canonical({"parser": PARSER_VERSION, "chunk_chars": config.chunk_chars}))
    counts = {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0,
              "files": 0, "bytes_read": 0, "chunks_written": 0}
    with Trace(config.traces, "ingest", pipeline_hash=pipeline_hash) as trace:
        with trace.stage("scan"):
            # Validate all roots before beginning the transaction; a missing drive is not a deletion.
            files = [(source, source_files(source)) for source in config.sources]
        with closing(connect(config.database)) as db, db:
            seen = set()
            for source, paths in files:
                for path in paths:
                    relative = path.relative_to(source.root).as_posix()
                    doc_id = f"{source.id}:{relative}"
                    seen.add(doc_id)
                    with trace.stage("read_hash"):
                        if path.stat().st_size > config.max_file_bytes:
                            raise ValueError(f"file exceeds size limit: {doc_id}")
                        raw = path.read_bytes()
                        source_hash = digest(raw)
                    counts["files"] += 1
                    counts["bytes_read"] += len(raw)
                    existing = db.execute("SELECT source_hash,pipeline_hash FROM documents WHERE id=?", (doc_id,)).fetchone()
                    if existing and existing["source_hash"] == source_hash and existing["pipeline_hash"] == pipeline_hash:
                        counts["unchanged"] += 1
                        continue
                    with trace.stage("parse_chunk"):
                        content = raw.decode("utf-8-sig")
                        title, metadata, chunks = parse_markdown(content, path.stem, config.chunk_chars)
                    with trace.stage("persist"):
                        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
                        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)", (
                            doc_id, source.id, relative, source_hash, pipeline_hash, title, canonical(metadata), content))
                        for chunk in chunks:
                            chunk_id = digest(canonical([doc_id, source_hash, pipeline_hash, chunk["start_line"], chunk["end_line"]]))
                            db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?)", (
                                chunk_id, doc_id, chunk["heading"], chunk["start_line"], chunk["end_line"], chunk["text"]))
                    counts["updated" if existing else "added"] += 1
                    counts["chunks_written"] += len(chunks)
            with trace.stage("delete_stale"):
                # The configuration is authoritative: removed sources are removed from this index too.
                stale = {row[0] for row in db.execute("SELECT id FROM documents")} - seen
                db.executemany("DELETE FROM documents WHERE id=?", [(doc_id,) for doc_id in stale])
                counts["deleted"] = len(stale)
        trace.record["counts"] = counts
    return {**counts, "trace_id": trace.record["trace_id"], "operation_ms": trace.record["operation_ms"]}


def manifest(database: Path) -> list[dict]:
    with closing(connect(database)) as db:
        return [dict(row) for row in db.execute("SELECT id,source_hash FROM documents ORDER BY id")]
