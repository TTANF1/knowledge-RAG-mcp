from __future__ import annotations

from collections import Counter
from contextlib import closing
import json
import math
import re

from .config import Config
from .index import connect
from .telemetry import Trace, canonical, digest

TOKENIZER_VERSION = "latin-word-cjk-bigram-v1"


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1)) if len(word) > 1 else tokens.append(word)
        else:
            tokens.append(word)
    return tokens


class Retriever:
    def __init__(self, config: Config, strategy: str = "bm25"):
        if strategy not in ("bm25", "overlap"):
            raise ValueError("strategy must be bm25 or overlap")
        self.config, self.strategy = config, strategy

    def list_knowledge_bases(self) -> list[dict]:
        with closing(connect(self.config.database)) as db:
            return [dict(row) for row in db.execute(
                "SELECT kb_id,COUNT(*) AS documents FROM documents GROUP BY kb_id ORDER BY kb_id")
                if row["kb_id"] in {s.id for s in self.config.sources}]

    def search(self, query: str, kb_ids: list[str] | None = None,
               top_k: int = 5, max_chars: int = 4000, filters: dict | None = None) -> dict:
        with Trace(self.config.traces, "search", query_hash=digest(query), strategy=self.strategy,
                   tokenizer=TOKENIZER_VERSION, kb_ids=kb_ids, top_k=top_k,
                   max_chars=max_chars, filter_keys=sorted(filters or {})) as trace:
            if not query.strip() or len(query) > 2000:
                raise ValueError("query must contain 1..2000 characters")
            if not 1 <= top_k <= 50 or not 1 <= max_chars <= 20000:
                raise ValueError("top_k must be 1..50; max_chars must be 1..20000")
            if kb_ids is not None and (not kb_ids or not set(kb_ids) <= {s.id for s in self.config.sources}):
                raise ValueError("unknown or empty knowledge-base selection")
            with trace.stage("load_filter"):
                with closing(connect(self.config.database)) as db:
                    rows = [dict(r) for r in db.execute("""
                        SELECT c.*,d.kb_id,d.path,d.title,d.metadata,d.source_hash
                        FROM chunks c JOIN documents d ON d.id=c.doc_id ORDER BY c.id
                    """)]
                allowed = set(kb_ids) if kb_ids is not None else {s.id for s in self.config.sources}
                rows = [r for r in rows if r["kb_id"] in allowed]
                for row in rows:
                    row["metadata"] = json.loads(row["metadata"])
                rows = [r for r in rows if all(r["metadata"].get(k) == v for k, v in (filters or {}).items())]
            with trace.stage("tokenize"):
                query_tokens = set(tokenize(query))
                bags = [Counter(tokenize(r["title"] + "\n" + r["heading"] + "\n" + r["text"])) for r in rows]
                lengths = [sum(b.values()) for b in bags]
                average = sum(lengths) / len(lengths) if lengths else 1
                frequencies = Counter(token for bag in bags for token in bag)
            with trace.stage("rank"):
                ranked = []
                for row, bag, length in zip(rows, bags, lengths):
                    if self.strategy == "overlap":
                        score = float(len(query_tokens & bag.keys()))
                    else:
                        score = sum(math.log(1 + (len(rows) - frequencies[t] + 0.5) / (frequencies[t] + 0.5))
                                    * (bag[t] * 2.5) / (bag[t] + 1.5 * (0.25 + 0.75 * length / (average or 1)))
                                    for t in sorted(query_tokens) if bag[t])
                    if score > 0:
                        ranked.append((score, row))
                ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
            with trace.stage("pack"):
                hits, remaining = [], max_chars
                for score, row in ranked[:top_k]:
                    if remaining <= 0:
                        break
                    excerpt = row["text"][:remaining]
                    remaining -= len(excerpt)
                    hits.append({"chunk_id": row["id"], "doc_id": row["doc_id"],
                                 "kb_id": row["kb_id"], "path": row["path"],
                                 "title": row["title"], "heading": row["heading"],
                                 "source_hash": row["source_hash"], "start_line": row["start_line"],
                                 "end_line": row["end_line"], "metadata": row["metadata"],
                                 "score": score, "text": excerpt, "truncated": len(excerpt) < len(row["text"])})
                result = {"trace_id": trace.record["trace_id"], "hits": hits,
                          "status": "candidates_found" if hits else "no_evidence",
                          "content_chars": max_chars - remaining}
                trace.record["candidate_count"] = len(rows)
                trace.record["matching_count"] = len(ranked)
                trace.record["returned"] = [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]
                trace.record["content_chars"] = result["content_chars"]
                trace.record["response_json_bytes"] = len(canonical(result).encode("utf-8"))
        return result

    def read_document(self, doc_id: str, source_hash: str | None = None,
                      start_line: int = 1, max_chars: int = 4000) -> dict:
        with Trace(self.config.traces, "read_document", doc_id=doc_id, start_line=start_line) as trace:
            if start_line < 1 or not 1 <= max_chars <= 20000:
                raise ValueError("invalid read bounds")
            with trace.stage("load"):
                with closing(connect(self.config.database)) as db:
                    row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
                if row is None or row["kb_id"] not in {s.id for s in self.config.sources}:
                    raise ValueError("document is not indexed")
                if source_hash is not None and source_hash != row["source_hash"]:
                    raise ValueError("document version changed; search again")
            with trace.stage("pack"):
                lines = row["content"].splitlines(keepends=True)
                if start_line > len(lines) + 1:
                    raise ValueError("start_line exceeds document length")
                remainder = "".join(lines[start_line - 1:])
                text = remainder[:max_chars]
                result = {"doc_id": doc_id, "source_hash": row["source_hash"],
                          "start_line": start_line, "text": text,
                          "truncated": len(text) < len(remainder),
                          "metadata": json.loads(row["metadata"]), "trace_id": trace.record["trace_id"]}
                trace.record["content_chars"] = len(text)
            return result
