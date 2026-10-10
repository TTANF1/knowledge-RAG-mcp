"""Explicit, pinned model download; never downloads while serving queries."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
E5_MODEL = "intfloat/multilingual-e5-small"
FILES = {
    "onnx/model.onnx": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "config.json": None, "tokenizer_config.json": None, "special_tokens_map.json": None,
}


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download_e5(destination: Path) -> dict:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in FILES.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or (expected and file_hash(path) != expected):
            temporary = path.with_suffix(path.suffix + ".partial")
            request = urllib.request.Request(f"https://huggingface.co/{E5_MODEL}/resolve/{E5_REVISION}/{name}",
                                             headers={"User-Agent": "knowledge-rag-mcp"})
            try:
                with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                if expected and file_hash(temporary) != expected:
                    raise ValueError("download checksum mismatch: " + name)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        files[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    report = {"model": E5_MODEL, "revision": E5_REVISION, "files": files,
              "total_bytes": sum(item["bytes"] for item in files.values())}
    (destination / "download.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    print(json.dumps(download_e5(parser.parse_args().destination), indent=2))
