"""Check Git publication candidates without displaying matched private content."""
from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

PRIVATE_PREFIXES = (".state/", ".local/", "logs/", ".logs/", "data/private/", "docs/private/",
                    "docs/learning/private/", "docs/experiments/private/",
                    "benchmarks/runs/", "benchmarks/baselines/", "benchmarks/private/")
PRIVATE_GLOBS = ("*.db", "*.db-*", "*.sqlite", "*.sqlite-*", "*.sqlite3", "*.sqlite3-*", "*.log",
                 "*.connected.toml", "*.local.toml", "*.pem", "*.key", "*.p12", "*.pfx", ".env", ".env.*")
PUBLIC_JSONL = "benchmarks/datasets/smoke-v1.jsonl"
MAX_TEXT_BYTES = 16_000_000


def git(root: Path, *args: str, check: bool = True) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=check)
    return result.stdout


def normalized(text: str) -> str:
    return re.sub(r"/+", "/", text.replace("\\", "/")).casefold()


def forbidden_path(name: str) -> bool:
    name = name.replace("\\", "/")
    leaf = name.rsplit("/", 1)[-1]
    return (name.startswith(PRIVATE_PREFIXES)
            or (leaf != ".env.example" and any(fnmatchcase(leaf, pattern) for pattern in PRIVATE_GLOBS))
            or (name.endswith(".jsonl") and name != PUBLIC_JSONL))


def private_tokens(root: Path) -> list[str]:
    tokens = [str(root.resolve()), str(Path.home())]
    state = root / ".state"
    for path in state.glob("*.connected.toml"):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            for source in data.get("sources", []):
                location = Path(source["root"]).resolve()
                if not location.is_relative_to(root):
                    tokens.append(str(location))
        except (OSError, ValueError, KeyError):
            raise ValueError("cannot read local source configuration for publication checking")
    return [normalized(token) for token in tokens if len(token) > 8]


def inspect_file(name: str, content: bytes, tokens: list[str]) -> list[str]:
    if forbidden_path(name):
        return ["private-artifact-path"]
    if len(content) > MAX_TEXT_BYTES:
        return ["large-file-needs-manual-review"]
    if b"\0" in content:
        return ["binary-file-needs-manual-review"]
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ["non-utf8-file-needs-manual-review"]
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        value = normalized(line)
        if any(token in value for token in tokens):
            findings.append(f"line {number}: local-absolute-path")
        elif re.search(r"(?i)\b[a-z]:/(?!path/|yourknowledge\b|yourproject\b|example\b|tmp\b)[\w.-]+", value):
            findings.append(f"line {number}: machine-absolute-path")
        elif re.search(r"/(?:users|home)/[^/\s<>\"']+", value):
            findings.append(f"line {number}: user-home-path")
        if re.search(r"\bsk-(?:proj-)?[a-zA-Z0-9_-]{20,}", line):
            findings.append(f"line {number}: possible-api-key")
        if re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", line):
            findings.append(f"line {number}: private-key")
    if name.startswith("benchmarks/public/") and name.endswith("report.json"):
        try:
            report = json.loads(text)
            allowed = {"schema_version", "kind", "provenance", "settings", "metrics", "by_category", "stages_ms", "limitations"}
            if report.get("kind") != "synthetic-benchmark-summary" or set(report) - allowed:
                findings.append("unreviewed-public-benchmark-report")
            if report.get("provenance", {}).get("dataset") != "smoke-v1":
                findings.append("public-benchmark-is-not-approved-synthetic-set")
        except (ValueError, TypeError, AttributeError):
            findings.append("invalid-public-benchmark-report")
    return findings


def candidates(root: Path, staged: bool = False) -> list[str]:
    if staged:
        data = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    else:
        data = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(set(name.decode("utf-8") for name in data.split(b"\0") if name))


def audit(root: Path, staged: bool = False, history: bool = False) -> list[dict]:
    tokens = private_tokens(root)
    findings = []
    for name in candidates(root, staged):
        path = root / name
        if not staged and not path.is_file():
            continue
        if path.is_symlink() or (staged and git(root, "ls-files", "--stage", "--", name).startswith(b"120000 ")):
            findings.append({"file": name, "reason": "symlink-needs-manual-review"})
            continue
        content = git(root, "show", ":" + name) if staged else path.read_bytes()
        for reason in inspect_file(name, content, tokens):
            findings.append({"file": name, "reason": reason})
    if history:
        # Inspect every reachable blob, including removed files. Do not print blob text.
        seen = set()
        for commit in git(root, "rev-list", "--all").decode().splitlines():
            for item in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
                if not item:
                    continue
                header, raw_name = item.split(b"\t", 1)
                mode, kind, oid = header.split()
                name = raw_name.decode("utf-8")
                if kind != b"blob" or (name, oid) in seen:
                    continue
                seen.add((name, oid))
                for reason in inspect_file(name, git(root, "cat-file", "blob", oid.decode()), tokens):
                    findings.append({"commit": commit[:12], "file": name, "reason": reason})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        findings = audit(args.root.resolve(), args.staged, args.history)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Publication check could not finish: {type(error).__name__}", file=sys.stderr)
        return 2
    if findings:
        for item in findings[:80]:
            print(f"BLOCK: {item.get('commit', 'working-tree')} {item['file']} ({item['reason']})")
        print(f"Blocked by {len(findings)} finding(s); matched private content is not printed.")
        return 1
    print("Publication check passed for the selected Git files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
