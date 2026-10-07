"""Create a checked source snapshot with no Git history or local runtime data."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from uuid import uuid4

if __package__:
    from .check_publication import audit, candidates
else:
    from check_publication import audit, candidates


def export(root: Path) -> Path:
    findings = audit(root)
    if findings:
        raise ValueError("publication check failed; run scripts/check_publication.py first")
    target = root / ".local" / "releases" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    for name in candidates(root):
        source = root / name
        if not source.is_file():
            continue
        destination = target / name
        if not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError("invalid public snapshot path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # .git is never among ls-files publication candidates.
    if (target / ".git").exists() or (target / ".state").exists() or (target / ".local").exists():
        raise ValueError("unexpected private directory in public snapshot")
    return target


if __name__ == "__main__":
    print(export(Path(__file__).resolve().parents[1]))
