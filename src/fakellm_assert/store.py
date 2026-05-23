"""Frozen-judgment storage.

A judgment is expensive and non-deterministic to produce (it calls a judge
model), so we produce it exactly once during an explicit `--update` run and
freeze the verdict to disk. Normal test runs only ever *read* frozen
verdicts. A cache miss in normal mode is an error, never a silent live call —
that is what keeps the test suite deterministic and offline.

The fingerprint binds a verdict to the exact thing it was a verdict about:
the response text, the assertion criterion, the judge model, and the prompt
template. Change any of those and the fingerprint changes, the old verdict no
longer applies, and the test fails until a human re-judges. That failure is
the feature.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Bump when the fingerprint inputs or schema change in a way that should
# invalidate every existing snapshot.
FINGERPRINT_VERSION = "1"


def fingerprint(
    *,
    response_text: str,
    criterion: str,
    judge_model: str,
    prompt_template: str,
) -> str:
    """Stable hash binding a verdict to exactly what it was a verdict about."""
    hasher = hashlib.sha256()
    for part in (
        FINGERPRINT_VERSION,
        response_text,
        criterion,
        judge_model,
        prompt_template,
    ):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")  # delimiter so concatenation can't collide
    return hasher.hexdigest()


@dataclass
class Judgment:
    fingerprint: str
    criterion: str
    verdict: str  # "pass" or "fail"
    reasoning: str
    judge_model: str
    response_excerpt: str  # human-readable context for the git diff
    judged_at: str

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


class SnapshotStore:
    """Reads/writes judgment snapshots for a single test file.

    On disk this is a JSON object mapping fingerprint -> judgment. We keep
    the judge's reasoning and a response excerpt in each record specifically
    so the git diff is human-readable: a reviewer running `--update` should
    be able to see *why* a verdict flipped without rerunning anything.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._loaded = False
        self._dirty = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
            except (json.JSONDecodeError, ValueError):
                self._data = {}
        self._loaded = True

    def get(self, fp: str) -> Optional[Judgment]:
        self._load()
        record = self._data.get(fp)
        if record is None:
            return None
        return Judgment(**record)

    def put(self, judgment: Judgment) -> None:
        self._load()
        self._data[judgment.fingerprint] = asdict(judgment)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Sorted keys + trailing newline => stable, diff-friendly files.
        serialized = json.dumps(self._data, indent=2, sort_keys=True) + "\n"
        self.path.write_text(serialized, "utf-8")
        self._dirty = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
