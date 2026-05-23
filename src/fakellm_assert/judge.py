"""Run modes and the judge interface.

Three modes, mirroring familiar snapshot-testing tools:

  REPLAY  (default): frozen verdicts only. A cache miss is a hard error.
                     Zero network. This is what CI runs.
  UPDATE:            cache misses trigger a real judge call and are written
                     to disk for a human to review in the diff. Run
                     intentionally, by a person, never in CI.
  STRICT:            like REPLAY, but attempting a live judge call is
                     impossible by construction — a guard against a
                     misconfigured machine sneaking onto the network.

Mode resolves from (in priority order): an explicit configure() call, then
the FAKELLM_ASSERT_MODE env var, else REPLAY.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Optional, Protocol


class Mode(str, Enum):
    REPLAY = "replay"
    UPDATE = "update"
    STRICT = "strict"


class JudgeResult:
    """What a judge returns: a boolean verdict plus its reasoning."""

    def __init__(self, passed: bool, reasoning: str):
        self.passed = passed
        self.reasoning = reasoning


class Judge(Protocol):
    """A judge maps (response_text, criterion) -> JudgeResult.

    Implement this however you like. The default implementation calls a
    model, but a judge could equally be a human-in-the-loop prompt or a
    rules engine. It is only ever invoked in UPDATE mode.
    """

    @property
    def model_name(self) -> str: ...

    def judge(self, response_text: str, criterion: str) -> JudgeResult: ...


# Default judge prompt. Part of the fingerprint, so editing it invalidates
# existing snapshots by design.
DEFAULT_PROMPT_TEMPLATE = (
    "You are a strict test oracle. Decide whether the RESPONSE satisfies the "
    "CRITERION. Answer with a JSON object: "
    '{{"pass": true|false, "reasoning": "<one sentence>"}}. '
    "Be literal; do not give credit for near-misses.\n\n"
    "CRITERION:\n{criterion}\n\nRESPONSE:\n{response}\n"
)


class _Config:
    def __init__(self) -> None:
        self.mode: Optional[Mode] = None
        self.judge: Optional[Judge] = None
        self.prompt_template: str = DEFAULT_PROMPT_TEMPLATE
        self.snapshot_dir: str = ".fakellm/judgments"

    def resolve_mode(self) -> Mode:
        if self.mode is not None:
            return self.mode
        env = os.environ.get("FAKELLM_ASSERT_MODE", "").lower()
        if env in (m.value for m in Mode):
            return Mode(env)
        return Mode.REPLAY


_config = _Config()


def configure(
    *,
    mode: Optional[Mode | str] = None,
    judge: Optional[Judge] = None,
    prompt_template: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
) -> None:
    """Set global config. Typically called once in a conftest or fixture."""
    if mode is not None:
        _config.mode = Mode(mode) if isinstance(mode, str) else mode
    if judge is not None:
        _config.judge = judge
    if prompt_template is not None:
        _config.prompt_template = prompt_template
    if snapshot_dir is not None:
        _config.snapshot_dir = snapshot_dir


def get_config() -> _Config:
    return _config


class CallableJudge:
    """Adapts a plain function into a Judge.

    The function receives the fully-rendered prompt string and must return
    raw model output text containing a JSON object with "pass" and
    "reasoning". This keeps the package free of any SDK dependency: wire in
    openai, anthropic, or fakellm itself in three lines at the call site.
    """

    def __init__(self, fn: Callable[[str], str], model_name: str):
        self._fn = fn
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def judge(self, response_text: str, criterion: str) -> JudgeResult:
        import json as _json

        prompt = _config.prompt_template.format(
            criterion=criterion, response=response_text
        )
        raw = self._fn(prompt)
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = _json.loads(cleaned)
            return JudgeResult(
                passed=bool(parsed.get("pass")),
                reasoning=str(parsed.get("reasoning", "")),
            )
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Judge returned unparseable output: {raw!r}"
            ) from exc
