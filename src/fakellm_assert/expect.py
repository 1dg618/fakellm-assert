r"""The `expect()` assertion API.

    expect(resp).contains("refund")
    expect(resp).matches(r"\d{4}-\d{2}-\d{2}")
    expect(resp).is_valid_json()
    expect(resp).json_path("$.status").equals("ok")
    expect(resp).called_tool("issue_refund")
    expect(resp).satisfies("apologizes for the delay")   # frozen judgment

Tier 1 matchers (everything except `satisfies`) are pure functions over the
normalized response: deterministic, free, instant, no snapshot machinery.
They are the recommended default. `satisfies` is the escape hatch for
genuinely fuzzy criteria — every call is a snapshot someone must maintain.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .extract import Normalized, normalize
from .judge import Mode, get_config
from .store import Judgment, SnapshotStore, fingerprint, now_iso


class AssertionFailure(AssertionError):
    """Raised when an expectation is not met."""


class MissingJudgment(AssertionError):
    """Raised in REPLAY/STRICT mode when no frozen verdict exists.

    The message always tells the user exactly how to fix it: run with
    --update. We never silently fall through to a live call.
    """


def _truthy_or_fail(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionFailure(message)


class _JsonPathResult:
    """Tiny chainable result for json_path(...).equals(...) style asserts."""

    def __init__(self, value: Any, found: bool, path: str):
        self._value = value
        self._found = found
        self._path = path

    def exists(self) -> "_JsonPathResult":
        _truthy_or_fail(self._found, f"json_path {self._path!r} not found")
        return self

    def equals(self, expected: Any) -> "_JsonPathResult":
        self.exists()
        _truthy_or_fail(
            self._value == expected,
            f"json_path {self._path!r} == {self._value!r}, expected {expected!r}",
        )
        return self

    @property
    def value(self) -> Any:
        return self._value


def _walk_path(data: Any, path: str) -> tuple[Any, bool]:
    """Minimal JSONPath-ish walker supporting $.a.b[0].c.

    Deliberately tiny — covers the dotted/indexed access that 95% of tool-call
    assertions need without pulling in a jsonpath dependency.
    """
    if not path.startswith("$"):
        return None, False
    tokens = re.findall(r"\.([^.\[\]]+)|\[(\d+)\]", path)
    current = data
    for key, index in tokens:
        try:
            if key:
                if not isinstance(current, dict) or key not in current:
                    return None, False
                current = current[key]
            else:
                idx = int(index)
                if not isinstance(current, list) or idx >= len(current):
                    return None, False
                current = current[idx]
        except (KeyError, IndexError, TypeError):
            return None, False
    return current, True


class Expectation:
    def __init__(self, response: Any):
        self._n: Normalized = normalize(response)

    # ---- Tier 1: deterministic text matchers ---------------------------

    def contains(self, substring: str, *, case_sensitive: bool = True) -> "Expectation":
        haystack = self._n.text if case_sensitive else self._n.text.lower()
        needle = substring if case_sensitive else substring.lower()
        _truthy_or_fail(
            needle in haystack,
            f"expected response to contain {substring!r}",
        )
        return self

    def not_contains(self, substring: str, *, case_sensitive: bool = True) -> "Expectation":
        haystack = self._n.text if case_sensitive else self._n.text.lower()
        needle = substring if case_sensitive else substring.lower()
        _truthy_or_fail(
            needle not in haystack,
            f"expected response NOT to contain {substring!r}",
        )
        return self

    def contains_all(self, *substrings: str, case_sensitive: bool = True) -> "Expectation":
        for s in substrings:
            self.contains(s, case_sensitive=case_sensitive)
        return self

    def matches(self, pattern: str, *, flags: int = 0) -> "Expectation":
        _truthy_or_fail(
            re.search(pattern, self._n.text, flags) is not None,
            f"expected response to match /{pattern}/",
        )
        return self

    def equals(self, expected: str) -> "Expectation":
        _truthy_or_fail(
            self._n.text == expected,
            f"expected response text to equal {expected!r}, got {self._n.text!r}",
        )
        return self

    def has_length(self, *, lte: int | None = None, gte: int | None = None) -> "Expectation":
        length = len(self._n.text)
        if lte is not None:
            _truthy_or_fail(length <= lte, f"response length {length} > {lte}")
        if gte is not None:
            _truthy_or_fail(length >= gte, f"response length {length} < {gte}")
        return self

    # ---- Tier 1: structural matchers -----------------------------------

    def is_valid_json(self) -> "Expectation":
        try:
            json.loads(self._n.text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AssertionFailure(f"response is not valid JSON: {exc}") from exc
        return self

    def json_path(self, path: str) -> _JsonPathResult:
        try:
            data = json.loads(self._n.text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AssertionFailure(
                f"cannot evaluate json_path; response is not JSON: {exc}"
            ) from exc
        value, found = _walk_path(data, path)
        return _JsonPathResult(value, found, path)

    # ---- Tier 1: tool-call matchers ------------------------------------

    def called_tool(self, name: str) -> "Expectation":
        names = [tc.name for tc in self._n.tool_calls]
        _truthy_or_fail(
            name in names,
            f"expected a tool call to {name!r}; got {names or 'none'}",
        )
        return self

    def tool_args(self, name: str) -> _JsonPathResult:
        for tc in self._n.tool_calls:
            if tc.name == name:
                # Wrap in a result so you can chain .json_path-style checks
                # by re-normalizing the args dict.
                return _JsonPathResult(tc.arguments, True, f"tool:{name}")
        return _JsonPathResult(None, False, f"tool:{name}")

    # ---- Tier 3: frozen judgment ---------------------------------------

    def satisfies(self, criterion: str) -> "Expectation":
        """Assert the response meets a fuzzy, natural-language criterion.

        The verdict is frozen on first `--update` run and replayed
        deterministically thereafter. In REPLAY mode a missing verdict is a
        hard error pointing you at --update; in STRICT mode a live call is
        impossible by construction.
        """
        cfg = get_config()
        mode = cfg.resolve_mode()
        fp = fingerprint(
            response_text=self._n.text,
            criterion=criterion,
            judge_model=cfg.judge.model_name if cfg.judge else "<none>",
            prompt_template=cfg.prompt_template,
        )
        store = SnapshotStore(_snapshot_path_for(criterion, cfg.snapshot_dir))
        existing = store.get(fp)

        if existing is not None:
            _truthy_or_fail(
                existing.passed,
                f"frozen verdict FAIL for {criterion!r}: {existing.reasoning}",
            )
            return self

        # Cache miss.
        if mode is Mode.STRICT:
            raise MissingJudgment(
                f"STRICT mode: no frozen verdict for {criterion!r} and live "
                "judging is disabled. Regenerate snapshots in a non-strict "
                "environment."
            )
        if mode is Mode.REPLAY:
            raise MissingJudgment(
                f"No frozen verdict for {criterion!r}. Re-run with "
                "FAKELLM_ASSERT_MODE=update (or `pytest --fakellm-update`) to "
                "judge and freeze it."
            )

        # UPDATE mode: produce, freeze, then assert.
        if cfg.judge is None:
            raise RuntimeError(
                "UPDATE mode requires a judge. Call "
                "fakellm_assert.configure(judge=...) first."
            )
        result = cfg.judge.judge(self._n.text, criterion)
        judgment = Judgment(
            fingerprint=fp,
            criterion=criterion,
            verdict="pass" if result.passed else "fail",
            reasoning=result.reasoning,
            judge_model=cfg.judge.model_name,
            response_excerpt=self._n.text[:280],
            judged_at=now_iso(),
        )
        store.put(judgment)
        store.flush()
        _truthy_or_fail(
            judgment.passed,
            f"judge FAIL for {criterion!r}: {judgment.reasoning}",
        )
        return self


def _snapshot_path_for(criterion: str, snapshot_dir: str) -> str:
    # One file per snapshot dir keeps it simple for v0; pytest plugin can
    # override to one-file-per-test-module.
    return f"{snapshot_dir}/judgments.json"


def expect(response: Any) -> Expectation:
    """Entry point. Wrap any LLM response to start asserting."""
    return Expectation(response)
