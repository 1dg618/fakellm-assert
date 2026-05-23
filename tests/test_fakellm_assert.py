"""Tests for fakellm-assert.

Covers response normalization, every Tier 1 matcher, and the full
freeze/replay/strict lifecycle of .satisfies() using a fake deterministic
judge so the suite itself stays offline.
"""

import json

import pytest

from fakellm_assert import (
    AssertionFailure,
    CallableJudge,
    Mode,
    MissingJudgment,
    configure,
    expect,
)
from fakellm_assert.judge import get_config


# ---- fixtures: sample provider responses -------------------------------

OPENAI_RESP = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "I'm sorry for the delay. Your refund is processed.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "issue_refund",
                            "arguments": '{"order_id": "A123", "amount": 49.99}',
                        },
                    }
                ],
            }
        }
    ]
}

ANTHROPIC_RESP = {
    "content": [
        {"type": "text", "text": "Apologies for the wait. "},
        {"type": "text", "text": "Refund done."},
        {"type": "tool_use", "name": "issue_refund", "input": {"order_id": "B456"}},
    ]
}


# ---- extraction --------------------------------------------------------

def test_extract_openai_text_and_tools():
    e = expect(OPENAI_RESP)
    e.contains("refund").called_tool("issue_refund")


def test_extract_anthropic_concatenates_text_blocks():
    expect(ANTHROPIC_RESP).contains("Apologies for the wait").contains("Refund done")


def test_extract_raw_string():
    expect("hello world").contains("hello")


# ---- Tier 1 text matchers ----------------------------------------------

def test_contains_case_insensitive():
    expect("REFUND ISSUED").contains("refund", case_sensitive=False)


def test_contains_failure():
    with pytest.raises(AssertionFailure):
        expect("nope").contains("refund")


def test_not_contains():
    expect("all good").not_contains("error")
    with pytest.raises(AssertionFailure):
        expect("there was an error").not_contains("error")


def test_contains_all():
    expect("a b c").contains_all("a", "b", "c")
    with pytest.raises(AssertionFailure):
        expect("a b").contains_all("a", "z")


def test_matches_regex():
    expect("date 2026-05-23 ok").matches(r"\d{4}-\d{2}-\d{2}")
    with pytest.raises(AssertionFailure):
        expect("no date").matches(r"\d{4}")


def test_has_length():
    expect("hi").has_length(lte=5)
    with pytest.raises(AssertionFailure):
        expect("way too long string here").has_length(lte=3)


# ---- Tier 1 structural matchers ----------------------------------------

def test_is_valid_json():
    expect('{"ok": true}').is_valid_json()
    with pytest.raises(AssertionFailure):
        expect("not json").is_valid_json()


def test_json_path_equals():
    resp = json.dumps({"status": "ok", "items": [{"id": 7}]})
    expect(resp).json_path("$.status").equals("ok")
    expect(resp).json_path("$.items[0].id").equals(7)


def test_json_path_missing_fails():
    with pytest.raises(AssertionFailure):
        expect('{"a": 1}').json_path("$.b").exists()


# ---- Tier 1 tool matchers ----------------------------------------------

def test_tool_args():
    expect(OPENAI_RESP).tool_args("issue_refund").value
    args = expect(OPENAI_RESP).tool_args("issue_refund").value
    assert args["order_id"] == "A123"


def test_called_tool_failure():
    with pytest.raises(AssertionFailure):
        expect(OPENAI_RESP).called_tool("send_email")


# ---- Tier 3 frozen-judgment lifecycle ----------------------------------

class _FakeJudge:
    """Deterministic stand-in for a real model. Passes iff 'sorry' appears."""

    model_name = "fake-judge-v1"

    def judge(self, response_text, criterion):
        from fakellm_assert import JudgeResult

        passed = "sorry" in response_text.lower()
        return JudgeResult(passed=passed, reasoning="contains apology" if passed else "no apology")


@pytest.fixture
def isolated_snapshots(tmp_path):
    """Point the snapshot dir at a temp location and reset config after."""
    cfg = get_config()
    saved = (cfg.mode, cfg.judge, cfg.snapshot_dir)
    configure(snapshot_dir=str(tmp_path / "judgments"), judge=_FakeJudge())
    yield tmp_path
    cfg.mode, cfg.judge, cfg.snapshot_dir = saved


def test_replay_miss_is_hard_error(isolated_snapshots):
    configure(mode=Mode.REPLAY)
    with pytest.raises(MissingJudgment):
        expect("I'm sorry").satisfies("apologizes")


def test_strict_miss_is_hard_error(isolated_snapshots):
    configure(mode=Mode.STRICT)
    with pytest.raises(MissingJudgment):
        expect("I'm sorry").satisfies("apologizes")


def test_update_freezes_then_replay_reads(isolated_snapshots):
    # 1. UPDATE judges live and freezes a passing verdict.
    configure(mode=Mode.UPDATE)
    expect("I'm sorry for the delay").satisfies("apologizes")

    # 2. REPLAY now finds the frozen verdict — no judge call needed.
    configure(mode=Mode.REPLAY)
    expect("I'm sorry for the delay").satisfies("apologizes")  # passes from cache


def test_update_freezes_failing_verdict(isolated_snapshots):
    configure(mode=Mode.UPDATE)
    with pytest.raises(AssertionFailure):
        expect("no apology here").satisfies("apologizes")  # judge fails it
    # And the failing verdict replays as a failure too.
    configure(mode=Mode.REPLAY)
    with pytest.raises(AssertionFailure):
        expect("no apology here").satisfies("apologizes")


def test_changed_response_invalidates_verdict(isolated_snapshots):
    # Freeze a verdict for one response...
    configure(mode=Mode.UPDATE)
    expect("I'm sorry").satisfies("apologizes")
    # ...a different response has a different fingerprint -> cache miss in REPLAY.
    configure(mode=Mode.REPLAY)
    with pytest.raises(MissingJudgment):
        expect("I'm SORRY now").satisfies("apologizes")


def test_callable_judge_parses_json():
    def fake_model(prompt):
        return '```json\n{"pass": true, "reasoning": "ok"}\n```'

    judge = CallableJudge(fake_model, model_name="m")
    result = judge.judge("text", "criterion")
    assert result.passed and result.reasoning == "ok"
