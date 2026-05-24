# fakellm-assert

[![PyPI](https://img.shields.io/pypi/v/fakellm-assert.svg)](https://pypi.org/project/fakellm-assert/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Deterministic semantic assertions for LLM tests. Part of the [fakellm](https://pypi.org/project/fakellm/) family.

Mocking the transport is the easy part of testing LLM apps — `fakellm` already does that. The hard part is asserting on output that's fuzzy by nature ("does the response apologize?", "did it call the right tool?"). `fakellm-assert` gives you a matcher API for exactly that, and keeps your test suite **deterministic and offline** by freezing any judge verdict once and replaying it forever.

```python
from fakellm_assert import expect

expect(resp).contains("refund")
expect(resp).called_tool("issue_refund")
expect(resp).json_path("$.status").equals("ok")
expect(resp).satisfies("apologizes for the delay and offers a solution")
```

`resp` can be a raw string, an OpenAI `ChatCompletion`, or an Anthropic `Message` — they're normalized automatically, with no SDK dependency.

## Installation

```bash
pip install fakellm-assert
```

Requires Python 3.9+. The pytest plugin auto-activates on install — no extra configuration needed to run in `replay` mode. To freeze new verdicts you'll also want to wire up a judge (see [pytest usage](#pytest-usage)); that's the only place an LLM SDK enters the picture, and it's your choice of one.

For a fully reproducible pipeline, pair it with [`fakellm`](https://pypi.org/project/fakellm/) to mock the transport (see [A note on determinism](#a-note-on-determinism)):

```bash
pip install fakellm fakellm-assert
```

## A note on determinism

`fakellm-assert` freezes verdicts about a *specific* response. That only buys you a deterministic test suite if the response itself is deterministic — otherwise every run produces new text, the fingerprint never matches, and `replay` mode hard-errors on a permanent cache miss.

So the response under test must be stable across runs. That's the job of the rest of the family: point your system-under-test at [`fakellm`](https://pypi.org/project/fakellm/) so each call replays a fixed response, and `fakellm-assert` freezes a verdict about that. **Mock the transport with `fakellm`, freeze the judgment with `fakellm-assert`** — together they make a fuzzy LLM pipeline reproducible end to end.

If your SUT calls a live model, expect misses. `--fakellm-update` will still let you judge and freeze a one-off verdict, but it'll go stale the next time the model's output drifts — which is the tool working as intended, not a bug.

## The matcher cascade

Climb only as high as you need. Lower rungs are cheaper and more deterministic; most assertions resolve on the bottom one.

**Tier 1 — deterministic matchers.** Pure functions over the response. `contains`, `not_contains`, `contains_all`, `matches` (regex), `equals`, `has_length`, `is_valid_json`, `json_path(...).equals(...)`, `called_tool`, `tool_args`. Free, instant, 100% deterministic, zero snapshot machinery. **Use these by default** — a surprising amount of "semantic" checking is really structural.

**Tier 3 — frozen judgment.** `satisfies("natural-language criterion")` is the escape hatch for genuinely fuzzy assertions. The verdict comes from a judge model, but only *once*, during an explicit update run — then it's frozen to disk and replayed deterministically. Every `.satisfies()` is a snapshot someone maintains, so reach for it sparingly.

(Tier 2, embedding similarity, is intentionally omitted from v0 — Tiers 1 and 3 straddle the middle ground without the extra dependency and determinism caveats.)

## How freezing works

Each `.satisfies()` assertion has a **fingerprint** — a hash of the response text, the criterion, the judge model, and the prompt template. The verdict is stored under that fingerprint in `.fakellm/judgments/`. Change the response and the fingerprint changes, the old verdict no longer applies, and the test fails until a human re-judges. **That failure is the feature** — verdicts stay valid only for the exact output they were made about.

Three run modes:

| Mode | Cache hit | Cache miss | Network |
|------|-----------|------------|---------|
| **replay** (default) | replay verdict | **hard error** → run update | never |
| **update** | replay verdict | judge live, freeze, assert | judge only |
| **strict** | replay verdict | **hard error**, live judging impossible | never, by construction |

A miss in `replay` never silently calls a model. That's what guarantees CI is deterministic and offline.

## pytest usage

The plugin auto-activates on install.

```bash
pytest                    # replay: frozen verdicts only
pytest --fakellm-update   # judge & freeze any missing verdicts (review the diff!)
pytest --fakellm-strict   # belt-and-suspenders: fail rather than ever judge live
```

Wire up a judge once (in `conftest.py`). It's just a callable returning the model's raw text — bring your own SDK, or point it at fakellm:

```python
from fakellm_assert import configure, CallableJudge
from openai import OpenAI

client = OpenAI()

def run_judge(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

configure(judge=CallableJudge(run_judge, model_name="gpt-4o-mini"))
```

## What this is (and isn't)

This gives you **deterministic regression detection**: it freezes a human-approved verdict and alerts you when output drifts away from it. It does **not** tell you whether your LLM is *correct* — a frozen wrong verdict is still wrong, just consistently so. The judge's reasoning is stored in every snapshot so the git diff tells you *why* a verdict is what it is; read those diffs.

## License

MIT
