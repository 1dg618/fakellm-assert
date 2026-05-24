"""fakellm-assert: deterministic semantic assertions for LLM tests.

Tier 1 matchers run as pure deterministic functions. `.satisfies()` freezes
a judge's verdict once and replays it forever, keeping CI offline and stable.

    from fakellm_assert import expect, configure, CallableJudge

    expect(resp).contains("refund")
    expect(resp).called_tool("issue_refund")
    expect(resp).satisfies("apologizes for the delay")
"""

from .expect import (
    AssertionFailure,
    Expectation,
    MissingJudgment,
    expect,
)
from .judge import (
    CallableJudge,
    Judge,
    JudgeResult,
    Mode,
    configure,
)

__version__ = "0.1.1"

__all__ = [
    "expect",
    "Expectation",
    "AssertionFailure",
    "MissingJudgment",
    "configure",
    "CallableJudge",
    "Judge",
    "JudgeResult",
    "Mode",
    "__version__",
]
