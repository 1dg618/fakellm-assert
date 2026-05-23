"""Pytest integration.

Adds a `--fakellm-update` flag. With it, `.satisfies()` cache misses are
judged live and frozen; without it, the default REPLAY mode applies and a
miss fails the test pointing you at the flag. Mirrors the ergonomics of
pytest --snapshot-update from syrupy.

Registered as a setuptools entry point (see pyproject), so it activates
automatically once the package is installed — no conftest wiring needed for
the flag itself.
"""

from __future__ import annotations

from .judge import Mode, configure


def pytest_addoption(parser):
    group = parser.getgroup("fakellm-assert")
    group.addoption(
        "--fakellm-update",
        action="store_true",
        default=False,
        help="Judge and freeze any missing .satisfies() verdicts (UPDATE mode).",
    )
    group.addoption(
        "--fakellm-strict",
        action="store_true",
        default=False,
        help="STRICT mode: fail on missing verdicts; live judging impossible.",
    )


def pytest_configure(config):
    if config.getoption("--fakellm-update"):
        configure(mode=Mode.UPDATE)
    elif config.getoption("--fakellm-strict"):
        configure(mode=Mode.STRICT)
    # else: leave mode unset so env var / default REPLAY applies.
