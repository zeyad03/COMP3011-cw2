"""Smoke test confirming pytest discovery and src package importability."""

import src


def test_package_imports() -> None:
    assert src.__version__ == "0.1.0"
