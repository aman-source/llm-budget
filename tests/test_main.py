"""Tests for __main__.py module."""
from __future__ import annotations

import subprocess
import sys


def test_python_m_llm_budget_help():
    """Verify `python -m llm_budget --help` works."""
    result = subprocess.run(
        [sys.executable, "-m", "llm_budget", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "llm-budget" in result.stdout


def test_python_m_llm_budget_version():
    """Verify `python -m llm_budget --version` works."""
    result = subprocess.run(
        [sys.executable, "-m", "llm_budget", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout
