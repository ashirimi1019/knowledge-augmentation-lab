from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^\s*uses:\s*[\w.-]+/[\w.-]+(?:/[\w.-]+)?@([0-9a-f]{40})\s+#\s+\S+\s*$")


def test_all_external_actions_are_pinned_to_full_commit_shas() -> None:
    uses_lines = [
        line
        for workflow in WORKFLOW_DIR.glob("*.yml")
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("uses:")
    ]

    assert uses_lines
    assert all(PINNED_ACTION.match(line) for line in uses_lines), uses_lines


def test_ci_declares_every_supported_python_version_and_quality_gate() -> None:
    workflow = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f'"{version}"' in workflow
    for command in (
        "ruff check .",
        "ruff format --check .",
        "pyright",
        "pytest --cov=knowledge_aug_lab --cov-branch",
    ):
        assert command in workflow


def test_ci_builds_validates_installs_and_smoke_tests_both_distributions() -> None:
    workflow = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "dist/*.whl" in workflow
    assert "dist/*.tar.gz" in workflow
    assert ".venv-wheel/bin/kal catalog --json" in workflow
    assert ".venv-sdist/bin/kal showcase" in workflow


def test_security_automation_covers_code_scanning_and_dependency_updates() -> None:
    codeql = (WORKFLOW_DIR / "codeql.yml").read_text(encoding="utf-8")
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    assert "security-events: write" in codeql
    assert "github/codeql-action/init@" in codeql
    assert "github/codeql-action/analyze@" in codeql
    assert "package-ecosystem: pip" in dependabot
    assert "package-ecosystem: github-actions" in dependabot
