from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^\s*uses:\s*[\w.-]+/[\w.-]+(?:/[\w.-]+)?@([0-9a-f]{40})\s+#\s+\S+\s*$")


def test_all_external_actions_are_pinned_to_full_commit_shas() -> None:
    workflows = [*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")]
    uses_lines = [
        line
        for workflow in workflows
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("uses:")
    ]

    assert uses_lines
    assert all(PINNED_ACTION.match(line) for line in uses_lines), uses_lines


def test_ci_declares_every_supported_python_version_and_quality_gate() -> None:
    workflow = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    assert "push:\n    branches: [main]" in workflow
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
    assert "python -m twine check --strict dist/*" in workflow
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


def test_release_uses_the_complete_contributor_quality_gate() -> None:
    quality_script = (ROOT / "scripts" / "quality.py").read_text(encoding="utf-8")
    release_workflow = (WORKFLOW_DIR / "release.yml").read_text(encoding="utf-8")

    for command in (
        '"ruff", "check", "."',
        '"ruff", "format", "--check", "."',
        '"pyright"',
        '"pytest"',
        '"--cov=knowledge_aug_lab"',
        '"--cov-branch"',
        '"build"',
        '"twine", "check"',
    ):
        assert command in quality_script
    assert "python scripts/check_release.py" in release_workflow
    assert "python scripts/quality.py" in release_workflow
    assert 'gh release create "$GITHUB_REF_NAME" dist/*' in release_workflow
    assert "cwd=environment" in quality_script


def test_ci_and_release_use_hashed_versioned_and_build_constraints() -> None:
    constraint_directory = ROOT / "constraints"
    development_constraints = {
        version: (constraint_directory / f"py{version.replace('.', '')}.txt").read_text(encoding="utf-8")
        for version in ("3.10", "3.11", "3.12", "3.13")
    }
    build_constraints = (constraint_directory / "build.txt").read_text(encoding="utf-8")
    ci_workflow = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    release_workflow = (WORKFLOW_DIR / "release.yml").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    quality_script = (ROOT / "scripts" / "quality.py").read_text(encoding="utf-8")
    compiler = (ROOT / "scripts" / "compile_constraints.py").read_text(encoding="utf-8")

    for version, constraints in development_constraints.items():
        assert f"--python-version {version}" in constraints
        for dependency in (
            "build",
            "hypothesis",
            "pandas",
            "pyright",
            "pytest",
            "pytest-cov",
            "ruff",
            "streamlit",
            "twine",
        ):
            assert re.search(rf"^{dependency.replace('-', '[-_]')}==", constraints, re.MULTILINE)
        assert "--hash=sha256:" in constraints
        assert version in compiler
    for dependency in ("setuptools", "wheel"):
        assert re.search(rf"^{dependency}==", build_constraints, re.MULTILINE)
    assert "--hash=sha256:" in build_constraints
    for constraint in ("py310.txt", "py311.txt", "py312.txt", "py313.txt"):
        assert constraint in ci_workflow
    assert ci_workflow.count("--require-hashes -r constraints/build.txt") == 4
    assert ci_workflow.count("--no-build-isolation --no-deps -e .") == 2
    assert release_workflow.count("--require-hashes -r constraints/build.txt") == 1
    assert "--require-hashes -r constraints/py313.txt" in release_workflow
    assert release_workflow.count("--no-build-isolation --no-deps -e .") == 1
    assert '"build", "--no-isolation"' in quality_script
    assert '"--no-build-isolation"' in quality_script
    assert '"pytest>=8.0"' in project
    assert '"twine>=6.2,<7"' in project


def test_public_security_policy_documents_private_reporting_and_response_targets() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "GitHub private vulnerability reporting" in policy
    assert "/security/advisories/new" in policy
    assert "0.2.x" in policy
    assert "three business days" in policy
    assert "not guaranteed resolution deadlines" in policy


def test_public_setup_and_catalog_surfaces_use_current_security_boundaries() -> None:
    public_setup_paths = (
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "requirements.txt",
        ROOT / "site" / "index.html",
    )
    public_claim_paths = (
        ROOT / "README.md",
        ROOT / "docs" / "taxonomy.md",
        ROOT / "docs" / "threat-model.md",
        ROOT / "site" / "index.html",
        ROOT / "src" / "knowledge_aug_lab" / "catalog.py",
    )
    setup_text = {path: path.read_text(encoding="utf-8") for path in public_setup_paths}
    claim_text = {path: path.read_text(encoding="utf-8") for path in public_claim_paths}
    readme = setup_text[ROOT / "README.md"]
    site = setup_text[ROOT / "site" / "index.html"]
    rendered_site = re.sub(r"<[^>]+>", "", site)

    for path, content in setup_text.items():
        assert not re.search(r"(?:python -m )?pip install -e", content), path
    for path, content in claim_text.items():
        assert "no value schema" not in content.casefold(), path
    assert "constraints/build.txt" in readme
    assert "constraints/py311.txt" in readme
    assert "--no-build-isolation --no-deps -e ." in readme
    assert "finite JSON-compatible" in readme
    assert "constraints/build.txt" in site
    assert "constraints/py311.txt" in site
    assert "--no-build-isolation --no-deps -e ." in rendered_site
