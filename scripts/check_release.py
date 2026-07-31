"""Validate that a release tag and both package versions agree."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from knowledge_aug_lab import __version__

_SEMANTIC_TAG = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_PROJECT_VERSION = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)
ROOT = Path(__file__).resolve().parents[1]


def validate_release_tag(tag: str, package_version: str) -> str:
    match = _SEMANTIC_TAG.fullmatch(tag)
    if match is None:
        raise ValueError("release tag must match vMAJOR.MINOR.PATCH without prerelease metadata")
    tag_version = ".".join(match.groups())
    if tag_version != package_version:
        raise ValueError(f"release tag {tag!r} does not match package version {package_version!r}")
    return tag_version


def read_project_version() -> str:
    match = _PROJECT_VERSION.search((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("pyproject.toml must contain a project version")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise ValueError("usage: python scripts/check_release.py vMAJOR.MINOR.PATCH")
    project_version = read_project_version()
    if project_version != __version__:
        raise ValueError("pyproject.toml and knowledge_aug_lab.__version__ must match")
    version = validate_release_tag(arguments[0], project_version)
    print(f"RELEASE_VERSION_VALID {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
