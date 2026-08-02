from __future__ import annotations

import pytest

from knowledge_aug_lab import __version__
from scripts.check_release import read_project_version, validate_release_tag


def test_release_tag_matches_package_version() -> None:
    assert validate_release_tag("v0.2.0", __version__) == "0.2.0"
    assert read_project_version() == __version__


@pytest.mark.parametrize("tag", ["0.2.0", "v0.2", "v0.2.0-rc1", "v01.2.0", "latest"])
def test_release_tag_rejects_non_semantic_versions(tag: str) -> None:
    with pytest.raises(ValueError, match="release tag must match"):
        validate_release_tag(tag, "0.2.0")


def test_release_tag_rejects_package_version_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match package version"):
        validate_release_tag("v0.2.0", "0.1.0")
