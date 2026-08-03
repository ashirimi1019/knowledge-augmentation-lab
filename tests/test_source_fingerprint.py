from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import source_fingerprint


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _new_repository(path: Path, *, object_format: str | None = None) -> Path:
    path.mkdir()
    arguments = ["init", "--quiet"]
    if object_format is not None:
        arguments.append(f"--object-format={object_format}")
    _git(path, *arguments)
    _git(path, "config", "user.name", "Fingerprint Test")
    _git(path, "config", "user.email", "fingerprint@example.invalid")
    return path


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "--allow-empty", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def test_identical_git_trees_produce_identical_fingerprints(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path / "repository")
    (repository / "source.txt").write_bytes(b"exact\r\nblob\nbytes\x00")
    first_commit = _commit(repository, "first")
    second_commit = _commit(repository, "same tree, different commit")

    first = source_fingerprint.compute_source_fingerprint(repository, first_commit)
    second = source_fingerprint.compute_source_fingerprint(repository, second_commit)

    assert first.commit_sha != second.commit_sha
    assert first.tree_sha == second.tree_sha
    assert first.file_count == second.file_count == 1
    assert first.digest == second.digest


def test_file_ordering_does_not_change_fingerprint() -> None:
    first = source_fingerprint.GitBlob(mode=b"100644", path=b"z.txt", content=b"last")
    second = source_fingerprint.GitBlob(mode=b"100755", path=b"a.txt", content=b"first")

    assert source_fingerprint.fingerprint_entries([first, second]) == source_fingerprint.fingerprint_entries(
        [second, first]
    )


def test_canonical_stream_matches_known_digest() -> None:
    entries = [
        source_fingerprint.GitBlob(mode=b"100644", path=b"a.txt", content=b"A\r\n"),
        source_fingerprint.GitBlob(mode=b"100755", path=b"z.bin", content=b"\x00\xff"),
    ]

    assert (
        source_fingerprint.fingerprint_entries(entries)
        == "a175d12703250af637283bcb01eaaa3d4cb36cda058a1f5d62ef0be6d8b42f79"
    )


def test_changing_one_blob_byte_changes_fingerprint(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path / "repository")
    source = repository / "source.txt"
    source.write_bytes(b"value-A")
    first_commit = _commit(repository, "first byte value")
    source.write_bytes(b"value-B")
    second_commit = _commit(repository, "second byte value")

    first = source_fingerprint.compute_source_fingerprint(repository, first_commit)
    second = source_fingerprint.compute_source_fingerprint(repository, second_commit)

    assert first.digest != second.digest


def test_changing_tracked_path_changes_fingerprint(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path / "repository")
    (repository / "first.txt").write_bytes(b"unchanged content")
    first_commit = _commit(repository, "first path")
    _git(repository, "mv", "first.txt", "second.txt")
    second_commit = _commit(repository, "second path")

    first = source_fingerprint.compute_source_fingerprint(repository, first_commit)
    second = source_fingerprint.compute_source_fingerprint(repository, second_commit)

    assert first.digest != second.digest


def test_untracked_files_do_not_affect_commit_fingerprint(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path / "repository")
    (repository / "tracked.txt").write_bytes(b"tracked")
    commit = _commit(repository, "tracked source")
    before = source_fingerprint.compute_source_fingerprint(repository, commit)
    (repository / "untracked-generated.bin").write_bytes(b"not part of the Git tree")

    after = source_fingerprint.compute_source_fingerprint(repository, commit)

    assert before == after


def test_git_replacement_refs_do_not_affect_commit_fingerprint(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path / "repository")
    tracked = repository / "tracked.txt"
    tracked.write_bytes(b"reviewed bytes")
    reviewed_commit = _commit(repository, "reviewed source")
    before = source_fingerprint.compute_source_fingerprint(repository, reviewed_commit)
    reviewed_blob = _git(repository, "rev-parse", f"{reviewed_commit}:tracked.txt")

    tracked.write_bytes(b"replacement bytes")
    replacement_commit = _commit(repository, "replacement source")
    replacement_blob = _git(repository, "rev-parse", f"{replacement_commit}:tracked.txt")
    _git(repository, "replace", reviewed_blob, replacement_blob)

    after = source_fingerprint.compute_source_fingerprint(repository, reviewed_commit)

    assert after == before


def test_verification_mode_succeeds_and_prints_machine_readable_identity(tmp_path: Path, capsys) -> None:
    repository = _new_repository(tmp_path / "repository")
    (repository / "tracked.txt").write_bytes(b"tracked")
    commit = _commit(repository, "tracked source")
    fingerprint = source_fingerprint.compute_source_fingerprint(repository, commit)

    exit_code = source_fingerprint.main([commit, "--repository", str(repository), "--verify", fingerprint.digest])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert f"repository_commit_sha={commit}" in captured.out
    assert f"tree_sha={fingerprint.tree_sha}" in captured.out
    assert "file_count=1" in captured.out
    assert "hashing_algorithm=sha256" in captured.out
    assert "fingerprint_format=kal-git-tree-v1" in captured.out
    assert f"digest={fingerprint.digest}" in captured.out
    assert "verification=passed" in captured.out


def test_verification_mode_fails_for_mismatched_digest(tmp_path: Path, capsys) -> None:
    repository = _new_repository(tmp_path / "repository")
    (repository / "tracked.txt").write_bytes(b"tracked")
    commit = _commit(repository, "tracked source")

    exit_code = source_fingerprint.main([commit, "--repository", str(repository), "--verify", "0" * 64])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "verification=failed" in captured.out
    assert "fingerprint mismatch" in captured.err


@pytest.mark.parametrize("revision", ["HEAD", "main", "", "deadbeef"])
def test_fingerprint_requires_a_full_commit_object_id(tmp_path: Path, revision: str) -> None:
    repository = _new_repository(tmp_path / "repository")

    with pytest.raises(ValueError, match="full 40-character hexadecimal Git object ID"):
        source_fingerprint.compute_source_fingerprint(repository, revision)


def test_sha256_repository_rejects_a_40_character_commit_prefix(tmp_path: Path) -> None:
    repository = _new_repository(tmp_path / "repository", object_format="sha256")
    (repository / "tracked.txt").write_bytes(b"tracked")
    commit = _commit(repository, "sha256 commit")
    assert len(commit) == 64

    with pytest.raises(ValueError, match="full 64-character hexadecimal Git object ID"):
        source_fingerprint.compute_source_fingerprint(repository, commit[:40])

    assert source_fingerprint.compute_source_fingerprint(repository, commit).commit_sha == commit


def test_audit_attestation_template_is_machine_parseable_and_complete() -> None:
    template = json.loads(Path("docs/audit-attestation-template.json").read_text(encoding="utf-8"))

    assert template == {
        "schema_version": 1,
        "base_commit_sha": "<full Git commit SHA>",
        "reviewed_head_commit_sha": "<full Git commit SHA>",
        "tree_sha": "<full Git tree SHA>",
        "file_count": "<nonnegative integer>",
        "hashing_algorithm": "sha256",
        "fingerprint_format": "kal-git-tree-v1",
        "fingerprint_value": "<64 lowercase hexadecimal characters>",
        "exact_command": "<exact source_fingerprint.py command>",
        "reviewer_identity": "<reviewer name or stable identifier>",
        "review_date": "<YYYY-MM-DD>",
        "signature_reference": None,
    }


def test_fingerprint_documentation_records_historical_and_canonical_values() -> None:
    documentation = Path("docs/source-fingerprints.md").read_text(encoding="utf-8")

    assert "d9150a163a8ba0558cd567b28197a65c0756e3b8322b082a069a76013d6ae99f" in documentation
    assert "historical external attestation" in documentation
    assert "f245702534a2e31c771938b29ce0f93185e3b9e0" in documentation
    assert "d05edf60abe6edc2c4f8d5d1769da12f26fb3007" in documentation
    assert "3e7fe7f9c9f9067fc9014b3f929810f54573861f0ce81da3a7a50d26dff963d2" in documentation
    assert (
        "python scripts/source_fingerprint.py f245702534a2e31c771938b29ce0f93185e3b9e0 "
        "--verify 3e7fe7f9c9f9067fc9014b3f929810f54573861f0ce81da3a7a50d26dff963d2" in documentation
    )
