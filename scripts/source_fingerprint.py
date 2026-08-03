"""Compute deterministic fingerprints from immutable Git tree objects."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

HASHING_ALGORITHM = "sha256"
FINGERPRINT_FORMAT = "kal-git-tree-v1"
_MAGIC = b"knowledge-augmentation-lab/source-fingerprint/v1\x00"
_OBJECT_ID_LENGTHS = {"sha1": 40, "sha256": 64}


@dataclass(frozen=True, slots=True)
class GitBlob:
    """One tracked blob entry as represented by a Git tree."""

    mode: bytes
    path: bytes
    content: bytes


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Resolved commit/tree identity and its canonical source digest."""

    commit_sha: str
    tree_sha: str
    file_count: int
    digest: str


def _git(repository: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        env=environment,
    )
    return completed.stdout


def _length_prefix(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def fingerprint_entries(entries: Iterable[GitBlob]) -> str:
    """Hash entries in bytewise path order using the versioned canonical stream."""

    ordered = sorted(tuple(entries), key=lambda entry: entry.path)
    if len({entry.path for entry in ordered}) != len(ordered):
        raise ValueError("Git tree contains duplicate paths")

    digest = hashlib.sha256()
    digest.update(_MAGIC)
    digest.update(struct.pack(">Q", len(ordered)))
    for entry in ordered:
        digest.update(_length_prefix(entry.mode))
        digest.update(_length_prefix(entry.path))
        digest.update(_length_prefix(entry.content))
    return digest.hexdigest()


def _resolve_commit(repository: Path, commit: str) -> tuple[str, str]:
    object_format = _git(repository, "rev-parse", "--show-object-format=storage").strip().decode("ascii")
    try:
        object_id_length = _OBJECT_ID_LENGTHS[object_format]
    except KeyError as exc:
        raise ValueError(f"unsupported Git object format: {object_format}") from exc
    if len(commit) != object_id_length or not re.fullmatch(r"[0-9a-fA-F]+", commit):
        raise ValueError(
            f"commit must be a full {object_id_length}-character hexadecimal Git object ID "
            f"for a {object_format} repository"
        )
    normalized = commit.lower()
    object_type = _git(repository, "cat-file", "-t", normalized).strip().decode("ascii")
    if object_type != "commit":
        raise ValueError("Git object must be a commit")
    commit_sha = _git(repository, "rev-parse", normalized).strip().decode("ascii")
    if commit_sha != normalized:
        raise ValueError("resolved commit object ID does not match the supplied object ID")
    tree_sha = _git(repository, "rev-parse", f"{commit_sha}^{{tree}}").strip().decode("ascii")
    return commit_sha, tree_sha


def _read_tree_entries(repository: Path, tree_sha: str) -> tuple[GitBlob, ...]:
    output = _git(repository, "ls-tree", "-r", "-z", "--full-tree", tree_sha)
    entries: list[GitBlob] = []
    for record in output.split(b"\x00"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ", 2)
        if object_type != b"blob":
            display_path = path.decode("utf-8", errors="backslashreplace")
            raise ValueError(f"unsupported non-blob tree entry: {display_path}")
        content = _git(repository, "cat-file", "blob", object_id.decode("ascii"))
        entries.append(GitBlob(mode=mode, path=path, content=content))
    return tuple(entries)


def compute_source_fingerprint(repository: Path, commit: str) -> SourceFingerprint:
    """Resolve an explicit commit and hash the exact blobs in its tree."""

    commit_sha, tree_sha = _resolve_commit(repository, commit)
    entries = _read_tree_entries(repository, tree_sha)
    return SourceFingerprint(
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        file_count=len(entries),
        digest=fingerprint_entries(entries),
    )


def _expected_digest(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("expected digest must be 64 hexadecimal characters")
    return value.lower()


def _print_fingerprint(fingerprint: SourceFingerprint) -> None:
    print(f"repository_commit_sha={fingerprint.commit_sha}")
    print(f"tree_sha={fingerprint.tree_sha}")
    print(f"file_count={fingerprint.file_count}")
    print(f"hashing_algorithm={HASHING_ALGORITHM}")
    print(f"fingerprint_format={FINGERPRINT_FORMAT}")
    print(f"digest={fingerprint.digest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fingerprint the exact tracked blobs in an explicit Git commit tree.")
    parser.add_argument("commit", help="full Git commit object ID")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="repository path (default: current directory)",
    )
    parser.add_argument(
        "--verify",
        type=_expected_digest,
        metavar="SHA256",
        help="exit nonzero unless the computed digest matches SHA256",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        fingerprint = compute_source_fingerprint(arguments.repository, arguments.commit)
    except ValueError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        parser.error(detail or "Git command failed")

    _print_fingerprint(fingerprint)
    if arguments.verify is None:
        return 0
    if fingerprint.digest == arguments.verify:
        print("verification=passed")
        return 0
    print("verification=failed")
    print(
        f"fingerprint mismatch: expected {arguments.verify}, computed {fingerprint.digest}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
