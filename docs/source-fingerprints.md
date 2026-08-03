# Reproducible source fingerprints

Post-release reviews identify an immutable Git commit and its tree with a canonical source fingerprint. The fingerprint is independent of the working directory, checkout line-ending settings, file timestamps, archive metadata, untracked files, and ignored build output.

## Reproduce or verify a fingerprint

Run the script from any checkout that contains the requested Git object. Supply a full commit object ID: 40 hexadecimal characters for a SHA-1 repository or 64 for a SHA-256 repository. The required length is derived from the repository's Git object format. Mutable names such as `HEAD`, branches, and abbreviated IDs are rejected.

```bash
python scripts/source_fingerprint.py <full-commit-sha>
python scripts/source_fingerprint.py <full-commit-sha> --verify <expected-64-character-sha256>
```

Use `--repository <path>` when the target repository is not the current directory. The command prints `repository_commit_sha`, `tree_sha`, `file_count`, `hashing_algorithm`, `fingerprint_format`, and `digest`. Verification also prints `verification=passed` or `verification=failed`; a mismatch exits with status 1. Invalid arguments or Git objects exit through argparse with status 2.

## Canonical algorithm: `sha256 / kal-git-tree-v1`

1. Set `GIT_NO_REPLACE_OBJECTS=1` for every Git subprocess so local replacement refs under `.git` cannot rewrite commit, tree, or blob objects. Verify that the supplied value is a full hexadecimal Git object ID and that `git cat-file -t` identifies it as a commit.
2. Resolve the commit and its root tree SHA. Read recursive tree entries with `git ls-tree -r -z --full-tree <tree-sha>`.
3. Require every leaf entry to be a Git blob. Gitlinks or other non-blob leaves fail closed. Tracked symlinks are blobs and contribute their stored link-target bytes. Tracked ignored files are included because they are in the tree; untracked and ignored-but-untracked files are absent.
4. Read every blob with `git cat-file blob <object-id>`. These are the exact Git blob bytes. The script does not read checkout files and does not normalize line endings, encodings, Unicode, or file contents.
5. Sort entries lexicographically by their raw Git path bytes. Git paths are not decoded before sorting or hashing.
6. Hash this byte stream with SHA-256:

```text
ASCII/UTF-8 bytes: "knowledge-augmentation-lab/source-fingerprint/v1" followed by NUL
u64be(number of entries)
for each path-sorted entry:
    u64be(length(mode))    || raw ASCII Git mode bytes
    u64be(length(path))    || raw Git path bytes
    u64be(length(content)) || exact Git blob bytes
```

`u64be` is an unsigned 64-bit big-endian integer. Every variable-length field is length-prefixed, the entry count is included, and duplicate paths fail closed. The Git mode is included so executable-bit and symlink-mode changes alter the digest. No path separator conversion is performed.

## PR #10 audit record

The approved value `d9150a163a8ba0558cd567b28197a65c0756e3b8322b082a069a76013d6ae99f` was a SHA-256 over an external staged-diff byte stream. The repository did not preserve that stream as a canonical source-tree algorithm. It is therefore recorded only as a **historical external attestation** and is not claimed as reproduced by `kal-git-tree-v1`.

The first reviewed PR #10 head commit was `f245702534a2e31c771938b29ce0f93185e3b9e0`. Its canonical source identity is:

- Repository commit SHA: `f245702534a2e31c771938b29ce0f93185e3b9e0`
- Tree SHA: `d05edf60abe6edc2c4f8d5d1769da12f26fb3007`
- File count: `91`
- Hashing algorithm: `sha256`
- Fingerprint format: `kal-git-tree-v1`
- Digest: `3e7fe7f9c9f9067fc9014b3f929810f54573861f0ce81da3a7a50d26dff963d2`

Exact verification command:

```bash
python scripts/source_fingerprint.py f245702534a2e31c771938b29ce0f93185e3b9e0 --verify 3e7fe7f9c9f9067fc9014b3f929810f54573861f0ce81da3a7a50d26dff963d2
```

## Audit attestations

Copy [`audit-attestation-template.json`](audit-attestation-template.json), replace every angle-bracket placeholder, and retain the exact command used. `signature_reference` may remain `null` or identify a detached signature, signed commit, transparency-log entry, or other verifiable signature artifact. The fingerprint binds the reviewed head tree; the attestation separately records its base, reviewer, date, and optional signature.
