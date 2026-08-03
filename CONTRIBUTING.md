# Contributing

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --require-hashes -r constraints/build.txt
# Choose the file matching the active interpreter: py310, py311, py312, or py313.
python -m pip install --require-hashes -r constraints/py311.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## One-command quality gate

Run the same complete gate used before a release:

```bash
python scripts/quality.py
```

It runs Ruff lint/format checks, strict Pyright, branch-enabled pytest coverage, wheel and sdist builds, strict Twine metadata validation, isolated installation of both artifacts, installed-version checks, and JSON smoke tests for `kal catalog`, `kal showcase`, and `kal evaluate --check`.

The script is cross-platform and uses the active `sys.executable`; do not replace it with shell-specific paths or globs.

`constraints/py310.txt` through `constraints/py313.txt` pin and hash the cross-platform development and application graph separately for every supported interpreter. `constraints/build.txt` independently pins the PEP 517 build tools. CI exercises each version-specific graph; builds disable isolation only after the hashed build tools are installed. `pyproject.toml` retains broad compatibility ranges for package consumers. The local package has no runtime dependencies, so it is installed separately with `--no-deps`; editable local projects cannot satisfy pip's artifact-hash requirement. Regenerate every snapshot after intentional dependency changes with:

```bash
python scripts/compile_constraints.py
```

## Change workflow

1. Add the narrowest failing behavior test first.
2. Confirm it fails for the intended reason.
3. Implement the smallest robust change.
4. Run focused tests, then `python scripts/quality.py`.
5. Update threat, evaluation, or fidelity documentation when a public claim changes.
6. Keep generated `dist/`, `build/`, and coverage files out of commits. Commit `constraints/` only when intentionally refreshing the toolchain.

Pull requests must keep required GitHub checks green. `main` requires strict up-to-date status checks, linear history, resolved conversations, and immutable Action SHAs.

## Evaluation changes

Never update the packaged baseline solely to make CI pass. Explain case-level metric changes and version the fixture when corpus, labels, or conventions change. See [Reproducible evaluation](docs/evaluation.md).

## Release contract

Releases use tags of the exact form `vMAJOR.MINOR.PATCH`. The tag must equal both project/package version declarations. The release workflow re-runs the complete quality gate and publishes the wheel and sdist to a GitHub Release. PyPI publication is intentionally not configured.
