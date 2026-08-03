"""Regenerate hashed development and build dependency snapshots."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13")


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, shell=False, check=True)


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to compile constraints")

    for version in PYTHON_VERSIONS:
        output = f"constraints/py{version.replace('.', '')}.txt"
        run(
            uv,
            "pip",
            "compile",
            "pyproject.toml",
            "--extra",
            "dev",
            "--extra",
            "app",
            "--python-version",
            version,
            "--universal",
            "--generate-hashes",
            "--output-file",
            output,
        )
    run(
        uv,
        "pip",
        "compile",
        "requirements-build.in",
        "--universal",
        "--generate-hashes",
        "--output-file",
        "constraints/build.txt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
