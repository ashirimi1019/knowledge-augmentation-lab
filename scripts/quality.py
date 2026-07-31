"""Run the complete contributor and release quality gate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from knowledge_aug_lab import __version__

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def isolated_subprocess_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def run(
    *command: str,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def capture_json(
    *command: str,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> object:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def smoke_install(distribution: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="kal-quality-") as directory:
        environment = Path(directory)
        venv.EnvBuilder(with_pip=True).create(environment)
        executable_directory = environment / ("Scripts" if os.name == "nt" else "bin")
        python = executable_directory / ("python.exe" if os.name == "nt" else "python")
        kal = executable_directory / ("kal.exe" if os.name == "nt" else "kal")
        clean_environment = isolated_subprocess_environment()
        run(
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(distribution),
            cwd=environment,
            env=clean_environment,
        )
        version_result = subprocess.run(
            [
                str(python),
                "-c",
                "from importlib.metadata import version; "
                "import knowledge_aug_lab; "
                "print(version('knowledge-augmentation-lab')); "
                "print(knowledge_aug_lab.__version__)",
            ],
            cwd=environment,
            env=clean_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        if version_result.stdout.splitlines() != [__version__, __version__]:
            raise RuntimeError(f"installed version mismatch for {distribution.name}")
        catalog = capture_json(str(kal), "catalog", "--json", cwd=environment, env=clean_environment)
        showcase = capture_json(str(kal), "showcase", cwd=environment, env=clean_environment)
        evaluation = capture_json(str(kal), "evaluate", "--check", cwd=environment, env=clean_environment)
        if not isinstance(catalog, list) or not catalog:
            raise RuntimeError(f"catalog smoke test returned invalid JSON for {distribution.name}")
        if not isinstance(showcase, dict) or "naive-rag" not in showcase:
            raise RuntimeError(f"showcase smoke test returned invalid JSON for {distribution.name}")
        if not isinstance(evaluation, dict) or evaluation.get("baseline_status") != "passed":
            raise RuntimeError(f"evaluation baseline failed for {distribution.name}")


def main() -> int:
    run(sys.executable, "-m", "ruff", "check", ".")
    run(sys.executable, "-m", "ruff", "format", "--check", ".")
    run(sys.executable, "-m", "pyright")
    run(
        sys.executable,
        "-m",
        "pytest",
        "--cov=knowledge_aug_lab",
        "--cov-branch",
        "--cov-report=term-missing",
    )

    shutil.rmtree(DIST, ignore_errors=True)
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    run(sys.executable, "-m", "build")
    distributions = sorted(DIST.iterdir())
    wheels = [path for path in distributions if path.suffix == ".whl"]
    sources = [path for path in distributions if path.name.endswith(".tar.gz")]
    if len(distributions) != 2 or len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError("build must produce exactly one wheel and one source distribution")
    run(sys.executable, "-m", "twine", "check", "--strict", *(str(path) for path in distributions))

    smoke_install(wheels[0])
    smoke_install(sources[0])
    print("QUALITY_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
