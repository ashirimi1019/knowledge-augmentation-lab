from pathlib import Path

from scripts import quality


def test_isolated_subprocess_environment_removes_python_overrides_without_mutating_source() -> None:
    source = {"PATH": "bin", "PYTHONHOME": "parent-home", "PYTHONPATH": "parent-path"}

    result = quality.isolated_subprocess_environment(source)

    assert result == {"PATH": "bin"}
    assert source == {"PATH": "bin", "PYTHONHOME": "parent-home", "PYTHONPATH": "parent-path"}


def test_quality_main_reaches_every_gate_and_smokes_exactly_two_artifacts(tmp_path, monkeypatch) -> None:
    dist = tmp_path / "dist"
    commands: list[tuple[str, ...]] = []
    smoked: list[Path] = []

    def fake_run(*command: str, **_kwargs: object) -> None:
        commands.append(command)
        if command[-3:] == ("-m", "build", "--no-isolation"):
            dist.mkdir()
            (dist / "package-0.2.0-py3-none-any.whl").touch()
            (dist / "package-0.2.0.tar.gz").touch()

    monkeypatch.setattr(quality, "DIST", dist)
    monkeypatch.setattr(quality, "run", fake_run)
    monkeypatch.setattr(quality, "smoke_install", smoked.append)

    assert quality.main() == 0

    flattened = [item for command in commands for item in command]
    assert ["ruff", "pyright", "pytest", "build", "twine"] == [
        name for name in ("ruff", "pyright", "pytest", "build", "twine") if name in flattened
    ]
    assert "--cov-branch" in flattened
    assert "--strict" in flattened
    assert any(command[-3:] == ("-m", "build", "--no-isolation") for command in commands)
    assert [path.suffix for path in smoked] == [".whl", ".gz"]
