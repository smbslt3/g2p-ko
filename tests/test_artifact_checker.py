from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from tools.check_artifacts import check_artifact


_DATA = Path(__file__).resolve().parents[1] / "src" / "g2p_ko" / "data"


def _valid_json_assets() -> dict[str, str]:
    return {
        f"g2p_ko/data/{name}": (_DATA / name).read_text(encoding="utf-8")
        for name in ("number_reading_model.json",)
    }


def _write_wheel(
    path: Path,
    *,
    omit_file: str | None = None,
    requirements: tuple[str, ...] = ("kiwipiepy>=0.23.2",),
    provides_extra: bool = False,
    json_overrides: dict[str, str] | None = None,
) -> None:
    metadata = "Metadata-Version: 2.4\nName: g2p-ko\nVersion: 0.1.0\n"
    if provides_extra:
        metadata += "Provides-Extra: g2p\n"
    for requirement in requirements:
        metadata += f"Requires-Dist: {requirement}\n"
    files = {
        "g2p_ko/__init__.py": "",
        "g2p_ko/py.typed": "",
        **_valid_json_assets(),
        "g2p_ko-0.1.0.dist-info/METADATA": metadata,
        "g2p_ko-0.1.0.dist-info/licenses/LICENSE": "MIT",
        "g2p_ko-0.1.0.dist-info/licenses/NOTICE": "AI-Hub data notice",
    }
    files.update(json_overrides or {})
    if omit_file is not None:
        del files[omit_file]
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_artifact_checker_accepts_required_runtime_dependency(tmp_path: Path) -> None:
    artifact = tmp_path / "g2p_ko-0.1.0-py3-none-any.whl"
    _write_wheel(artifact)

    report = check_artifact(artifact)

    assert report["ok"] is True
    assert report["checks"]["package_data"]["ok"] is True
    assert report["checks"]["license"]["ok"] is True
    assert report["checks"]["no_legacy"]["ok"] is True
    assert report["checks"]["json_assets"]["ok"] is True
    assert report["checks"]["runtime_dependencies"]["ok"] is True


def test_artifact_checker_rejects_invalid_json_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "g2p_ko-0.1.0-py3-none-any.whl"
    member = "g2p_ko/data/number_reading_model.json"
    _write_wheel(artifact, json_overrides={member: "{}"})

    report = check_artifact(artifact)

    assert report["ok"] is False
    assert report["checks"]["json_assets"]["ok"] is False


def test_artifact_checker_rejects_malformed_packaged_json(tmp_path: Path) -> None:
    artifact = tmp_path / "g2p_ko-0.1.0-py3-none-any.whl"
    _write_wheel(
        artifact,
        json_overrides={"g2p_ko/data/number_reading_model.json": "{"},
    )

    report = check_artifact(artifact)

    assert report["ok"] is False
    assert report["checks"]["json_assets"]["ok"] is False


def test_artifact_checker_rejects_unpublished_extra_declaration(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "g2p_ko-0.1.0-py3-none-any.whl"
    _write_wheel(artifact, provides_extra=True)

    report = check_artifact(artifact)

    assert report["ok"] is False
    assert report["checks"]["runtime_dependencies"]["ok"] is False


@pytest.mark.parametrize(
    "missing_file, check",
    [
        ("g2p_ko/data/number_reading_model.json", "package_data"),
        ("g2p_ko/py.typed", "package_data"),
        ("g2p_ko-0.1.0.dist-info/METADATA", "runtime_dependencies"),
        ("g2p_ko-0.1.0.dist-info/licenses/LICENSE", "license"),
        ("g2p_ko-0.1.0.dist-info/licenses/NOTICE", "license"),
    ],
)
def test_artifact_checker_rejects_missing_required_file(
    tmp_path: Path,
    missing_file: str,
    check: str,
) -> None:
    artifact = tmp_path / "g2p_ko-0.1.0-py3-none-any.whl"
    _write_wheel(artifact, omit_file=missing_file)

    report = check_artifact(artifact)

    assert report["ok"] is False
    assert report["checks"][check]["ok"] is False


@pytest.mark.parametrize(
    "requirements",
    [
        (),
        ("kiwipiepy",),
        ("kiwipiepy>=0.23.3",),
        ("kiwipiepy @ git+https://github.com/bab2min/kiwipiepy.git@main",),
        ("requests>=2", "kiwipiepy>=0.23.2"),
    ],
)
def test_artifact_checker_rejects_incorrect_runtime_dependency(
    tmp_path: Path,
    requirements: tuple[str, ...],
) -> None:
    artifact = tmp_path / "g2p_ko-0.1.0-py3-none-any.whl"
    _write_wheel(artifact, requirements=requirements)

    report = check_artifact(artifact)

    assert report["ok"] is False
    assert report["checks"]["runtime_dependencies"]["ok"] is False
