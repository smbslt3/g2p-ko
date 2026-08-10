from __future__ import annotations

import importlib.resources
from pathlib import Path
import subprocess
import sys
from inspect import signature
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_project_metadata_and_dependencies_are_minimal() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    kiwi_commit = "4f5ee9beacfdc84d537f57d1648a1f6898c71935"

    assert metadata["project"]["name"] == "g2p-ko"
    assert metadata["project"]["version"] == "0.1.0"
    assert metadata["project"]["dependencies"] == ["kiwipiepy>=0.23.2"]
    assert "optional-dependencies" not in metadata["project"]
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE", "NOTICE"]
    assert metadata["project"]["urls"]["Repository"] == "https://github.com/smbslt3/g2p-ko"
    assert metadata["build-system"]["requires"] == ["hatchling==1.31.0"]
    assert "Programming Language :: Python :: 3.14" in metadata["project"]["classifiers"]
    assert metadata["tool"]["uv"]["sources"]["kiwipiepy"] == {
        "git": "https://github.com/bab2min/kiwipiepy.git",
        "rev": kiwi_commit,
    }
    assert "kiwipiepy>=0.23.2" not in metadata["dependency-groups"]["dev"]
    assert metadata["tool"]["uv"]["extra-build-dependencies"]["kiwipiepy"] == [
        "cmake>=3.12"
    ]
    assert "force-include" not in metadata["tool"]["hatch"]["build"]["targets"]["wheel"]
    sdist_include = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert {"/src", "/tests", "/tools", "/docs"} <= set(sdist_include)
    assert {"/README.md", "/README.en.md", "/NOTICE"} <= set(sdist_include)
    assert not any(item.startswith("/benchmarks/") for item in sdist_include)


def test_ci_covers_representative_runtimes_and_reproducible_builds() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    for runtime in (
        '- os: ubuntu-latest\n            python-version: "3.10"',
        '- os: ubuntu-latest\n            python-version: "3.13"',
        '- os: ubuntu-latest\n            python-version: "3.14"',
        '- os: windows-latest\n            python-version: "3.13"',
    ):
        assert runtime in workflow
    assert "cancel-in-progress: true" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "SOURCE_DATE_EPOCH" in workflow
    assert "재현 빌드 SHA-256 비교" in workflow


def test_kiwi_lock_is_pinned_to_the_requested_commit() -> None:
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    kiwi = next(item for item in lock["package"] if item["name"] == "kiwipiepy")

    assert kiwi["source"]["git"].endswith(
        "#4f5ee9beacfdc84d537f57d1648a1f6898c71935"
    )


def test_normalizer_and_normalized_g2p_import_without_kiwi() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(source_root)!r})

class BlockKiwi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "kiwipiepy" or fullname.startswith("kiwipiepy."):
            raise AssertionError("normalizer import가 Kiwi를 요청했습니다.")
        return None

sys.meta_path.insert(0, BlockKiwi())

from g2p_ko import G2P, KoreanTTSNormalizer

assert KoreanTTSNormalizer()("3개") == "세 개"
assert callable(G2P())
assert "kiwipiepy" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_foundation_data_is_package_resource() -> None:
    package = importlib.resources.files("g2p_ko")
    assert package.joinpath("py.typed").is_file()
    assert package.joinpath("data/number_reading_model.json").is_file()


def test_public_constructors_do_not_expose_backend_selection() -> None:
    from g2p_ko import G2P, KoreanTTSNormalizer

    assert set(signature(G2P).parameters) == {"lexicon", "max_length"}
    assert not {"kiwi", "analyzer"} & set(signature(KoreanTTSNormalizer).parameters)
