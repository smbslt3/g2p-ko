"""wheel과 sdist의 최소 배포 계약을 검사한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import Parser
import json
from pathlib import Path
import tarfile
from typing import Iterable
import zipfile

from g2p_ko.normalizer.number_context import _parse_model


_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DATA = (
    "g2p_ko/py.typed",
    "g2p_ko/data/number_reading_model.json",
)
_JSON_PACKAGE_DATA = tuple(item for item in _PACKAGE_DATA if item.endswith(".json"))
_REQUIRED_LICENSES = ("LICENSE", "NOTICE")
_RUNTIME_DEPENDENCIES = ("kiwipiepy>=0.23.2",)


@dataclass(frozen=True, slots=True)
class ArtifactContents:
    """배포 형식과 정규화한 archive 멤버 이름을 묶는다."""

    kind: str
    members: tuple[str, ...]
    json_assets: tuple[tuple[str, str], ...]
    metadata: str | None = None


def _normalized_name(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _matching_member(names: Iterable[str], expected: str) -> tuple[str, ...]:
    """sdist 최상위 디렉터리를 무시하고 패키지 파일을 찾는다."""

    return tuple(
        name
        for name in names
        if (normalized := _normalized_name(name)) == expected
        or normalized.endswith("/" + expected)
    )


def _contents(path: Path) -> ArtifactContents:
    """압축을 풀지 않고 배포물 멤버 목록을 읽는다."""

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            archive_members = tuple(archive.namelist())
            members = tuple(_normalized_name(item) for item in archive_members)
            json_assets = tuple(
                (expected, archive.read(matches[0]).decode("utf-8"))
                for expected in _JSON_PACKAGE_DATA
                if len(matches := _matching_member(archive_members, expected)) == 1
            )
            metadata_names = tuple(
                item
                for item in archive_members
                if _normalized_name(item).endswith(".dist-info/METADATA")
            )
            metadata = (
                archive.read(metadata_names[0]).decode("utf-8")
                if len(metadata_names) == 1
                else None
            )
            return ArtifactContents(
                "wheel",
                members,
                json_assets,
                metadata,
            )
    if path.name.endswith(".tar.gz") or path.suffix in {".tgz", ".tar"}:
        with tarfile.open(path, "r:*") as archive:
            archive_members = tuple(archive.getmembers())
            names = tuple(item.name for item in archive_members)
            json_assets: list[tuple[str, str]] = []
            for expected in _JSON_PACKAGE_DATA:
                matches = _matching_member(names, expected)
                if len(matches) != 1:
                    continue
                member = archive.getmember(matches[0])
                extracted = archive.extractfile(member)
                if extracted is not None:
                    json_assets.append((expected, extracted.read().decode("utf-8")))
            return ArtifactContents(
                "sdist",
                tuple(_normalized_name(item.name) for item in archive_members),
                tuple(json_assets),
            )
    raise ValueError(f"지원하지 않는 artifact 형식입니다: {path}")


def _strip_sdist_root(members: Iterable[str]) -> tuple[str, ...]:
    """sdist의 공통 최상위 디렉터리를 제거한다."""

    materialized = tuple(member for member in members if member)
    roots = {member.split("/", 1)[0] for member in materialized if "/" in member}
    if len(roots) != 1:
        return materialized
    root = next(iter(roots)) + "/"
    return tuple(
        member[len(root) :] if member.startswith(root) else member
        for member in materialized
    )


def _has_path(members: Iterable[str], expected: str) -> bool:
    return any(
        member == expected or member.endswith("/" + expected)
        for member in members
    )


def _required_files_check(
    members: tuple[str, ...],
    required: tuple[str, ...],
) -> dict[str, object]:
    missing = [item for item in required if not _has_path(members, item)]
    return {
        "ok": not missing,
        "detail": "" if not missing else f"누락: {', '.join(missing)}",
    }


def _contains_legacy(members: Iterable[str]) -> bool:
    return any(
        "legacy" in {part.casefold() for part in member.split("/")}
        for member in members
    )


def _runtime_dependency_check(metadata: str | None) -> dict[str, object]:
    """wheel이 정확한 최소 런타임 의존성만 선언하는지 검사한다."""

    parsed = Parser().parsestr(metadata) if metadata is not None else None
    requirements = parsed.get_all("Requires-Dist", []) if parsed is not None else []
    extras = parsed.get_all("Provides-Extra", []) if parsed is not None else []
    normalized = tuple(sorted("".join(item.split()).casefold() for item in requirements))
    expected = tuple(sorted(item.casefold() for item in _RUNTIME_DEPENDENCIES))
    normalized_extras = [item.strip().casefold() for item in extras]
    valid = metadata is not None and normalized == expected and not normalized_extras
    return {
        "ok": valid,
        "detail": (
            ""
            if valid
            else "기본 wheel은 Kiwi 런타임 의존성만 정확히 선언해야 합니다."
        ),
    }


def _validate_json_schema(asset: str, payload: object) -> None:
    """패키지 JSON의 런타임에 필요한 최소 계약을 검증한다."""

    if not asset.endswith("number_reading_model.json"):
        raise ValueError(f"알 수 없는 JSON 자산입니다: {asset}")
    _parse_model(payload)


def _json_assets_check(
    assets: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """압축 안 JSON을 직접 파싱하고 배포 스키마를 검증한다."""

    by_name = dict(assets)
    errors: list[str] = []
    for expected in _JSON_PACKAGE_DATA:
        raw = by_name.get(expected)
        if raw is None:
            errors.append(f"{expected}: 읽을 수 없음")
            continue
        try:
            payload = json.loads(raw)
            _validate_json_schema(expected, payload)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{expected}: {error}")
    return {
        "ok": not errors,
        "detail": "; ".join(errors),
    }


def check_artifact(path: str | Path) -> dict[str, object]:
    """artifact 하나의 패키지 데이터·고지·legacy 제외 계약을 검사한다."""

    artifact = Path(path)
    report: dict[str, object] = {"path": str(artifact), "checks": {}}
    try:
        contents = _contents(artifact)
    except (
        OSError,
        UnicodeError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        report["checks"] = {"readable": {"ok": False, "detail": str(error)}}
        report["ok"] = False
        return report

    members = (
        _strip_sdist_root(contents.members)
        if contents.kind == "sdist"
        else contents.members
    )
    checks = {
        "package_data": _required_files_check(members, _PACKAGE_DATA),
        "json_assets": _json_assets_check(contents.json_assets),
        "license": _required_files_check(members, _REQUIRED_LICENSES),
        "no_legacy": {
            "ok": not _contains_legacy(members),
            "detail": "legacy 경로가 배포물에 포함되면 안 됩니다.",
        },
    }
    if contents.kind == "wheel":
        checks["runtime_dependencies"] = _runtime_dependency_check(contents.metadata)
    report["kind"] = contents.kind
    report["checks"] = checks
    report["ok"] = all(bool(item["ok"]) for item in checks.values())
    return report


def _latest_artifacts(dist: Path) -> list[Path]:
    """wheel과 sdist가 각각 있으면 최신 하나씩 반환한다."""

    groups = {
        "wheel": sorted(
            dist.glob("*.whl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ),
        "sdist": sorted(
            [*dist.glob("*.tar.gz"), *dist.glob("*.tgz"), *dist.glob("*.tar")],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ),
    }
    return [items[0] for items in groups.values() if items]


def main(argv: list[str] | None = None) -> int:
    """지정 artifact 또는 dist의 최신 wheel/sdist를 검사한다."""

    parser = argparse.ArgumentParser(
        description="g2p-ko wheel/sdist 릴리스 게이트 검사"
    )
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="검사할 wheel 또는 sdist 경로",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=_ROOT / "dist",
        help="artifact 미지정 시 검색할 dist 디렉터리",
    )
    args = parser.parse_args(argv)
    artifacts = args.artifacts or _latest_artifacts(args.dist)
    if not artifacts:
        print(
            json.dumps(
                {"ok": False, "error": "검사할 wheel 또는 sdist가 없습니다."},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    reports = [check_artifact(item) for item in artifacts]
    result = {"ok": all(bool(item["ok"]) for item in reports), "artifacts": reports}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
