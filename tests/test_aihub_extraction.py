import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import extract_aihub_verified as intake


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path, extra=None):
    root = tmp_path / "sources"
    rows = []
    specs = [(f"{number}.data/Training/TS.zip", "images/clip/a.jpg") for number in (13, 14, 15)]
    specs += [(f"{number}.data/Validation/VS.zip", "images/other/b.jpg") for number in (13, 14, 15)]
    if extra:
        specs.append(extra)
    for rel, member in specs:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member, b"fixture bytes")
        snapshot = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        rows.append({"relative_path": rel, "sha256": intake.sha256(path), "stat_after": snapshot,
                     "declared_expanded_bytes": 13, "member_count": 1, "verified": True,
                     "crc_all_members_passed": True, "stable_during_verification": True,
                     "unchanged_since_enrollment": True, "stable_at_end_of_all_archives": True})
    report = {"status": "complete", "dataset_root": str(root), "all_observed_archives_verified": True,
              "observed_inventory_unchanged": True, "fixed_enrollment": [{"relative_path": r["relative_path"]} for r in rows], "archives": rows}
    verification = tmp_path / "verification.json"
    write(verification, report)
    completion = tmp_path / "completion.json"
    write(completion, {"download_complete": True, "expected_inventory_confirmed": True,
                       "verification_sha256": intake.sha256(verification), "evidence_kind": "user_report",
                       "evidence": "User confirmed the selected archive list and completed download."})
    return verification, completion, report


def refresh(verification, completion, report):
    write(verification, report)
    evidence = json.loads(completion.read_text())
    evidence["verification_sha256"] = intake.sha256(verification)
    write(completion, evidence)


def test_extracts_fresh_roots_and_publishes_three_road_source_pairs(tmp_path):
    verification, completion, report = fixture(tmp_path, ("12.data/Training/TS_01.zip", "/image.jpg"))
    original = {r["relative_path"]: r["sha256"] for r in report["archives"]}
    destination = tmp_path / "extracted"
    result = intake.extract(verification, completion, destination, reserve_bytes=0)
    assert result["complete"] and len(result["archives"]) == 7
    assert not (destination / "INCOMPLETE").exists()
    assert (destination / "12.data/Training/TS_01/image.jpg").read_bytes() == b"fixture bytes"
    spec = json.loads((destination / "road_source_spec.json").read_text())
    assert {r["dataset_number"] for r in spec["sources"]} == {13, 14, 15}
    assert all(Path(r["train_extraction_report"]).exists() and Path(r["val_extraction_report"]).exists() for r in spec["sources"])
    assert all(intake.sha256(Path(report["dataset_root"]) / rel) == sha for rel, sha in original.items())
    with pytest.raises(FileExistsError):
        intake.extract(verification, completion, destination, reserve_bytes=0)


@pytest.mark.parametrize("field,value", [("download_complete", False), ("expected_inventory_confirmed", False), ("evidence_kind", "stable_file_sizes"), ("evidence", "")])
def test_file_stability_is_not_download_completion(tmp_path, field, value):
    verification, completion, _ = fixture(tmp_path)
    evidence = json.loads(completion.read_text())
    evidence[field] = value
    write(completion, evidence)
    with pytest.raises(ValueError, match="completion evidence"):
        intake.extract(verification, completion, tmp_path / "out", reserve_bytes=0)
    assert not (tmp_path / "out").exists()


def test_changed_archive_rejected_before_output_creation(tmp_path):
    verification, completion, report = fixture(tmp_path)
    path = Path(report["dataset_root"]) / report["archives"][0]["relative_path"]
    with path.open("ab") as stream:
        stream.write(b"download continued")
    with pytest.raises(ValueError, match="changed after verification"):
        intake.extract(verification, completion, tmp_path / "out", reserve_bytes=0)
    assert not (tmp_path / "out").exists()


def test_user_may_select_verified_current_snapshot_without_claiming_download_completion(tmp_path):
    verification, completion, _ = fixture(tmp_path)
    evidence = json.loads(completion.read_text())
    evidence.update(download_complete=False, evidence_kind="user_selected_snapshot",
                    use_current_verified_inventory=True, scope="verified_archives_only",
                    evidence="User: currently available data is the selected experiment scope.")
    write(completion, evidence)
    result = intake.extract(verification, completion, tmp_path / "out", reserve_bytes=0)
    assert result["complete"]
    assert result["completion_evidence"]["download_complete"] is False


@pytest.mark.parametrize("field,value", [("use_current_verified_inventory", False),
                                       ("scope", "all_downloads"),
                                       ("download_complete", True),
                                       ("verification_sha256", "0" * 64)])
def test_snapshot_requires_explicit_scope_and_exact_verification(tmp_path, field, value):
    verification, completion, _ = fixture(tmp_path)
    evidence = json.loads(completion.read_text())
    evidence.update(download_complete=False, evidence_kind="user_selected_snapshot",
                    use_current_verified_inventory=True, scope="verified_archives_only")
    evidence[field] = value
    write(completion, evidence)
    with pytest.raises(ValueError, match="completion evidence"):
        intake.extract(verification, completion, tmp_path / "out", reserve_bytes=0)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["../outside.jpg", "/other/file.jpg", "//server/file", "C:/file", "normal/../file", "INCOMPLETE", "CON.jpg", "dir\\file"])
def test_unsafe_member_paths_never_escape_even_legacy_mode(name):
    with pytest.raises(ValueError):
        intake.member_path(zipfile.ZipInfo(name), allow_root_basename=True)


def test_root_basename_exception_is_opt_in_and_rejects_symlinks():
    with pytest.raises(ValueError):
        intake.member_path(zipfile.ZipInfo("/image.jpg"))
    entry = zipfile.ZipInfo("/link")
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symlink"):
        intake.member_path(entry, allow_root_basename=True)


def test_windows_manifest_separators_are_portable_without_accepting_traversal():
    assert str(intake.relative_path(r"13.data\Training\TS.zip")) == "13.data/Training/TS.zip"
    for path in (r"..\outside.zip", r"\\server\share\file.zip", r"C:\file.zip"):
        with pytest.raises(ValueError):
            intake.relative_path(path)


def test_budget_is_checked_before_creating_output(tmp_path):
    verification, completion, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        intake.extract(verification, completion, tmp_path / "out", max_expanded_bytes=1, reserve_bytes=0)
    assert not (tmp_path / "out").exists()


def test_failed_crc_leaves_incomplete_and_no_success_marker(tmp_path):
    verification, completion, report = fixture(tmp_path)
    row = report["archives"][0]
    archive = Path(report["dataset_root"]) / row["relative_path"]
    blob = archive.read_bytes().replace(b"fixture bytes", b"corrupt bytes")
    archive.write_bytes(blob)
    row["sha256"] = intake.sha256(archive)
    row["stat_after"]["mtime_ns"] = archive.stat().st_mtime_ns
    refresh(verification, completion, report)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        intake.extract(verification, completion, tmp_path / "out", reserve_bytes=0)
    assert (tmp_path / "out/INCOMPLETE").exists()
    assert not (tmp_path / "out/EXTRACTION_COMPLETE.json").exists()


def test_enrollment_omission_is_not_a_complete_selection(tmp_path):
    verification, completion, report = fixture(tmp_path)
    report["archives"].pop()
    refresh(verification, completion, report)
    with pytest.raises(ValueError, match="enrolled"):
        intake.extract(verification, completion, tmp_path / "out", reserve_bytes=0)
