import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def helper(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("corrected_label_preparation_test", scripts / "prepare_corrected_label_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Small synthetic fixture; production constants/CLI expose no override.
    names = list(module.EXPECTED_SOURCES["train"])
    monkeypatch.setattr(module, "EXPECTED_COUNTS", {"train": 10, "dev": 5})
    monkeypatch.setattr(module, "EXPECTED_SOURCES", {"train": dict.fromkeys(names, 2), "dev": dict.fromkeys(names, 1)})
    return module


@pytest.fixture
def revision(helper, tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    rows = {"train": [], "dev": []}
    for split in rows:
        for source in helper.EXPECTED_SOURCES[split]:
            for i in range(helper.EXPECTED_SOURCES[split][source]):
                number = sum(map(len, rows.values())) + 1
                image = root / f"image_{number}.png"
                target = root / f"target_{number}.png"
                Image.new("RGB", (32, 24), (number * 9, 50, 110)).save(image)
                mask = np.ones((24, 32), np.uint8)
                mask[:, :16] = 0
                mask[0, :] = 255
                Image.fromarray(mask).save(target)
                import hashlib
                rgb = np.asarray(Image.open(image).convert("RGB"))
                digest = hashlib.sha256(b"RGB:32:24:")
                digest.update(rgb.tobytes())
                rows[split].append({
                    "image": str(image), "target": str(target), "source_image": str(image),
                    "target_original": str(target), "dataset_id": source, "sample_id": str(number),
                    "group_id": f"scene_{number}", "origin_id": f"origin_{number}",
                    "development_group_id": f"devgroup_{number}",
                    "task": "segmentation", "label_space_id": "road_surface_v1",
                    "split": split, "export_split": split,
                    "image_sha256": helper.identity(image)["sha256"],
                    "target_sha256": helper.identity(target)["sha256"],
                    "image_rgb_sha256": digest.hexdigest(),
                    "preprocessing_provenance": {"previous_export_split": "train",
                        "development_role": helper.EXPOSURE if split == "dev" else "training"},
                })

    def write(path, obj):
        path.write_text(json.dumps(obj), encoding="utf-8")

    def manifest(path, items):
        path.write_text("".join(json.dumps(row) + "\n" for row in items), encoding="utf-8")

    protected_image, protected_mask = root / "protected.png", root / "protected_mask.png"
    Image.new("RGB", (32, 24), (250, 250, 250)).save(protected_image)
    Image.new("L", (32, 24), 1).save(protected_mask)
    protected = {"image": str(protected_image), "target": str(protected_mask),
                 "dataset_id": "official", "group_id": "official_group",
                 "image_sha256": helper.identity(protected_image)["sha256"],
                 "image_rgb_sha256": "f" * 64}
    manifest(root / "official.jsonl", [protected])
    write(root / "official_contract.json", {"complete": True, "sources": [{"records": [
        {"row": protected, "image": helper.identity(protected_image), "image_rgb_sha256": "f" * 64}]}], "public_tracks": {}})
    write(root / "old_export.json", {"complete": True})
    write(root / "old_marker.json", {"report_sha256": helper.identity(root / "old_export.json")["sha256"]})
    for name in ("policy", "mapping", "changes"):
        write(root / f"{name}.json", {"fixture": name})
    write(root / "split_report.json", {"complete": True, "counts": {"input": 15, "train": 10, "dev": 5}})
    manifest(root / "old_train.jsonl", [row for split in rows.values() for row in split])
    report = {
        "complete": True, "format": helper.FORMAT, "pool_count": 15,
        "inputs": {key: helper.identity(root / filename) for key, filename in (
            ("export_report", "old_export.json"), ("export_marker", "old_marker.json"),
            ("train_manifest", "old_train.jsonl"), ("reporting_manifest", "official.jsonl"),
            ("official_contract", "official_contract.json"))},
        "validation": {
            "all_referenced_training_images_byte_verified": True,
            "all_referenced_training_masks_byte_verified_and_decoded": True,
            "only_ignore255_to_nonroad1_changed": True,
            "official_reporting_manifests_contract_reports_unchanged": True,
            "reporting_cross_split_byte_rgb_group_overlap": 0,
        },
    }
    for name in ("policy", "mapping", "changes", "split_report"):
        report[f"{name}_sha256"] = helper.identity(root / f"{name}.json")["sha256"]

    def seal():
        report["manifests"] = {}
        for split, items in rows.items():
            manifest(root / f"{split}.jsonl", items)
            report["manifests"][split] = {**helper.identity(root / f"{split}.jsonl"), "samples": len(items)}
        write(root / "report.json", report)
        digest = helper.identity(root / "report.json")["sha256"]
        write(root / ".complete", {"report_sha256": digest})
        return digest

    digest = seal()
    return root, rows, report, seal, digest, protected


def test_reviewed_partition_and_hashes_accepted(helper, revision):
    root, _, _, _, digest, _ = revision
    checked = helper.validate_revision(root / "report.json", digest)
    assert {k: len(v) for k, v in checked["partitions"].items()} == {"train": 10, "dev": 5}


@pytest.mark.parametrize("artifact", ["report.json", "train.jsonl", "dev.jsonl", "mapping.json", "split_report.json", "official_contract.json", "old_train.jsonl"])
def test_changed_provenance_rejected(helper, revision, artifact):
    root, _, _, _, digest, _ = revision
    path = root / artifact
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        helper.validate_revision(root / "report.json", digest)


def test_count_change_rejected_even_with_resealed_manifest(helper, revision):
    root, rows, _, seal, _, _ = revision
    rows["dev"].pop()
    with pytest.raises(ValueError, match="count mismatch"):
        helper.validate_revision(root / "report.json", seal())


@pytest.mark.parametrize("field", ["image_sha256", "image_rgb_sha256", "origin_id", "development_group_id"])
def test_train_dev_leakage_rejected(helper, revision, field):
    root, rows, _, seal, _, _ = revision
    rows["dev"][0][field] = rows["train"][0][field]
    with pytest.raises(ValueError, match="overlap"):
        helper.validate_revision(root / "report.json", seal())


def test_new_training_image_outside_original_pool_rejected(helper, revision):
    root, rows, _, seal, _, _ = revision
    rows["train"][0]["sample_id"] = "unreviewed addition"
    with pytest.raises(ValueError, match="exactly preserve"):
        helper.validate_revision(root / "report.json", seal())


@pytest.mark.parametrize("kind", ["official_bytes", "public_dataset", "official_role"])
def test_official_and_public_selection_forbidden(helper, revision, kind):
    root, rows, _, seal, _, protected = revision
    if kind == "official_bytes":
        rows["dev"][0]["image_sha256"] = protected["image_sha256"]
    elif kind == "public_dataset":
        rows["dev"][0]["dataset_id"] = "road_obstacle_21_public"
    else:
        rows["dev"][0]["preprocessing_provenance"]["previous_export_split"] = "val"
    with pytest.raises(ValueError, match="Official/public|public/official|training-derived"):
        helper.validate_revision(root / "report.json", seal())


def test_completion_marker_required(helper, revision):
    root, _, _, _, digest, _ = revision
    (root / ".complete").write_text('{"report_sha256":"' + "0" * 64 + '"}')
    with pytest.raises(ValueError, match="marker mismatch"):
        helper.validate_revision(root / "report.json", digest)


def _mock_initializer(helper, monkeypatch, root):
    path = root / "initial_training_report.json"
    path.write_text('{"training_complete":true}', encoding="utf-8")
    checkpoint = root / "best.pt"
    checkpoint.write_bytes(b"synthetic test; never load")
    checkpoint_id = helper.identity(checkpoint)
    monkeypatch.setattr(helper, "_verify_initializer", lambda report: (
        {"training_complete": True, "dataset": {"provenance_sha256": helper.identity(root / "old_export.json")["sha256"]}}, checkpoint_id,
        {"kind": "completed_best_checkpoint", "checkpoint": checkpoint_id}))
    return path


def test_plan_precommits_sequence_metrics_exposure_and_never_materializes(helper, revision, tmp_path, monkeypatch):
    root, _, _, _, digest, _ = revision
    initial = _mock_initializer(helper, monkeypatch, root)
    monkeypatch.setattr(helper, "_materialize", lambda *args: pytest.fail("plan-only must not export"))
    output = tmp_path / "new_plan"
    result = helper.prepare(root / "report.json", digest, initial, output)
    plan = json.loads((output / "plan/hypothesis.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert config["data"]["train_manifest"] == str(root / "train.jsonl")
    assert config["data"]["val_manifest"] == str(root / "dev.jsonl")
    assert plan["max_new_training_runs"] == 1 and plan["epochs"] == 3
    assert plan["evaluation"]["role"] == helper.EXPOSURE
    assert plan["acceptance"]["maximum_nonroad_as_road_fpr_increase"] == 0
    assert plan["acceptance"]["maximum_road_recall_decrease"] == .01
    assert plan["acceptance"]["maximum_road_iou_decrease"] == .01
    assert "--init-training-report" in plan["commands"]["2_single_candidate_training"]
    assert plan["commands"]["1_baseline_development"][2:4] == ["--prepared", str(root)]
    assert "baseline" in plan["commands"]["1_baseline_development"]
    assert "candidate" in plan["commands"]["3_candidate_development"]
    assert result["mode"] == "plan_only" and not result["ready_for_training_after_baseline"]
    assert result["gpu_used"] is False and result["training_started"] is False
    assert not (output / "export").exists()
    assert not (output / ".incomplete").exists()


def test_existing_output_not_overwritten(helper, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    note = output / "keep.txt"
    note.write_text("keep")
    with pytest.raises(FileExistsError):
        helper.prepare("missing", "0" * 64, "missing", output)
    assert list(output.iterdir()) == [note]


def test_wrong_parent_export_rejected_before_output_creation(helper, revision, tmp_path, monkeypatch):
    root, _, _, _, digest, _ = revision
    initial = _mock_initializer(helper, monkeypatch, root)
    monkeypatch.setattr(helper, "_verify_initializer", lambda path: (
        {"dataset": {"provenance_sha256": "0" * 64}}, {}, {}))
    output = tmp_path / "wrong_parent"
    with pytest.raises(ValueError, match="completed AIHub parent"):
        helper.prepare(root / "report.json", digest, initial, output)
    assert not output.exists()


def test_cpu_export_and_contract_preserve_every_row(helper, revision, tmp_path, monkeypatch):
    root, _, _, _, digest, _ = revision
    initial = _mock_initializer(helper, monkeypatch, root)
    # CPU exporter imports torch-backed dataset code, but must never touch CUDA.
    import torch
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CPU prep initialized CUDA"))
    output = tmp_path / "materialized"
    result = helper.prepare(root / "report.json", digest, initial, output, prepare_data=True, workers=1)
    export = json.loads((output / "export/report.json").read_text())
    contract = json.loads((output / "dev_contract/contract.json").read_text())
    assert sum(export["export_counts"]["train"].values()) == 10
    assert sum(export["export_counts"]["val"].values()) == 5
    assert sum(source["images"] for source in contract["sources"]) == 5
    assert contract["public_tracks"] == {}
    assert contract["runtime_threshold"] == .65
    assert result["ready_for_training_after_baseline"] and result["baseline_evaluation_still_required"]


def test_changed_source_image_rejected_by_cpu_export_and_failure_preserved(helper, revision, tmp_path, monkeypatch):
    root, rows, _, _, digest, _ = revision
    initial = _mock_initializer(helper, monkeypatch, root)
    Image.new("RGB", (32, 24), (1, 1, 1)).save(rows["train"][0]["image"])
    output = tmp_path / "failed_export"
    with pytest.raises(ValueError, match="SHA256"):
        helper.prepare(root / "report.json", digest, initial, output, prepare_data=True, workers=1)
    assert (output / ".incomplete").exists()
    assert (output / "plan/hypothesis.json").exists()
    assert (output / "failure.json").exists()
    assert not (output / ".complete").exists()
