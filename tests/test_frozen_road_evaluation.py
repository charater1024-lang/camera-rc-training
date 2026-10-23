"""CPU-only identity, leakage and paired-denominator tests; no performance claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluation_contract as cohort
import evaluate_frozen_road as frozen
import evaluate_yolo_semantic as yolo
from compare_frozen_road import compare_reports
from test_yolo_semantic_evaluation import complete_export as fixture_export, FakePredictor, labelled_files


@pytest.fixture(autouse=True)
def official_fixture_digest(monkeypatch):
    monkeypatch.setattr(frozen, "OFFICIAL_SHA256", hashlib.sha256(b"fixture official").hexdigest())


def complete_export(tmp_path):
    result = fixture_export(tmp_path)
    weights = tmp_path / "official.pt"
    weights.write_bytes(b"fixture official")
    report = json.loads(result[2].read_text())
    report["pretrained"] = {"path": str(weights), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    write_json(result[2], report)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_contract(tmp_path, *, include_original=False, public=False):
    image, target = labelled_files(tmp_path, "new_holdout", pixel=110)
    manifest = tmp_path / "new_val.jsonl"
    manifest.write_text(json.dumps({"image": str(image), "target": str(target), "dataset_id": "aihub_15",
        "group_id": "night_clip", "origin_id": "night_clip", "label_space_id": "road_surface_v1"}) + "\n")
    sources = [{"name": "aihub_15_night", "manifest": str(manifest)}]
    if include_original:
        sources.insert(0, {"name": "old", "manifest": str(tmp_path / "val.jsonl")})
    spec = {"schema_version": 1, "sources": sources, "runtime_threshold": .65}
    if public:
        for track, count, ext in (("obstacle", 30, "webp"), ("anomaly", 10, "jpg")):
            root = tmp_path / track
            (root / "images").mkdir(parents=True)
            (root / "labels_masks").mkdir()
            for index in range(count):
                values = np.full((32, 32, 3), index + (130 if track == "obstacle" else 170), dtype=np.uint8)
                Image.fromarray(values).save(root / "images" / f"{index}.{ext}", lossless=True, quality=100, subsampling=0)
                labels = np.zeros((32, 32), dtype=np.uint8)
                labels[1, 1] = 1
                Image.fromarray(labels).save(root / "labels_masks" / f"{index}_labels_semantic.png")
            spec[track + "_root"] = str(root)
    spec_path = tmp_path / "eval_spec.json"
    write_json(spec_path, spec)
    result = cohort.prepare(spec_path, tmp_path / "frozen")
    return tmp_path / "frozen" / "contract.json", result


def test_freeze_roundtrip_preserves_original_membership_counts(tmp_path):
    path, expected = make_contract(tmp_path)
    actual, identity, sources, public = cohort.load(path, yolo.IdentityLedger())
    assert actual == expected and identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(sources) == 1 and len(sources[0]["rows"]) == 1 and not public
    assert sum(actual["sources"][0]["records"][0]["evaluated_gt_pixels"].values()) == 192 * 320
    assert actual["training_authorized"] is False


@pytest.mark.parametrize("change", ["image", "target", "manifest", "contract", "marker"])
def test_contract_mutation_rejected_before_inference(tmp_path, change):
    path, report = make_contract(tmp_path)
    record = report["sources"][0]["records"][0]
    file = {"image": Path(record["row"]["image"]), "target": Path(record["row"]["target"]),
            "manifest": Path(report["sources"][0]["manifest"]["path"]), "contract": path,
            "marker": path.parent / ".complete"}[change]
    if change == "marker":
        file.unlink()
    else:
        file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises((ValueError, FileNotFoundError)):
        cohort.load(path, yolo.IdentityLedger())


def test_duplicate_pixels_with_different_encoding_rejected(tmp_path):
    path, _ = make_contract(tmp_path)
    spec = json.loads((tmp_path / "eval_spec.json").read_text())
    row = json.loads((tmp_path / "new_val.jsonl").read_text())
    image = tmp_path / "copy.bmp"
    Image.open(row["image"]).save(image)
    row.update(image=str(image), group_id="second", origin_id="second")
    (tmp_path / "copy.jsonl").write_text(json.dumps(row))
    spec["sources"].append({"name": "copy", "manifest": str(tmp_path / "copy.jsonl")})
    write_json(tmp_path / "eval_spec.json", spec)
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        cohort.prepare(tmp_path / "eval_spec.json", tmp_path / "new_contract")


@pytest.mark.parametrize("retain", [False, True])
def test_within_source_duplicates_need_explicit_multiplicity_policy(tmp_path, retain):
    make_contract(tmp_path)
    spec = json.loads((tmp_path / "eval_spec.json").read_text())
    spec["retain_source_duplicate_multiplicity"] = retain
    row = json.loads((tmp_path / "new_val.jsonl").read_text())
    other = deepcopy(row)
    image = tmp_path / "copy.bmp"
    Image.open(row["image"]).save(image)
    other.update(image=str(image), group_id="second", origin_id="second")
    (tmp_path / "new_val.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    write_json(tmp_path / "eval_spec.json", spec)
    if not retain:
        with pytest.raises(ValueError, match="Duplicate evaluation"):
            cohort.prepare(tmp_path / "eval_spec.json", tmp_path / "new_contract")
    else:
        result = cohort.prepare(tmp_path / "eval_spec.json", tmp_path / "new_contract")
        assert result["sources"][0]["images"] == 2 and result["sources"][0]["duplicate_frames_retained"] == 1
        assert len(result["retained_within_source_duplicates"]) == 1
        loaded, _, _, _ = cohort.load(tmp_path / "new_contract" / "contract.json", yolo.IdentityLedger())
        assert loaded == result
        # The same policy must never authorize duplicates in another source.
        (tmp_path / "other.jsonl").write_text(json.dumps(other))
        spec["sources"].append({"name": "other", "manifest": str(tmp_path / "other.jsonl")})
        write_json(tmp_path / "eval_spec.json", spec)
        with pytest.raises(ValueError, match="Duplicate evaluation"):
            cohort.prepare(tmp_path / "eval_spec.json", tmp_path / "bad_contract")


@pytest.mark.parametrize("overlap", ["path", "scene", "origin", "bytes", "pixels"])
def test_frozen_contract_train_leakage_guards(tmp_path, overlap):
    _, contract = make_contract(tmp_path)
    record = contract["sources"][0]["records"][0]
    row = {"image": str(tmp_path / "unrelated.png"), "group_id": "train", "origin_id": "train", "dataset_id": "train"}
    snapshot = {"training_image_sha256": set(), "training_rgb_sha256": set()}
    if overlap == "path":
        row["image"] = record["row"]["image"]
    elif overlap == "scene":
        row.update(group_id=record["row"]["group_id"], dataset_id=record["row"]["dataset_id"])
    elif overlap == "origin":
        row["origin_id"] = record["row"]["origin_id"]
    elif overlap == "bytes":
        snapshot["training_image_sha256"].add(record["image"]["sha256"])
    else:
        snapshot["training_rgb_sha256"].add(record["image_rgb_sha256"])
    with pytest.raises(ValueError, match="leakage|overlap"):
        cohort.assert_no_leakage(contract, [{"rows": [row]}], snapshot)


def test_old_yolo_uses_new_holdout_without_provenance_rewrite(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    old_config, old_report = config.read_bytes(), training_report.read_bytes()
    contract, _ = make_contract(tmp_path, include_original=True)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", FakePredictor)
    report = frozen.evaluate(contract, "yolo26x-sem", config, best, training_report,
                             tmp_path / "result.json", device="cpu")
    assert report["evaluation_complete"] and report["training_provenance_preserved"]
    assert set(report["native_sources"]) == {"old", "aihub_15_night"}
    assert old_config == config.read_bytes() and old_report == training_report.read_bytes()
    same = deepcopy(report)
    same["config"]["sha256"] = "a" * 64  # Own training configs can legitimately differ.
    same["checkpoint"]["sha256"] = "b" * 64
    compared = compare_reports(report, same)
    assert compared["regressions"] == [] and len(compared["deltas"]) == 24


@pytest.mark.parametrize("field", ["evaluation_contract", "input_size", "runtime_threshold", "precision_condition", "gt_denominators"])
def test_comparison_rejects_changed_cohort_or_protocol(tmp_path, monkeypatch, field):
    config, best, training_report, _ = complete_export(tmp_path)
    contract, _ = make_contract(tmp_path)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", FakePredictor)
    report = frozen.evaluate(contract, "yolo26x-sem", config, best, training_report, tmp_path / "result.json", device="cpu")
    other = deepcopy(report)
    other[field] = "changed"
    with pytest.raises(ValueError, match=field):
        compare_reports(report, other)


def test_public_original_one_pixel_denominator_and_development_role(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    contract, _ = make_contract(tmp_path, public=True)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", FakePredictor)
    result = frozen.evaluate(contract, "yolo26x-sem", config, best, training_report, tmp_path / "result.json", device="cpu")
    assert "not_training_or_threshold_tuning" in result["public_role"]
    for track, count in (("obstacle", 30), ("anomaly", 10)):
        assert result["public_obstacle_anomaly_subsets"][track]["aggregate"]["decisions"]["runtime_road"]["labelled_obstacle_pixels"] == count
    comparison = compare_reports(result, result)
    assert comparison["baseline"]["public"]["obstacle"]["images"] == 30


def test_inference_time_mutation_prevents_publication(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    contract, frozen_contract = make_contract(tmp_path)
    class Mutator(FakePredictor):
        def predict_logits(self, images):
            Path(frozen_contract["sources"][0]["records"][0]["row"]["image"]).write_bytes(b"changed")
            return super().predict_logits(images)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", Mutator)
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="SHA256"):
        frozen.evaluate(contract, "yolo26x-sem", config, best, training_report, output, device="cpu")
    assert not output.exists()


def test_native_checkpoint_own_training_sources_are_not_replaceable(tmp_path):
    from camera_rc.config import load_config
    config, _, _, _ = complete_export(tmp_path)
    stored_config = load_config(config)
    checkpoint = tmp_path / "native.pt"
    torch.save({"format_version": 1, "config": stored_config, "epoch_complete": True,
                "optimizer_steps": 1, "diagnostic": False}, checkpoint)
    summary = tmp_path / "summary.json"
    write_json(summary, {"training_complete": True, "diagnostic": False, "completed_epochs": 1,
                         "requested_epochs": 1, "best_checkpoint": str(checkpoint)})
    _, _, _, proofs, _ = frozen.native_provenance(config, checkpoint, summary, yolo.IdentityLedger(), tmp_path / "export" / "report.json")
    assert proofs[0]["historical_content_binding"] is False
    changed = json.loads(config.read_text())
    changed["data"]["train_manifest"] = "other.jsonl"
    write_json(config, changed)
    with pytest.raises(ValueError, match="training data/model"):
        frozen.native_provenance(config, checkpoint, summary, yolo.IdentityLedger())


def test_tiny_and_yolo_use_same_native_scoring_path(tmp_path, monkeypatch):
    from camera_rc.config import load_config
    config, best, training_report, _ = complete_export(tmp_path)
    contract, _ = make_contract(tmp_path)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", FakePredictor)
    yolo_result = frozen.evaluate(contract, "yolo26x-sem", config, best, training_report, tmp_path / "yolo.json", device="cpu")
    native = tmp_path / "native.pt"
    torch.save({"format_version": 1, "config": load_config(config), "epoch_complete": True,
                "optimizer_steps": 1, "diagnostic": False}, native)
    summary = tmp_path / "summary.json"
    write_json(summary, {"training_complete": True, "diagnostic": False, "completed_epochs": 1,
                         "requested_epochs": 1, "best_checkpoint": str(native)})
    monkeypatch.setattr(frozen, "TinyPredictor", lambda config, stored, **kwargs: FakePredictor(**kwargs))
    tiny_result = frozen.evaluate(contract, "tinyunet", config, native, summary, tmp_path / "tiny.json", device="cpu",
                                  training_content_export=tmp_path / "export" / "report.json")
    assert tiny_result["native_sources"] == yolo_result["native_sources"]
    assert not compare_reports(tiny_result, yolo_result)["regressions"]


def test_yolo_ancestor_training_hashes_are_retained(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    parent = tmp_path / "parent.json"
    parent.write_bytes(training_report.read_bytes())
    current = json.loads(training_report.read_text())
    current["initialization"] = {"kind": "completed_best_checkpoint", "training_report": {"path": str(parent), "sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}}
    current["pretrained"] = {"path": str(best), "sha256": hashlib.sha256(best.read_bytes()).hexdigest()}
    write_json(training_report, current)
    original = yolo.validate_export_snapshot
    def export(completion, training, validation, ledger):
        result = original(completion, training, validation, ledger)
        if completion["identity"]["path"] == str(parent):
            result["training_image_sha256"].add("a" * 64)
            result["training_rgb_sha256"].add("b" * 64)
        return result
    monkeypatch.setattr(yolo, "validate_export_snapshot", export)
    _, _, snapshot, proofs, _ = frozen.yolo_provenance(config, best, training_report, yolo.IdentityLedger())
    assert "a" * 64 in snapshot["training_image_sha256"] and "b" * 64 in snapshot["training_rgb_sha256"]
    assert len(proofs) == 2


def test_native_fallback_checks_current_train_pixels(tmp_path):
    config, _, _, _ = complete_export(tmp_path)
    training, _ = frozen.training_sources(frozen.load_config(config), yolo.IdentityLedger())
    snapshot = frozen.current_training_snapshot(training, yolo.IdentityLedger())
    assert len(snapshot["training_image_sha256"]) == 1 and len(snapshot["training_rgb_sha256"]) == 1
    assert snapshot["historical_content_binding"] is False


def test_missing_local_ancestor_report_is_rejected(tmp_path):
    config, best, training_report, _ = complete_export(tmp_path)
    report = json.loads(training_report.read_text())
    report["initialization"] = {"kind": "completed_best_checkpoint"}
    write_json(training_report, report)
    with pytest.raises(ValueError, match="ancestor training report"):
        frozen.yolo_provenance(config, best, training_report, yolo.IdentityLedger())

