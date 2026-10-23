import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_road_safety", PROJECT / "scripts/evaluate_road_safety.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_road_precision_recall_and_false_road_have_different_denominators():
    metrics = evaluation.road_metrics([[8, 2], [3, 7]])
    assert metrics["road_precision"] == pytest.approx(8 / 11)
    assert metrics["road_recall"] == pytest.approx(0.8)
    assert metrics["false_road_rate"] == pytest.approx(0.3)
    assert metrics["road_tp_pixels"] == 8 and metrics["road_fp_pixels"] == 3
    undefined = evaluation.road_metrics([[0, 0], [0, 0]])
    assert undefined["road_precision"] is None and undefined["false_road_rate"] is None


@pytest.mark.parametrize("matrix", [[[1.0, 2], [3, 4]], [[1, -1], [3, 4]], [[1, 2, 3]], [[True, False], [False, True]]])
def test_invalid_confusion_is_rejected(matrix):
    with pytest.raises(ValueError, match="confusion"):
        evaluation.road_metrics(matrix)


def test_usual_and_ignore_pixels_do_not_inflate_obstacle_or_road_denominators():
    gt = np.array([[1, 1, 0], [1, 255, 0]], np.uint8)
    probabilities = np.array([[0.5, 0.65, 1.0], [0.64, 1.0, 0.0]], np.float64)
    result = evaluation.obstacle_pixel_metrics(probabilities, gt)
    assert result["labelled_obstacle_pixels"] == 3
    assert result["usual_pixels_not_used_as_road_gt"] == 2
    assert result["ignored_pixels"] == 1
    assert result["decisions"]["road_p_gt_0_5"]["obstacle_as_road_rate"] == pytest.approx(2 / 3)
    assert result["decisions"]["runtime_road"]["obstacle_as_road_rate"] == pytest.approx(1 / 3)
    # GT=0 has no road precision/recall meaning on the anomaly track.
    assert "road_precision" not in result and "road_recall" not in result


def test_micro_aggregation_is_pixel_weighted_not_mean_image_rate():
    small = evaluation.obstacle_pixel_metrics(np.ones((1, 1)), np.ones((1, 1), np.uint8))
    large = evaluation.obstacle_pixel_metrics(np.zeros((3, 3)), np.ones((3, 3), np.uint8))
    empty = evaluation.obstacle_pixel_metrics(np.ones((2, 2)), np.zeros((2, 2), np.uint8))
    metrics = evaluation.aggregate_obstacle_metrics([small, large, empty])["decisions"]["runtime_road"]
    assert metrics["micro_obstacle_as_road_rate"] == pytest.approx(0.1)
    assert metrics["mean_per_image_obstacle_as_road_rate"] == pytest.approx(0.5)
    assert metrics["images_with_labelled_obstacles"] == 2
    with pytest.raises(ValueError, match="no GT=1"):
        evaluation.aggregate_obstacle_metrics([empty])


@pytest.mark.parametrize("probability,target", [
    (np.ones((2, 2)), np.array([[0, 2], [1, 255]], np.uint8)),
    (np.ones((2, 2)), np.zeros((2, 2, 3), np.uint8)),
    (np.ones((1, 1)), np.ones((2, 2), np.uint8)),
    (np.full((2, 2), np.nan), np.ones((2, 2), np.uint8)),
    (np.full((2, 2), 1.01), np.ones((2, 2), np.uint8)),
])
def test_bad_gt_or_probability_is_rejected(probability, target):
    with pytest.raises(ValueError):
        evaluation.obstacle_pixel_metrics(probability, target)


@pytest.mark.parametrize("threshold", [0.5, 0, 1.1, float("nan"), True])
def test_invalid_runtime_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="runtime_threshold"):
        evaluation.obstacle_pixel_metrics(np.ones((1, 1)), np.ones((1, 1), np.uint8), threshold)


def test_upsampling_probabilities_preserves_a_one_pixel_gt_obstacle():
    target = np.zeros((9, 9), np.uint8)
    target[4, 4] = 1
    assert np.array(Image.fromarray(target).resize((2, 2), Image.Resampling.NEAREST)).sum() == 0
    logits = torch.zeros((1, 2, 2, 2))
    logits[:, 0] = np.log(4)  # Road probability 0.8 at model output resolution.
    restored = evaluation.restore_road_probability(logits, target.shape)
    result = evaluation.obstacle_pixel_metrics(restored, target)
    assert restored.shape == target.shape
    assert result["labelled_obstacle_pixels"] == 1
    assert result["decisions"]["runtime_road"]["obstacle_pixels_predicted_road"] == 1


def make_pair(root, track="obstacle", fid="validation_00", *, size=(9, 7), gt=None):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels_masks").mkdir(parents=True, exist_ok=True)
    image = root / "images" / (fid + evaluation.TRACKS[track]["extension"])
    target = root / "labels_masks" / (fid + "_labels_semantic.png")
    Image.new("RGB", size, (15, 30, 45)).save(image)
    if gt is None:
        gt = np.zeros((size[1], size[0]), np.uint8)
        gt[1, 1] = 1
        gt[0, 0] = 255
    Image.fromarray(gt).save(target)
    return image, target


def test_public_gt_pairing_ignores_unlabelled_test_images_and_records_hashes(tmp_path):
    image, target = make_pair(tmp_path)
    Image.new("RGB", (9, 7)).save(tmp_path / "images" / "unlabelled_test.webp")
    discovered = evaluation.discover_pairs(tmp_path, "obstacle", expected_count=1)
    assert discovered["unlabelled_images_not_evaluated"] == 1
    report = evaluation.evaluate_track(discovered, lambda rgb: np.full(rgb.shape[:2], 0.7), "obstacle")
    assert report["ground_truth_resizing"] == "none"
    assert report["aggregate"]["decisions"]["runtime_road"]["micro_obstacle_as_road_rate"] == 1
    row = report["per_image"][0]
    assert row["image"] == evaluation.file_identity(image)
    assert row["ground_truth"] == evaluation.file_identity(target)
    assert row["labelled_obstacle_pixels"] == 1


def test_default_public_counts_prevent_incomplete_download_being_scored(tmp_path):
    make_pair(tmp_path)
    with pytest.raises(ValueError, match="exactly 30"):
        evaluation.discover_pairs(tmp_path, "obstacle")
    anomaly = tmp_path / "anomaly"
    make_pair(anomaly, "anomaly", fid="example")
    with pytest.raises(ValueError, match="exactly 10"):
        evaluation.discover_pairs(anomaly, "anomaly")


def test_missing_pair_wrong_shape_color_gt_and_ambiguous_image_fail(tmp_path):
    image, target = make_pair(tmp_path)
    image.rename(image.with_suffix(".jpeg"))
    with pytest.raises(FileNotFoundError, match="Missing paired"):
        evaluation.discover_pairs(tmp_path, "obstacle", expected_count=1)
    image.with_suffix(".jpeg").rename(image)
    Image.new("RGB", (9, 7)).save(image.with_suffix(".png"))
    with pytest.raises(ValueError, match="Ambiguous"):
        evaluation.discover_pairs(tmp_path, "obstacle", expected_count=1)
    pair = {"frame_id": "test", "image": image, "target": target}
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(target)
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluation.read_pair(pair)
    Image.new("RGB", (9, 7)).save(target)
    with pytest.raises(ValueError, match="RGB label"):
        evaluation.read_pair(pair)


def test_eval_public_images_cannot_overlap_training_source_paths(tmp_path):
    image, _ = make_pair(tmp_path)
    discovered = evaluation.discover_pairs(tmp_path, "obstacle", expected_count=1)
    for field in ("image", "source_image"):
        with pytest.raises(ValueError, match="training manifest"):
            evaluation._assert_not_training(discovered, [{"rows": [{field: str(image)}]}])


def test_native_predictor_uses_training_rgb_resize_then_original_probability_resolution(monkeypatch):
    class FixedModel(torch.nn.Module):
        def forward(self, tensor):
            self.received = tensor.detach().cpu()
            logits = torch.zeros((1, 2, 192, 320), device=tensor.device)
            logits[:, 0] = np.log(4)
            return logits

    model = FixedModel()
    monkeypatch.setattr(evaluation, "build_model", lambda config: model)
    config = {"task": "segmentation", "model": {"num_classes": 2, "width": 4},
              "data": {"label_space_id": "road_surface_v1", "image_size": [192, 320]},
              "training": {"device": "cpu", "amp": False}}
    predictor = evaluation.NativeRoadPredictor(config, {"config": config, "model": {}})
    probability = predictor(np.full((11, 19, 3), 255, np.uint8))
    assert model.received.shape == (1, 3, 192, 320)
    assert torch.all(model.received == 1)
    assert probability.shape == (11, 19) and np.allclose(probability, 0.8)


def configured_checkpoint(tmp_path):
    manifests = {}
    for split in ("train", "val"):
        folder = tmp_path / split
        folder.mkdir()
        Image.new("RGB", (8, 8)).save(folder / "rgb.png")
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(folder / "gt.png")
        manifest = folder / "manifest.jsonl"
        manifest.write_text(json.dumps({"image": "rgb.png", "target": "gt.png", "group_id": split,
                                        "dataset_id": "bdd", "label_space_id": "road_surface_v1"}) + "\n", encoding="utf-8")
        manifests[split] = str(manifest)
    cfg = {"task": "segmentation", "model": {"num_classes": 2, "width": 4},
           "data": {"label_space_id": "road_surface_v1", "image_size": [192, 320],
                    "train_sources": [{"name": "bdd", "manifest": manifests["train"]}],
                    "val_manifests": [{"name": "bdd", "manifest": manifests["val"]}]},
           "training": {"device": "cpu", "amp": False, "epochs": 1, "batch_size": 1, "num_workers": 0,
                        "lr": 0.001, "output_dir": str(tmp_path / "unused_training_output")}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save({"format_version": 1, "config": cfg, "model": evaluation.build_model(cfg).state_dict(),
                "epoch_complete": True, "diagnostic": False}, checkpoint)
    return config_path, checkpoint


def test_full_report_uses_evaluate_sources_and_never_overwrites_or_trains(tmp_path, monkeypatch):
    config, checkpoint = configured_checkpoint(tmp_path)
    calls = []
    def native(cfg, path):
        calls.append((cfg, path))
        return {"bdd": {"confusion_matrix": [[8, 2], [3, 7]], "miou": 0.5, "partial_validation": False}}
    monkeypatch.setattr(evaluation, "evaluate_sources", native)
    output = tmp_path / "report.json"
    report = evaluation.evaluate(config, checkpoint, output)
    assert len(calls) == 1 and calls[0][1] == checkpoint
    assert report["native_sources"]["bdd"]["road_safety_pixels"]["false_road_rate"] == 0.3
    assert report["checkpoint"] == evaluation.file_identity(checkpoint)
    assert report["training_performed"] is False and report["deployment_validated"] is False
    assert report["public_obstacle_anomaly_subsets"] == {}
    assert not (tmp_path / "unused_training_output").exists()
    assert json.loads(output.read_text(encoding="utf-8"))["label_space_id"] == "road_surface_v1"
    with pytest.raises(FileExistsError):
        evaluation.evaluate(config, checkpoint, output)


def test_checkpoint_changed_during_evaluation_is_not_reported_as_one_model(tmp_path, monkeypatch):
    config, checkpoint = configured_checkpoint(tmp_path)
    def mutate_checkpoint(cfg, path):
        with path.open("ab") as stream:
            stream.write(b"changed")
        return {"bdd": {"confusion_matrix": [[1, 0], [0, 1]], "partial_validation": False}}
    monkeypatch.setattr(evaluation, "evaluate_sources", mutate_checkpoint)
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        evaluation.evaluate(config, checkpoint, output)
    assert not output.exists()
