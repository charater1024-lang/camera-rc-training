"""Benchmark orchestration/provenance tests, using no real model or GPU."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_detector_candidates.py"
    spec = importlib.util.spec_from_file_location("detector_candidate_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_ultralytics(tmp_path, monkeypatch, *, supports_end2end=True, broken_fusion=False, expected_model="yolo26s"):
    # Restore process environment after the script changes its local settings.
    for key in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    package = tmp_path / "installed_ultralytics"
    source = package / "cfg/models/26/yolo26.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("nc: 80\nend2end: true\n", encoding="utf-8")
    creations, views = [], []
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            # Match installed 8.4.182: YAML creates both branches but does not
            # select one-to-one inference. Fusion removes the unselected one.
            self.model = [SimpleNamespace(end2end=False, export=False, cv2=object(), cv3=object(),
                                          one2one_cv2=object() if supports_end2end else None,
                                          one2one_cv3=object() if supports_end2end else None)]
        @property
        def end2end(self):
            return self.model[-1].end2end
        @end2end.setter
        def end2end(self, value):
            self.model[-1].end2end = value and self.model[-1].one2one_cv2 is not None
        def forward(self, image):
            views.append((image.data_ptr(), tuple(image.shape)))
            return torch.zeros((1, 3, 6)), {"one2one": {"scores": torch.zeros((1, 80, 1))}}
    class Wrapper:
        def __init__(self, path, *, task, verbose):
            assert path.endswith(expected_model + ".yaml") and task == "detect"
            assert not path.endswith(".pt")
            self.model, self.ckpt, self.fused = Network(), {}, False
            creations.append(self)
        def fuse(self, *, verbose, imgsz):
            self.fused = True
            head = self.model.model[-1]
            self.selected_end2end_before_fusion = head.end2end
            if head.end2end and not broken_fusion:
                head.cv2 = head.cv3 = None
            else:
                head.one2one_cv2 = head.one2one_cv3 = None
                head.end2end = False
            return self
    package_module = SimpleNamespace(__file__=str(package / "__init__.py"), YOLO=Wrapper,
                                     settings={"sync": True, "hub": True})
    monkeypatch.setitem(sys.modules, "ultralytics", package_module)
    # A module stub does not supply installed-distribution metadata. Keep this
    # mocked test independent of the optional Ultralytics installation too.
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name:
                        "8.4.182-test-fixture" if name == "ultralytics" else original_version(name))
    return creations, views


@pytest.mark.parametrize("model", ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
def test_shared_batch_one_views_and_warmup_are_reported_without_accuracy_claim(tmp_path, monkeypatch, model):
    module = script()
    creations, observed = fake_ultralytics(tmp_path, monkeypatch, expected_model=model)
    result = module.benchmark(tmp_path / "report.json", model=model, imgsz=640, views=8,
                              device="cpu", precision="fp32", groups=2, warmup=1, gpu_budget_gib=8)
    assert len(creations) == 1 and creations[0].fused and creations[0].selected_end2end_before_fusion
    assert len(observed) == 24 and all(shape == (1, 3, 640, 640) for _, shape in observed)
    assert len({pointer for pointer, _ in observed[:8]}) == 8
    assert observed[:8] == observed[8:16] == observed[16:]
    assert result["single_view_latency"]["frames"] == 16
    assert result["all_views_group_latency"]["frames"] == 2
    assert result["model_random_initialized"] and not result["pretrained_weights_loaded"]
    assert not result["accuracy_evaluated"] and not result["physical_output"] and not result["evidence_for_80kmh"]
    assert result["graph"]["end2end"] and result["graph"]["output_structure"][0]["shape"] == [1, 3, 6]
    assert not result["graph"]["end2end_default_before_selection"]
    assert result["graph"]["end2end_explicitly_selected_before_fusion"]
    assert result["graph"]["one_to_one_branch_present"] and not result["graph"]["one_to_many_branch_present"]
    assert len(result["architecture_yaml_sha256"]) == 64
    assert result["selected_scale"] == model[-1] and result["model"] == model
    assert result["ultralytics_version"] == "8.4.182-test-fixture"


def test_unsupported_head_is_not_mislabeled_as_e2e_timing(tmp_path, monkeypatch):
    module = script()
    creations, observed = fake_ultralytics(tmp_path, monkeypatch, supports_end2end=False)
    with pytest.raises(ValueError, match="end-to-end"):
        module.benchmark(tmp_path / "report.json", model="yolo26s", imgsz=640, views=1,
                         device="cpu", groups=1, warmup=0, gpu_budget_gib=4)
    assert len(creations) == 1 and not observed
    assert not (tmp_path / "report.json").exists()


def test_fusion_that_discards_selected_e2e_branch_is_rejected(tmp_path, monkeypatch):
    module = script()
    creations, observed = fake_ultralytics(tmp_path, monkeypatch, broken_fusion=True)
    with pytest.raises(ValueError, match="Fusion did not preserve"):
        module.benchmark(tmp_path / "report.json", model="yolo26s", imgsz=640, views=1,
                         device="cpu", groups=1, warmup=0, gpu_budget_gib=4)
    assert creations[0].selected_end2end_before_fusion and not observed
    assert not (tmp_path / "report.json").exists()


def test_cpu_half_rejected_before_output_or_model_creation(tmp_path):
    with pytest.raises(ValueError, match="fp32 only"):
        script().benchmark(tmp_path / "report.json", precision="fp16", device="cpu")
    assert not list(tmp_path.iterdir())
