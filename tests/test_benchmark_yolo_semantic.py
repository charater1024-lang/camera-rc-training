"""CPU-only contract, probability, dataflow and failure-report tests."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_yolo_semantic.py"
    spec = importlib.util.spec_from_file_location("semantic_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_fixture(tmp_path):
    config = tmp_path / "source.json"
    config.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test-checkpoint-provenance-only")
    def ident(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    report = {"model_family": "yolo26x-sem", "task": "semantic", "training_complete": True,
              "diagnostic": False, "source_config": ident(config), "checkpoints": {"best": ident(checkpoint)}}
    path = tmp_path / "training.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, checkpoint, report


def test_training_provenance_requires_completed_artifact_and_exact_bytes(tmp_path):
    module = script()
    path, checkpoint, report = report_fixture(tmp_path)
    assert module.verify_training_report(path, checkpoint)["selected_checkpoint"] == "best"
    checkpoint.write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="SHA256/size"):
        module.verify_training_report(path, checkpoint)
    report["training_complete"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        module.verify_training_report(path, checkpoint)


def test_training_source_config_changes_are_rejected(tmp_path):
    module = script()
    path, checkpoint, report = report_fixture(tmp_path)
    Path(report["source_config"]["path"]).write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="source_config SHA256"):
        module.verify_training_report(path, checkpoint)


def test_latency_percentiles_and_multi_view_throughput_are_distinct():
    result = script().latency_summary([80., 160.], 8)
    assert result["mean_cycle_ms"] == 120
    assert result["p50_cycle_ms"] == 120 and result["p95_cycle_ms"] == 156
    assert result["p99_cycle_ms"] == 159.2
    assert result["images_per_second"] == pytest.approx(66.66666667)
    assert result["camera_work_cycles_per_second"] == pytest.approx(8.33333333)
    for bad in ([], [float("nan")], [0.], [-1.], [True]):
        with pytest.raises(ValueError):
            script().latency_summary(bad, 8)


def test_probability_upsamples_logits_before_softmax_and_uses_road_zero():
    module = script()
    logits = torch.tensor([[[[0., 4.], [-2., 1.]], [[2., -1.], [3., 0.]]]])
    class Network:
        def __call__(self, tensor):
            return logits
    probability = module.road_probability(Network(), torch.zeros(1, 3, 32, 64), (32, 64))
    expected = F.interpolate(logits, size=(32, 64), mode="bilinear", align_corners=False).softmax(1)[:, 0]
    assert torch.equal(probability, expected)
    assert not torch.allclose(probability, F.interpolate(logits.softmax(1)[:, :1], size=(32, 64), mode="bilinear", align_corners=False)[:, 0])
    with pytest.raises(ValueError, match="raw"):
        module.road_probability(lambda image: logits.argmax(1), torch.zeros(1, 3, 32, 64), (32, 64))
    with pytest.raises(ValueError, match="raw"):
        module.road_probability(lambda image: logits.to(torch.int64), torch.zeros(1, 3, 32, 64), (32, 64))


def test_raw_head_class_mapping_rejects_reversed_or_exported_models():
    module = script()
    class Head:
        nc = 2
        export = False
        bake_argmax = False
    class Network:
        yaml = {"scale": "x", "yaml_file": "yolo26x-sem.yaml"}
        names = {0: "road_surface", 1: "not_road_surface"}
        model = [Head()]
    network = Network()
    wrapper = SimpleNamespace(task="semantic", model=network)
    assert module.validate_yolo_contract(wrapper, Network, Head) is network
    network.names = {0: "not_road_surface", 1: "road_surface"}
    with pytest.raises(ValueError, match="class names"):
        module.validate_yolo_contract(wrapper, Network, Head)
    network.names = module.CLASS_NAMES
    network.model[0].export = True
    with pytest.raises(ValueError, match="unexported"):
        module.validate_yolo_contract(wrapper, Network, Head)


def test_explicit_other_yolo_generation_rejected_without_guessing_from_custom_filename():
    module = script()
    class Head:
        nc, export, bake_argmax = 2, False, False
    class Network:
        names = module.CLASS_NAMES
        model = [Head()]
    network = Network()
    wrapper = SimpleNamespace(task="semantic", model=network)
    network.yaml = {"scale": "x", "yaml_file": "yolo99x-sem.yaml"}
    with pytest.raises(ValueError, match="different YOLO"):
        module.validate_yolo_contract(wrapper, Network, Head)
    for filename in ("yolo26-sem.yaml", "yolo26x-sem.yaml", "custom_road_architecture.yaml"):
        network.yaml["yaml_file"] = filename
        assert module.validate_yolo_contract(wrapper, Network, Head) is network


def test_real_image_selection_skips_synthetic_and_preserves_rgb(tmp_path):
    module = script()
    Image.fromarray(np.full((24, 48, 3), [255, 0, 7], dtype=np.uint8)).save(tmp_path / "real.png")
    Image.fromarray(np.zeros((24, 48, 3), dtype=np.uint8)).save(tmp_path / "synthetic.png")
    manifest = tmp_path / "val.jsonl"
    rows = [{"image": "synthetic.png", "synthetic": True}, {"image": "real.png", "dataset_id": "bdd100k_drivable"}]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    images, records = module.select_images({"data": {"val_manifest": str(manifest)}}, None, 1)
    tensor = module.prepare_image(images[0], (32, 64), torch.device("cpu"), "fp32")
    assert tensor.shape == (1, 3, 32, 64)
    assert tensor[0, 0, 0, 0] == 1 and tensor[0, 1, 0, 0] == 0
    assert records[0]["dataset_id"] == "bdd100k_drivable"
    images[0].close()


def test_warmup_exclusion_and_host_output_dataflow():
    module = script()
    calls = []
    class Network:
        def __call__(self, tensor):
            calls.append(tuple(tensor.shape))
            return tensor[:, :2, ::8, ::8]
    monitor = SimpleNamespace(sample=lambda: None)
    images = [Image.fromarray(np.full((24, 48, 3), 20 + i, dtype=np.uint8)) for i in range(8)]
    result = module.measure_mode(Network(), images, device=torch.device("cpu"), precision="fp32",
                                 input_size=(32, 64), iterations=2, warmup=1,
                                 mode="host_to_road_map", monitor=monitor)
    assert len(calls) == 24 and set(calls) == {(1, 3, 32, 64)}
    assert result["measured_cycles"] == 2 and result["views_per_cycle"] == 8
    assert result["output_shape"] == [1, 32, 64] and result["output_bytes_per_image"] == 32 * 64 * 4
    assert result["raw_model_logits"] == {"shape": [1, 2, 4, 8], "dtype": "torch.float32"}
    for image in images:
        image.close()


@pytest.mark.parametrize("mode,preparations", [("batched_model_only", 1), ("batched_host_to_road_map", 3)])
def test_batch_eight_one_forward_per_cycle_and_host_preparation(mode, preparations, monkeypatch):
    module = script()
    calls, prepared = [], []
    original_prepare = module.prepare_batch
    def prepare(*args):
        tensor = original_prepare(*args)
        prepared.append(tuple(tensor.shape))
        return tensor
    monkeypatch.setattr(module, "prepare_batch", prepare)
    class Network:
        def __call__(self, tensor):
            calls.append((tuple(tensor.shape), tensor[:, 0, 0, 0].clone()))
            return tensor[:, :2, ::8, ::8]
    images = [Image.fromarray(np.full((24, 48, 3), 20 + i, dtype=np.uint8)) for i in range(8)]
    result = module.measure_mode(Network(), images, device=torch.device("cpu"), precision="fp32",
                                 input_size=(32, 64), iterations=2, warmup=1, mode=mode,
                                 monitor=SimpleNamespace(sample=lambda: None))
    assert prepared == [(8, 3, 32, 64)] * preparations
    assert len(calls) == 3 and all(shape == (8, 3, 32, 64) for shape, _ in calls)
    assert torch.allclose(calls[0][1], torch.arange(20, 28, dtype=torch.float32) / 255)
    assert result["measured_cycles"] == 2 and result["views_per_cycle"] == 8
    assert result["model_calls_per_cycle"] == 1 and result["batch_size_per_call"] == 8
    assert result["output_shape"] == [8, 32, 64]
    assert result["output_bytes_per_image"] == 32 * 64 * 4
    assert result["output_bytes_per_cycle"] == 8 * 32 * 64 * 4
    assert result["raw_model_logits"]["shape"] == [8, 2, 4, 8]
    for image in images:
        image.close()


def test_probability_rejects_wrong_batch_size():
    module = script()
    with pytest.raises(ValueError, match="matching input batch"):
        module.road_probability(lambda image: torch.zeros(1, 2, 4, 8), torch.zeros(8, 3, 32, 64), (32, 64))


def test_setup_failure_is_atomically_saved_without_false_pass(tmp_path):
    module = script()
    output = tmp_path / "result.json"
    result = module.benchmark(tmp_path / "missing.pt", tmp_path / "missing.json",
                               tmp_path / "native.json", tmp_path / "native.pt", output,
                               precision="fp16", device="cpu")
    assert result["status"] == "failed" and result["error_type"] == "ValueError"
    assert not result["physical_output"] and not result["accuracy_evaluated"]
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        module.atomic_report(output, {"status": "wrong"})
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_nonfinal_view_nonfinite_output_cannot_be_hidden_by_last_view():
    module = script()
    calls = 0
    class Network:
        def __call__(self, tensor):
            nonlocal calls
            calls += 1
            logits = torch.zeros(1, 2, 4, 8)
            if calls == 1:
                logits[:] = float("nan")
            return logits
    images = [Image.new("RGB", (16, 16)) for _ in range(8)]
    with pytest.raises(FloatingPointError, match="Road probability"):
        module.measure_mode(Network(), images, device=torch.device("cpu"), precision="fp32",
                            input_size=(32, 64), iterations=1, warmup=0,
                            mode="model_only", monitor=SimpleNamespace(sample=lambda: None))
    assert calls == 8
    for image in images:
        image.close()


def test_one_model_oom_still_records_other_model_without_claiming_success(tmp_path, monkeypatch):
    module = script()
    report_path, checkpoint, proof = report_fixture(tmp_path)
    source_config = Path(proof["source_config"]["path"])
    config = {"data": {"image_size": [32, 64]}}
    for key in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "select_images", lambda *args: ([Image.new("RGB", (32, 64))], []))
    def fail_yolo(path):
        raise torch.cuda.OutOfMemoryError("controlled allocation failure; no GPU used")
    monkeypatch.setattr(module, "load_yolo", fail_yolo)
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, tensor):
            return tensor[:, :2, ::8, ::8] + self.weight
    monkeypatch.setattr(module, "load_native", lambda *args: (Network(), {"model_class": "TestNetwork"}))
    monkeypatch.setattr(module, "ResourceMonitor", lambda *args: SimpleNamespace(sample=lambda: None, report=lambda: {}))
    output = tmp_path / "partial.json"
    result = module.benchmark(checkpoint, report_path, source_config, checkpoint, output,
                              input_size=(32, 64), iterations=1, warmup=1, device="cpu", precision="fp32")
    assert result["status"] == "partial_failure"
    assert result["models"]["yolo26x_sem"]["out_of_memory"]
    assert result["models"]["yolo26x_sem"]["status"] == "failed"
    assert result["models"]["tinyunet"]["status"] == "completed"
    assert result["models"]["tinyunet"]["model_only"]["measured_cycles"] == 1
    assert result["models"]["tinyunet"]["host_to_road_map"]["measured_cycles"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "partial_failure"


def test_batched_oom_preserves_completed_sequential_metrics(tmp_path, monkeypatch):
    module = script()
    report_path, checkpoint, proof = report_fixture(tmp_path)
    source_config = Path(proof["source_config"]["path"])
    for key in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(module, "load_config", lambda path: {"data": {"image_size": [32, 64]}})
    monkeypatch.setattr(module, "select_images", lambda *args: ([Image.new("RGB", (32, 64)) for _ in range(8)], []))
    class Network(torch.nn.Module):
        def __init__(self, fail_batch):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.fail_batch = fail_batch
        def forward(self, tensor):
            if tensor.shape[0] == 8 and self.fail_batch:
                raise torch.cuda.OutOfMemoryError("controlled batch-only failure; no GPU used")
            return tensor[:, :2, ::8, ::8] + self.weight
    monkeypatch.setattr(module, "load_yolo", lambda *args: (Network(True), {"model_class": "TestNetwork"}))
    monkeypatch.setattr(module, "load_native", lambda *args: (Network(False), {"model_class": "TestNetwork"}))
    monkeypatch.setattr(module, "ResourceMonitor", lambda *args: SimpleNamespace(sample=lambda: None, report=lambda: {}))
    output = tmp_path / "batch_oom.json"
    result = module.benchmark(checkpoint, report_path, source_config, checkpoint, output,
                              input_size=(32, 64), views=8, iterations=1, warmup=1, device="cpu", precision="fp32")
    yolo, native = result["models"]["yolo26x_sem"], result["models"]["tinyunet"]
    assert result["status"] == "partial_failure" and yolo["status"] == "failed" and yolo["out_of_memory"]
    assert yolo["failed_mode"] == "batched_model_only"
    assert yolo["failed_mode_cuda_memory"]["torch_peak_allocated_gib"] is None
    assert not yolo["failed_mode_cuda_memory"]["allocator_peaks_reset"]
    assert yolo["model_only"]["measured_cycles"] == yolo["host_to_road_map"]["measured_cycles"] == 1
    assert "batched_model_only" not in yolo and "batched_host_to_road_map" not in yolo
    assert native["status"] == "completed"
    assert native["batched_model_only"]["batch_size_per_call"] == 8
    assert native["batched_host_to_road_map"]["output_shape"] == [8, 32, 64]
    assert json.loads(output.read_text(encoding="utf-8"))["models"]["yolo26x_sem"]["failed_mode"] == "batched_model_only"


def test_dtype_device_move_and_per_mode_peaks_preserve_cumulative_peak(tmp_path, monkeypatch):
    """Fake CUDA counters exercise accounting; no GPU allocation or model is used."""
    module = script()
    report_path, checkpoint, proof = report_fixture(tmp_path)
    source_config = Path(proof["source_config"]["path"])
    for key in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE", "YOLO_AUTOINSTALL"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(module, "load_config", lambda path: {"data": {"image_size": [32, 64]}})
    monkeypatch.setattr(module, "select_images", lambda *args: ([Image.new("RGB", (32, 64)) for _ in range(8)], []))
    counters = {"allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0, "resets": 0}
    gib = 2**30
    moves = []
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def to(self, *, device, dtype):
            moves.append((str(device), dtype))
            counters.update(allocated=gib // 8, reserved=gib // 4,
                            peak_allocated=gib // 8, peak_reserved=gib // 4)
            return self  # Deliberately remain on CPU; this test mocks CUDA only.
        def half(self):
            raise AssertionError("Must cast in the device transfer, not after an FP32 CUDA allocation")
    monkeypatch.setattr(module, "load_yolo", lambda *args: (Network(), {}))
    monkeypatch.setattr(module, "load_native", lambda *args: (Network(), {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=16 * gib, name="Mock CUDA; no hardware measurement", uuid=None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    def reset(*args):
        counters["resets"] += 1
        counters["peak_allocated"] = counters["allocated"]
        counters["peak_reserved"] = counters["reserved"]
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    for api, key in (("memory_allocated", "allocated"), ("memory_reserved", "reserved"),
                     ("max_memory_allocated", "peak_allocated"), ("max_memory_reserved", "peak_reserved")):
        monkeypatch.setattr(torch.cuda, api, lambda *args, key=key: counters[key])
    monkeypatch.setattr(module, "nvidia_smi_snapshot", lambda *args: {"available": False})
    monkeypatch.setattr(module, "ResourceMonitor", lambda *args: SimpleNamespace(
        sample=lambda: None, report=lambda: {"torch_peak_allocated_gib": .125, "torch_peak_reserved_gib": .25}))
    peaks = {"model_only": .5, "host_to_road_map": .75,
             "batched_model_only": 2., "batched_host_to_road_map": 1.5}
    def measure(*args, mode, **kwargs):
        # Reset must happen for every mode, even after a larger previous peak.
        assert counters["peak_allocated"] == counters["allocated"] == gib // 8
        counters.update(peak_allocated=int(peaks[mode] * gib), peak_reserved=int((peaks[mode] + .25) * gib))
        return {"measured_cycles": 1}
    monkeypatch.setattr(module, "measure_mode", measure)
    result = module.benchmark(checkpoint, report_path, source_config, checkpoint, tmp_path / "memory.json",
                              input_size=(32, 64), views=8, iterations=1, warmup=1, device="cuda", precision="fp16")
    assert result["status"] == "completed"
    assert moves == [("cuda:0", torch.float16), ("cuda:0", torch.float16)]
    assert counters["resets"] == 10  # Initial model load and four modes, for each model.
    for model in result["models"].values():
        assert model["model_load_cuda_memory"]["torch_peak_allocated_gib"] == .125
        for mode, peak in peaks.items():
            memory = model[mode]["cuda_memory"]
            assert memory["torch_peak_allocated_gib"] == peak
            assert memory["torch_peak_reserved_gib"] == peak + .25
            assert memory["torch_allocated_at_mode_start_gib"] == .125
            assert memory["allocator_peaks_reset"]
        assert model["resources"]["torch_peak_allocated_gib"] == 2.
        assert model["resources"]["torch_peak_reserved_gib"] == 2.25
        assert "all attempted" in model["resources"]["torch_peak_scope"]


@pytest.mark.skipif(os.name != "nt", reason="Windows rename has no-overwrite semantics")
def test_windows_atomic_rename_fallback_cannot_replace_report(tmp_path, monkeypatch):
    module = script()
    def denied_link(*args):
        raise PermissionError("Hard links denied by the restricted token")
    monkeypatch.setattr(module.os, "link", denied_link)
    output = tmp_path / "result.json"
    module.atomic_report(output, {"status": "first"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
    with pytest.raises(FileExistsError):
        module.atomic_report(output, {"status": "must_not_replace"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
