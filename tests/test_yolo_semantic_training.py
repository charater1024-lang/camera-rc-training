import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import yaml


SPEC = importlib.util.spec_from_file_location("semantic_training", Path(__file__).parents[1] / "scripts/train_yolo_semantic.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_amp_helper_updates_imported_global_without_touching_baseline(tmp_path, monkeypatch):
    path = tmp_path / 'pretrained/yolo26n.pt'
    path.parent.mkdir()
    path.write_bytes(b'verified local fixture')
    monkeypatch.setattr(MODULE, 'AMP_HELPER_SHA256', MODULE.identity(path)['sha256'])
    settings = {'weights_dir': 'prior-run/weights'}
    utils = SimpleNamespace(WEIGHTS_DIR=Path('stale-import-default'))
    proof = MODULE.configure_amp_helper(path, settings, utils)
    assert utils.WEIGHTS_DIR == path.parent
    assert settings['weights_dir'] == str(path.parent)
    assert proof == MODULE.identity(path)


def test_unverified_amp_helper_fails_before_settings_change(tmp_path):
    path = tmp_path / 'yolo26n.pt'
    path.write_bytes(b'not official')
    settings = {'weights_dir': 'original'}
    utils = SimpleNamespace(WEIGHTS_DIR=Path('original'))
    with pytest.raises(ValueError, match='verified local'):
        MODULE.configure_amp_helper(path, settings, utils)
    assert settings['weights_dir'] == 'original'
    assert utils.WEIGHTS_DIR == Path('original')


def fixture_export(tmp_path):
    config = tmp_path / "source.json"
    config.write_text('{}', encoding="utf-8")
    for split in ("train", "val"):
        (tmp_path / f"{split}.txt").write_text(str(tmp_path / f"{split}.jpg") + "\n", encoding="utf-8")
    data = {"train": str(tmp_path / "train.txt"), "val": str(tmp_path / "val.txt"),
            "names": dict(enumerate(MODULE.CLASS_NAMES)), "masks_dir": "masks"}
    (tmp_path / "dataset.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    lists = {s: MODULE.identity(tmp_path / f"{s}.txt") for s in ("train", "val")}
    MODULE.write_json(tmp_path / "report.json", {"complete": True, "diagnostic_only": False, "source_config": MODULE.identity(config),
                      "image_lists": lists, "manifest_outputs": lists,
                      "full_input_counts": {"train": {"fixture": 1}, "val": {"fixture": 1}}})
    marker = {"report_sha256": MODULE.identity(tmp_path / "report.json")["sha256"],
              "dataset_yaml_sha256": MODULE.identity(tmp_path / "dataset.yaml")["sha256"], "diagnostic_only": False}
    MODULE.write_json(tmp_path / ".complete", marker)
    return config


def test_complete_export(tmp_path):
    config = fixture_export(tmp_path)
    data, source = MODULE.verify_export(tmp_path, config)
    assert data["counts"] == {"train": 1, "val": 1}
    assert source == MODULE.identity(config)


@pytest.mark.parametrize("mutation", ["incomplete", "changed", "diagnostic", "config", "list"])
def test_rejects_invalid_provenance(tmp_path, mutation):
    config = fixture_export(tmp_path)
    if mutation == "incomplete":
        (tmp_path / ".incomplete").touch()
    elif mutation == "changed":
        (tmp_path / "dataset.yaml").write_text('names: [bad]', encoding="utf-8")
    elif mutation == "diagnostic":
        marker = json.loads((tmp_path / ".complete").read_text())
        marker["diagnostic_only"] = True
        MODULE.write_json(tmp_path / ".complete", marker)
    elif mutation == "config":
        config.write_text('{"changed":true}', encoding="utf-8")
    else:
        (tmp_path / "train.txt").write_text('unverified.jpg\n', encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.verify_export(tmp_path, config)


@pytest.mark.parametrize("args", [(0,384,8,2,1e-4), (1,385,8,2,1e-4), (1,384,0,2,1e-4), (1,384,8,-1,1e-4), (1,384,8,2,float('nan'))])
def test_invalid_parameters(args):
    with pytest.raises(ValueError):
        MODULE.validate_parameters(*args)


def completed_parent(tmp_path, monkeypatch):
    """Only provenance is exercised; these bytes are never unpickled."""
    config = fixture_export(tmp_path)
    official, best, last = [tmp_path / name for name in ("official.pt", "best.pt", "last.pt")]
    for path in (official, best, last):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(MODULE, "OFFICIAL_SHA256", MODULE.identity(official)["sha256"])
    dataset, source = MODULE.verify_export(tmp_path, config)
    report = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic",
              "class_names": MODULE.CLASS_NAMES, "road_class_id": 0,
              "training_complete": True, "diagnostic": False, "source_config": source,
              "dataset": dataset, "pretrained": MODULE.identity(official),
              "checkpoints": {"best": MODULE.identity(best), "last": MODULE.identity(last)},
              "actual_args": {"epochs": 1, "batch": 1, "resume": False, "fraction": 1.0},
              "history": [{"epoch": 1.0}], "batches_per_epoch": 1,
              "progress": {"train_batches": 1, "optimizer_attempts": 1,
                           "optimizer_updates": 1, "amp_skipped_updates": 0}}
    path = tmp_path / "training_report.json"
    MODULE.write_json(path, report)
    return path, report, official, best, last


def test_default_initializer_remains_pinned_to_official_weights(tmp_path, monkeypatch):
    _, _, official, best, _ = completed_parent(tmp_path, monkeypatch)
    selected, proof = MODULE.verify_initialization(official)
    assert selected == MODULE.identity(official)
    assert proof["kind"] == "official_pretrained"
    assert proof["optimizer_state"] == "fresh" and proof["resume"] is False
    with pytest.raises(ValueError, match="official pretrained"):
        MODULE.verify_initialization(best)


def test_completed_best_initializer_records_fresh_optimizer_and_exact_parent(tmp_path, monkeypatch):
    path, report, official, best, _ = completed_parent(tmp_path, monkeypatch)
    selected, proof = MODULE.verify_initialization(best, path)
    assert selected == MODULE.identity(best)
    assert proof["kind"] == "completed_best_checkpoint"
    assert proof["training_report"] == MODULE.identity(path)
    assert proof["parent_source_config"] == report["source_config"]
    assert proof["parent_dataset"] == report["dataset"]
    assert proof["parent_completed_epochs"] == proof["parent_optimizer_updates"] == 1
    assert proof["root_pretrained"] == MODULE.identity(official)
    assert proof["ancestor_training_reports"] == [MODULE.identity(path)]
    assert proof["optimizer_state"] == "fresh" and proof["resume"] is False


@pytest.mark.parametrize("field,value", [
    ("training_complete", False), ("diagnostic", True), ("schema_version", True),
    ("model_family", "yolo26n-sem"), ("task", "detect"), ("road_class_id", True),
    ("class_names", ["not_road_surface", "road_surface"]), ("history", []),
    ("history", [{"epoch": 2.0}]), ("history", [{"epoch": True}]),
    ("batches_per_epoch", 2), ("initialization", {"kind": "unverified"}),
])
def test_incomplete_or_incompatible_initializer_report_is_rejected(tmp_path, monkeypatch, field, value):
    path, report, _, best, _ = completed_parent(tmp_path, monkeypatch)
    report[field] = value
    MODULE.write_json(path, report)
    with pytest.raises(ValueError):
        MODULE.verify_initialization(best, path)


@pytest.mark.parametrize("section,field,value", [
    ("actual_args", "resume", True), ("actual_args", "fraction", 0.1),
    ("actual_args", "epochs", 2), ("actual_args", "batch", True),
    ("progress", "train_batches", 0), ("progress", "optimizer_updates", 0),
    ("progress", "amp_skipped_updates", 1), ("progress", "optimizer_attempts", 2),
])
def test_initializer_requires_actual_full_training_and_successful_updates(tmp_path, monkeypatch, section, field, value):
    path, report, _, best, _ = completed_parent(tmp_path, monkeypatch)
    report[section][field] = value
    MODULE.write_json(path, report)
    with pytest.raises(ValueError):
        MODULE.verify_initialization(best, path)


@pytest.mark.parametrize("name", ["best.pt", "last.pt", "official.pt", "source.json", "train.txt", "report.json"])
def test_changed_parent_artifacts_are_rejected_before_checkpoint_loading(tmp_path, monkeypatch, name):
    path, _, _, best, _ = completed_parent(tmp_path, monkeypatch)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError):
        MODULE.verify_initialization(best, path)


def test_last_or_unrecorded_copy_is_not_selected_for_finetuning(tmp_path, monkeypatch):
    path, _, _, best, last = completed_parent(tmp_path, monkeypatch)
    copied = tmp_path / "unrecorded.pt"
    copied.write_bytes(best.read_bytes())
    for weights in (last, copied):
        with pytest.raises(ValueError, match="best checkpoint"):
            MODULE.verify_initialization(weights, path)


def test_staged_finetune_lineage_is_verified_back_to_official_initializer(tmp_path, monkeypatch):
    path, report, official, best, _ = completed_parent(tmp_path, monkeypatch)
    _, initialization = MODULE.verify_initialization(best, path)
    second_report = json.loads(json.dumps(report))
    second_report["pretrained"] = MODULE.identity(best)
    second_report["initialization"] = initialization
    for role in ("best", "last"):
        artifact = tmp_path / f"second_{role}.pt"
        artifact.write_bytes(f"second {role}".encode())
        second_report["checkpoints"][role] = MODULE.identity(artifact)
    second_path = tmp_path / "second_training_report.json"
    MODULE.write_json(second_path, second_report)
    second_best = Path(second_report["checkpoints"]["best"]["path"])
    _, proof = MODULE.verify_initialization(second_best, second_path)
    assert proof["root_pretrained"] == MODULE.identity(official)
    assert proof["ancestor_training_reports"] == [MODULE.identity(second_path), MODULE.identity(path)]
    report["note"] = "Changed after selecting the initializer"
    MODULE.write_json(path, report)
    with pytest.raises(ValueError, match="ancestor training report"):
        MODULE.verify_initialization(second_best, second_path)


class FakeSemanticHead:
    nc, export, bake_argmax = 2, False, False


class FakeSemanticModel:
    def __init__(self):
        self.model = [FakeSemanticHead()]
        self.names = dict(enumerate(MODULE.CLASS_NAMES))
        self.yaml = {"scale": "x", "yaml_file": "yolo26x-sem.yaml"}


def test_loaded_finetune_initializer_must_keep_canonical_two_class_head():
    wrapper = SimpleNamespace(task="semantic", model=FakeSemanticModel())
    assert MODULE.validate_finetune_model(wrapper, model_class=FakeSemanticModel,
                                          head_class=FakeSemanticHead) is wrapper.model


@pytest.mark.parametrize("wrong", ["task", "scale", "generation", "classes", "names", "export", "bake_argmax"])
def test_incompatible_loaded_finetune_initializer_is_rejected(wrong):
    wrapper = SimpleNamespace(task="semantic", model=FakeSemanticModel())
    if wrong == "task":
        wrapper.task = "detect"
    elif wrong in {"scale", "generation"}:
        wrapper.model.yaml["scale" if wrong == "scale" else "yaml_file"] = "n" if wrong == "scale" else "yolo11x-sem.yaml"
    elif wrong == "classes":
        wrapper.model.model[-1].nc = 19
    elif wrong == "names":
        wrapper.model.names = {0: "not_road_surface", 1: "road_surface"}
    else:
        setattr(wrapper.model.model[-1], wrong, True)
    with pytest.raises(ValueError):
        MODULE.validate_finetune_model(wrapper, model_class=FakeSemanticModel, head_class=FakeSemanticHead)


@pytest.mark.parametrize("physical,nominal,steps,effective", [(4, None, 1, 4), (4, 32, 8, 32), (8, 32, 4, 32), (6, 32, 5, 30)])
def test_nominal_batch_retains_old_default_and_reports_rounding(physical, nominal, steps, effective):
    policy = MODULE.batch_policy(physical, nominal)
    assert policy["nominal_batch"] == (physical if nominal is None else nominal)
    assert policy["nominal_batch_explicit"] is (nominal is not None)
    assert policy["target_accumulation_steps"] == steps
    assert policy["target_full_batch_samples_per_update"] == effective
    assert "warmup" in policy["note"] and "BatchNorm" in policy["note"]


@pytest.mark.parametrize("physical,nominal", [(True, 32), (4.0, 32), (0, 32), (4, 3), (4, 32.0), (4, True)])
def test_invalid_nominal_batch_fails(physical, nominal):
    with pytest.raises(ValueError, match="batch"):
        MODULE.batch_policy(physical, nominal)


class FakeCUDA:
    def __init__(self, calls):
        self.calls, self.fraction = calls, 1.0

    def is_available(self):
        return True

    def get_device_properties(self, device):
        self.calls.append(("device_properties", device))
        return SimpleNamespace(total_memory=16 * 1024 ** 3)

    def set_per_process_memory_fraction(self, fraction, *, device):
        self.calls.append(("set_memory_fraction", fraction, device))
        self.fraction = fraction

    def get_per_process_memory_fraction(self, *, device):
        self.calls.append(("get_memory_fraction", device))
        return self.fraction

    def set_device(self, device):
        self.calls.append(("set_device", device))

    def reset_peak_memory_stats(self):
        pass

    def get_device_name(self, device):
        return "CPU test double; not real GPU evidence"

    def max_memory_allocated(self):
        return 1024

    def max_memory_reserved(self):
        return 4096


def test_allocator_cap_validates_physical_capacity_and_sets_explicit_device():
    calls = []
    torch = SimpleNamespace(cuda=FakeCUDA(calls))
    policy = MODULE.configure_gpu_memory_limit(torch, 1, 5)
    assert policy["applied"] and policy["requested_limit_bytes"] == 5 * 1024 ** 3
    assert policy["fraction"] == 5 / 16 and policy["device_index"] == 1
    assert calls == [("device_properties", 1), ("set_memory_fraction", 5 / 16, 1), ("get_memory_fraction", 1)]
    assert "Other processes" in policy["limitation"]


def test_omitted_allocator_cap_does_not_change_legacy_allocator():
    calls = []
    policy = MODULE.configure_gpu_memory_limit(SimpleNamespace(cuda=FakeCUDA(calls)), 0, None)
    assert not policy["applied"] and calls == []


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), 1e308, 10 ** 400, 1e-20, True, "4"])
def test_invalid_memory_limit_fails_without_cuda_queries(limit):
    calls = []
    with pytest.raises(ValueError, match="finite and positive"):
        MODULE.configure_gpu_memory_limit(SimpleNamespace(cuda=FakeCUDA(calls)), 0, limit)
    assert not calls


def test_memory_limit_above_physical_total_does_not_set_cap():
    calls = []
    with pytest.raises(ValueError, match="physical total"):
        MODULE.configure_gpu_memory_limit(SimpleNamespace(cuda=FakeCUDA(calls)), 0, 17)
    assert calls == [("device_properties", 0)]


def test_allocator_readback_mismatch_fails():
    cuda = FakeCUDA([])
    cuda.get_per_process_memory_fraction = lambda **kwargs: .5
    with pytest.raises(RuntimeError, match="did not retain"):
        MODULE.configure_gpu_memory_limit(SimpleNamespace(cuda=cuda), 0, 4)


def test_new_cli_controls_are_optional_and_typed():
    required = ["--dataset", "data", "--config", "config.json", "--weights", "best.pt", "--output", "run"]
    defaults = MODULE.build_parser().parse_args(required)
    assert defaults.gpu_memory_gib is None and defaults.nominal_batch is None
    explicit = MODULE.build_parser().parse_args([*required, "--batch", "4", "--nominal-batch", "32", "--gpu-memory-gib", "5.5"])
    assert explicit.batch == 4 and explicit.nominal_batch == 32 and explicit.gpu_memory_gib == 5.5


def mocked_training(tmp_path, monkeypatch, *, cap=None, nominal=None, reduce_batch=False, physical=4):
    """Exercise argument wiring, callbacks and reports without importing real CUDA code."""
    config = fixture_export(tmp_path)
    weights, helper = tmp_path / "official.pt", tmp_path / "yolo26n.pt"
    weights.write_bytes(b"official fixture")
    helper.write_bytes(b"helper fixture")
    monkeypatch.setattr(MODULE, "OFFICIAL_SHA256", MODULE.identity(weights)["sha256"])
    monkeypatch.setattr(MODULE, "configure_amp_helper", lambda *args: MODULE.identity(helper))
    calls, state = [], {}
    cuda = FakeCUDA(calls)
    torch = ModuleType("torch")
    torch.cuda, torch.__version__, torch.version = cuda, "cpu-test-double", SimpleNamespace(cuda="test")
    torch.set_num_threads = lambda count: None
    torch.tensor = lambda *args, **kwargs: calls.append(("tensor", kwargs.get("device"))) or SimpleNamespace()
    torchvision = ModuleType("torchvision")
    torchvision.__version__ = "cpu-test-double"
    def nms(*args):
        calls.append(("cuda_nms",))
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0]))
    torchvision.ops = SimpleNamespace(nms=nms)
    package_names = ["ultralytics", "ultralytics.models", "ultralytics.models.yolo", "ultralytics.models.yolo.semantic", "ultralytics.models.yolo.semantic.train"]
    packages = {name: ModuleType(name) for name in package_names}
    for module in packages.values():
        module.__path__ = []
    ultra = packages["ultralytics"]
    ultra.__version__, ultra.settings, ultra.utils = "cpu-test-double", {}, SimpleNamespace()
    class Loader:
        drop_last = False
        dataset = [None]
        def __len__(self):
            return 1
    class BaseTrainer:
        def __init__(self, options):
            self.args = SimpleNamespace(**options)
            self.batch_size = options["batch"]
            self.accumulate = min(max(round(options["nbs"] / self.batch_size), 1), options["epochs"])
            self.start_epoch = self.epoch = 0
            self.amp = True
            self.data = {"names": dict(enumerate(MODULE.CLASS_NAMES))}
            self.train_loader = self.test_loader = Loader()
            self.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=True, numel=lambda: 10)])
            self.scaler = SimpleNamespace(get_scale=lambda: 16.)
            self.save_dir = Path(options["project"]) / "fit"
            self.save_dir.mkdir()
            self.best, self.last = self.save_dir / "best.pt", self.save_dir / "last.pt"
            self.best.write_bytes(b"best fixture")
            self.last.write_bytes(b"last fixture")
            (self.save_dir / "results.csv").write_text("epoch,metric\n1,0.5\n")
        def _build_train_pipeline(self):
            calls.append(("build_pipeline", self.batch_size))
        def optimizer_step(self):
            calls.append(("optimizer_step",))
    packages["ultralytics.models.yolo.semantic.train"].SemanticSegmentationTrainer = BaseTrainer
    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            self.task, self.model, self.callbacks = "semantic", FakeSemanticModel(), {}
        def add_callback(self, name, callback):
            self.callbacks[name] = callback
        def train(self, *, trainer, **options):
            state["train_kwargs"] = options
            self.trainer = trainer(options)
            self.trainer._build_train_pipeline()
            self.callbacks["on_pretrain_routine_end"](self.trainer)
            if reduce_batch:
                self.trainer._oom_retries = 1
                self.trainer.batch_size = max(self.trainer.batch_size // 2, 1)
                self.trainer.args.batch = self.trainer.batch_size
                self.trainer._build_train_pipeline()
            self.trainer.optimizer_step()
            self.callbacks["on_train_batch_end"](self.trainer)
            self.callbacks["on_fit_epoch_end"](self.trainer)
    ultra.YOLO = FakeYOLO
    for name, module in {"torch": torch, "torchvision": torchvision, **packages}.items():
        monkeypatch.setitem(sys.modules, name, module)
    args = SimpleNamespace(dataset=tmp_path, config=config, weights=weights, output=tmp_path / "run", epochs=1,
                           image_size=384, batch=physical, workers=0, lr=1e-4, device=0)
    if cap is not None:
        args.gpu_memory_gib = cap
    if nominal is not None:
        args.nominal_batch = nominal
    return args, calls, state


def test_capped_nominal_batch_is_passed_to_training_and_persisted(tmp_path, monkeypatch):
    args, calls, state = mocked_training(tmp_path, monkeypatch, cap=5, nominal=32)
    report = MODULE.train(args)
    assert state["train_kwargs"]["batch"] == 4 and state["train_kwargs"]["nbs"] == 32
    names = [call[0] for call in calls]
    assert names.index("set_memory_fraction") < names.index("set_device") < names.index("tensor") < names.index("cuda_nms")
    assert report["gpu_memory_policy"]["applied"] and report["gpu_memory_policy"]["fraction"] == 5 / 16
    assert report["batch_policy"]["target_accumulation_steps"] == 8
    assert report["batch_policy"]["initial_actual_accumulation_steps"] == 1  # short run clamp, not promised8
    assert report["observed_accumulation_batches"] == {"1": 1}
    assert report["actual_args"]["nbs"] == 32 and report["progress"]["optimizer_updates"] == 1
    assert report["training_complete"] and report["automatic_batch_retry_allowed"] is False
    persisted = json.loads((args.output / "training_report.json").read_text())
    assert persisted["gpu_memory_policy"] == report["gpu_memory_policy"]


def test_default_nominal_batch_and_allocator_behavior_remain_unchanged(tmp_path, monkeypatch):
    args, calls, state = mocked_training(tmp_path, monkeypatch)
    report = MODULE.train(args)
    assert state["train_kwargs"]["nbs"] == args.batch
    assert not any(call[0] == "set_memory_fraction" for call in calls)
    assert report["gpu_memory_policy"]["applied"] is False
    assert report["automatic_batch_retry_allowed"] is True


def test_capped_run_blocks_automatic_batch_retry_and_keeps_failure_report(tmp_path, monkeypatch):
    args, calls, _ = mocked_training(tmp_path, monkeypatch, cap=5, nominal=32, reduce_batch=True)
    with pytest.raises(RuntimeError, match="forbids automatic batch reduction"):
        MODULE.train(args)
    report = json.loads((args.output / "training_report.json").read_text())
    assert report["training_complete"] is False and report["blocked_batch_change"] == {"requested": 4, "attempted": 2}
    assert report["progress"]["optimizer_attempts"] == report["progress"]["train_batches"] == 0
    assert [call for call in calls if call[0] == "build_pipeline"] == [("build_pipeline", 4)]
    assert report["gpu_memory_policy"]["requested_gib"] == 5


def test_uncapped_run_does_not_disable_existing_batch_retry_hook(tmp_path, monkeypatch):
    args, calls, _ = mocked_training(tmp_path, monkeypatch, reduce_batch=True)
    report = MODULE.train(args)
    assert [call for call in calls if call[0] == "build_pipeline"] == [("build_pipeline", 4), ("build_pipeline", 2)]
    assert "blocked_batch_change" not in report and report["last_observed_physical_batch"] == 2


def test_capped_batch_one_still_blocks_same_size_oom_epoch_retry(tmp_path, monkeypatch):
    args, calls, _ = mocked_training(tmp_path, monkeypatch, cap=5, nominal=32, reduce_batch=True, physical=1)
    with pytest.raises(RuntimeError, match="epoch retry"):
        MODULE.train(args)
    report = json.loads((args.output / "training_report.json").read_text())
    assert report["blocked_oom_retry_count"] == 1
    assert report["blocked_batch_change"] == {"requested": 1, "attempted": 1}
    assert [call for call in calls if call[0] == "build_pipeline"] == [("build_pipeline", 1)]
