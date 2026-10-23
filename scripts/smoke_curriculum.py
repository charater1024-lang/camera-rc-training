"""Two complete tiny stages with source replay and heldout re-evaluation."""
import argparse
import json
from pathlib import Path
from camera_rc.demo import make_demo
from camera_rc.curriculum import run_curriculum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs = []
    for index in (1, 2):
        generated = make_demo(root / f"source_{index}")
        path = Path(generated["segmentation"])
        cfg = json.loads(path.read_text())
        cfg["training"]["device"] = args.device
        cfg["data"]["label_space_id"] = "synthetic_three_class_v1"
        configs.append((path, cfg))
    path1, cfg1 = configs[0]
    path2, cfg2 = configs[1]
    cfg2["data"].pop("train_manifest")
    cfg2["data"].pop("val_manifest")
    cfg2["data"]["train_sources"] = [
        {"manifest": str(path1.parent / "train.jsonl"), "name": "public_replay", "weight": 1},
        {"manifest": str(path2.parent / "train.jsonl"), "name": "target_domain", "weight": 2}]
    cfg2["data"]["val_manifests"] = [
        {"manifest": str(path1.parent / "val.jsonl"), "name": "public_heldout"},
        {"manifest": str(path2.parent / "val.jsonl"), "name": "target_heldout"}]
    for path, cfg in configs:
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    curriculum = root / "curriculum.json"
    curriculum.write_text(json.dumps({"output_dir": "training", "stages": [
        {"name": "public", "config": str(path1)},
        {"name": "mixed", "config": str(path2), "init_from": "previous"}]}, indent=2), encoding="utf-8")
    run_curriculum(curriculum)
    result = run_curriculum(curriculum, execute=True)
    if not result["training_complete"] or not result.get("diagnostic"):
        raise RuntimeError("Expected a fully executed diagnostic-only curriculum")
    resumed = run_curriculum(curriculum, execute=True, resume_run=True)
    report = {"diagnostic_only": True, "training_complete": result["training_complete"],
              "stages": len(result["stages"]), "resume_skipped_completed_stages": resumed == result,
              "report": str(root / "training/report.json")}
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
