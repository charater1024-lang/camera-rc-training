"""Prepare local official BDD100K/Cityscapes splits without downloading or training."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from camera_rc.curriculum import run_curriculum
from camera_rc.data import ManifestDataset, prepare_cityscapes, read_manifest, remap_masks
from camera_rc.prepare import prepare_bdd_drivable


PROJECT = Path(__file__).resolve().parents[1]
LABEL_SPACE = "road_surface_v1"


def _progress(phase, done, total):
    if done % 1000 == 0 or done == total:
        print(f"{phase}: {done}/{total}", flush=True)


def _decode_all(dataset, workers, phase):
    """Discard decoded tensors immediately; retain at most 2*workers futures."""
    def check(index):
        dataset[index]

    if workers == 0:
        for index in range(len(dataset)):
            check(index)
            _progress(phase, index + 1, len(dataset))
        return
    indices = iter(range(len(dataset)))
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for _ in range(2 * workers):
                index = next(indices, None)
                if index is None:
                    break
                pending.append(pool.submit(check, index))
            done = 0
            while pending:
                pending.popleft().result()
                done += 1
                _progress(phase, done, len(dataset))
                index = next(indices, None)
                if index is not None:
                    pending.append(pool.submit(check, index))
        finally:
            for future in pending:
                future.cancel()


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage(names, *, epochs, device, lr):
    config = {
        "task": "segmentation", "model": {"num_classes": 2, "width": 16},
        "data": {
            "label_space_id": LABEL_SPACE, "image_size": [192, 320],
            "train_sources": [
                {"name": name, "manifest": f"../road/{name}_train/manifest.jsonl", "weight": 1.0}
                for name in names
            ],
            "val_manifests": [
                {"name": name, "manifest": f"../road/{name}_val/manifest.jsonl"}
                for name in names
            ],
        },
        "training": {
            "output_dir": "../training", "epochs": epochs, "device": device,
            "batch_size": 2, "grad_accum_steps": 4, "num_workers": 0,
            "lr": lr, "seed": 42, "amp": True, "amp_init_scale": 1024.0,
        },
    }
    if len(names) == 1:
        # One epoch should visit every frame once for a single dataset.
        # Weighted replacement sampling is retained only for mixed sources.
        config["data"]["train_manifest"] = config["data"].pop("train_sources")[0]["manifest"]
    return config


def prepare_public(bdd_images, bdd_masks, output, *, cityscapes_images=None,
                   cityscapes_labels=None, device="cuda", epochs=20, mixed_epochs=15,
                   bdd_encoding="current_2021", workers=0):
    """Create a fresh, fully checked binary-road curriculum from local raw IDs.

    Official source layouts/IDs and scene identities are checked. This cannot
    authenticate downloaded bytes or determine license compliance. No source
    images or labels are changed; only converted masks are written.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new folder: {output}")
    if (cityscapes_images is None) != (cityscapes_labels is None):
        raise ValueError("Supply both --cityscapes-images and --cityscapes-labels")
    if device not in {"cuda", "cpu", "auto"}:
        raise ValueError("device must be cuda, cpu, or auto")
    if bdd_encoding not in {"current_2021", "legacy_2018"}:
        raise ValueError("Choose current_2021 or legacy_2018 explicitly")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 0 <= workers <= 8:
        raise ValueError("workers must be an integer in 0..8")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (epochs, mixed_epochs)):
        raise ValueError("epochs and mixed_epochs must be positive integers")
    inputs = {"bdd_images": bdd_images, "bdd_masks": bdd_masks}
    if cityscapes_images is not None:
        inputs.update(cityscapes_images=cityscapes_images, cityscapes_labels=cityscapes_labels)
    roots = {name: Path(value).expanduser().resolve() for name, value in inputs.items()}
    for name, root in roots.items():
        if root == output or root in output.parents or output in root.parents:
            raise ValueError(f"Output must be separate from source directory: {name}")
        for split in ("train", "val"):
            if not (root / split).is_dir():
                raise FileNotFoundError(f"{name} must contain official train/val folders: {root / split}")

    output.mkdir(parents=True, exist_ok=False)
    pending = output / ".incomplete"
    pending.write_text("Preparation incomplete. Do not train from this directory.\n", encoding="utf-8")
    report = {
        "complete": False, "training_started": False,
        "source_roots": {name: str(root) for name, root in roots.items()},
        "label_space_id": LABEL_SPACE,
        "bdd_input_encoding": bdd_encoding,
        "preparation_workers": workers,
        "source_policy": "Local official train/val layouts and raw ID masks; no automatic splitting.",
        "authenticity": "Layout, IDs and scene separation checked; download authenticity and license not certified.",
        "duplicate_check": "Canonical scene IDs and resolved paths, not pixel-content hashing.",
        "original_files_modified": False,
    }
    try:
        for name in ("raw", "road", "label_maps", "stages"):
            (output / name).mkdir()
        names = ["bdd"]
        print(f"pairing BDD train/val (input encoding: {bdd_encoding})", flush=True)
        report["pairing"] = {
            "bdd": prepare_bdd_drivable(
                roots["bdd_images"] / "train", roots["bdd_images"] / "val",
                roots["bdd_masks"] / "train", roots["bdd_masks"] / "val", output / "raw/bdd",
            )
        }
        if cityscapes_images is not None:
            names.append("cityscapes")
            report["pairing"]["cityscapes"] = prepare_cityscapes(
                roots["cityscapes_images"], roots["cityscapes_labels"], output / "raw/cityscapes",
            )
        report["manifests"], report["label_map_sha256"] = {}, {}
        probe_config = _stage(names, epochs=epochs, device=device, lr=0.001)
        for name in names:
            map_name = "bdd_2018_to_road.json" if name == "bdd" and bdd_encoding == "legacy_2018" else f"{name}_to_road.json"
            source_map = PROJECT / "configs/label_maps" / map_name
            mapping = output / "label_maps" / source_map.name
            shutil.copyfile(source_map, mapping)
            report["label_map_sha256"][name] = hashlib.sha256(mapping.read_bytes()).hexdigest()
            for split in ("train", "val"):
                manifest_key = f"{name}_{split}"
                print(f"remapping {manifest_key}", flush=True)
                converted = remap_masks(output / "raw" / name / f"{split}.jsonl",
                                        mapping, output / "road" / manifest_key, workers=workers,
                                        progress=lambda done, total: _progress(f"remapped {manifest_key}", done, total))
                rows = read_manifest(converted["manifest"])
                # Decode every image and converted label lazily, one sample at
                # a time, including the exact training resize/ignore rules.
                dataset = ManifestDataset(rows, probe_config)
                print(f"decoding {manifest_key}: {len(dataset)} images and masks", flush=True)
                _decode_all(dataset, workers, f"decoded {manifest_key}")
                report["manifests"][manifest_key] = {
                    "manifest": converted["manifest"], "samples": len(rows),
                    "groups": len({row["group_id"] for row in rows}),
                    "sha256": hashlib.sha256(Path(converted["manifest"]).read_bytes()).hexdigest(),
                    "all_samples_decoded": True,
                    "target_pixel_counts": converted["target_pixel_counts"],
                }

        _write(output / "stages/road_public.json",
               _stage(["bdd"], epochs=epochs, device=device, lr=0.001))
        curriculum = {"output_dir": "training", "stages": [
            {"name": "road_public", "config": "stages/road_public.json"},
        ]}
        if "cityscapes" in names:
            _write(output / "stages/road_mixed.json",
                   _stage(names, epochs=mixed_epochs, device=device, lr=0.0003))
            curriculum["stages"].append({
                "name": "road_mixed", "config": "stages/road_mixed.json", "init_from": "previous",
            })
        # Do not publish the final curriculum name until global split checks
        # pass. Planning is read-only and never allocates a training output.
        temporary_curriculum = output / "curriculum.pending.json"
        _write(temporary_curriculum, curriculum)
        plan = run_curriculum(temporary_curriculum, execute=False)
        report.update(
            complete=True, curriculum=str(output / "curriculum.json"), stages=len(plan["stages"]),
            model=probe_config["model"], image_size=probe_config["data"]["image_size"],
            training_device=device, global_holdouts_checked=plan["validation"]["global_holdouts_checked"],
            deployment_validated=False,
        )
        temporary_curriculum.replace(output / "curriculum.json")
        _write(output / "report.json", report)
        pending.unlink()
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        _write(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-images", type=Path, required=True, help="BDD100K images/100k containing train and val")
    parser.add_argument("--bdd-masks", type=Path, required=True, help="BDD drivable raw ID masks containing train and val")
    parser.add_argument("--cityscapes-images", type=Path, help="Optional leftImg8bit root")
    parser.add_argument("--cityscapes-labels", type=Path, help="Optional gtFine root with raw labelIds")
    parser.add_argument("--output", type=Path, required=True, help="New directory, e.g. data/public_v1")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mixed-epochs", type=int, default=15)
    parser.add_argument("--bdd-encoding", choices=("current_2021", "legacy_2018"), required=True,
                        help="Raw label meaning: current 0/1=road, legacy 1/2=road; never infer from filename")
    parser.add_argument("--workers", type=int, default=0, help="Preparation threads 0..8; 0 is sequential")
    args = parser.parse_args()
    print(json.dumps(prepare_public(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
