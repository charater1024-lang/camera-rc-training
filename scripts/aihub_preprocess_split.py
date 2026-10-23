"""Deterministic, metadata-only TRAINING-to-development split. No file/image I/O."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import ntpath
from pathlib import PurePosixPath, PureWindowsPath
import posixpath
import re


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CLIP = re.compile(r"(?P<device>[0-9]+)_(?P<time>[0-9]{6})_(?P<date>[0-9]{6})\Z")


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Nonempty absolute image/target path required")
    if PureWindowsPath(value).is_absolute():
        return ntpath.normcase(ntpath.normpath(value))
    if PurePosixPath(value).is_absolute():
        return posixpath.normpath(value)
    raise ValueError("Relative image/target paths are forbidden")


def _base_group(row):
    dataset, group, origin = (row.get(k) for k in ("dataset_id", "group_id", "origin_id"))
    if any(not isinstance(v, str) or not v for v in (dataset, group, origin)):
        raise ValueError("dataset_id, original group_id and origin_id are required")
    if dataset == "bdd100k_drivable":
        if origin != "bdd100k:" + group:
            raise ValueError("BDD original clip/group and origin disagree")
        return "bdd_clip:" + group
    if dataset == "vkitti2_0_3":
        scene, video = row.get("source_scene"), row.get("source_video")
        if (not isinstance(scene, str) or not re.fullmatch(r"Scene[0-9]+", scene)
                or not isinstance(video, str) or not video or origin != "kitti:" + video or group != origin):
            raise ValueError("VKITTI requires consistent original scene, drive and origin")
        return "vkitti_scene:" + scene
    if dataset in {"aihub_13", "aihub_14", "aihub_15"}:
        clip = row.get("source_clip")
        match = CLIP.fullmatch(clip or "")
        if not match or group != clip or origin != "aihub_commercial:" + clip:
            raise ValueError("AIHub requires original clip/group/origin agreement")
        # Calendar/time syntax is checked, but names alone do not prove physical identity.
        datetime.strptime(match["date"], "%y%m%d")
        datetime.strptime(match["time"], "%H%M%S")
        if row.get("source_split") != "Training":
            raise ValueError("Only official AIHub Training rows may be repartitioned")
        return f"aihub_filename_device_date:{match['device']}:{match['date']}"
    raise ValueError("Unsupported training dataset: " + dataset)


def partition_rows(rows, seed=42, fraction=0.1):
    """Return copied train/dev rows and a JSON report; never read/write source files.

    Whole-group selection uses only scene/clip/date identities and row counts.
    Pixel hashes connect exact duplicates; target class frequencies, labels and
    model/evaluation scores never influence assignment. Dev is not an unseen test.
    """
    if type(seed) is not int or isinstance(fraction, bool) or type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < .5:
        raise ValueError("Integer seed and finite development fraction between 0 and 0.5 required")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Nonempty rows required")
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    identities, keys, base_keys, datasets = {}, [], [], Counter()
    images, exact_groups, duplicate_links = set(), defaultdict(set), Counter()
    scene_to_drive, drive_to_scene = {}, {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("split") != "train":
            raise ValueError("Only existing train rows may be repartitioned")
        if row.get("export_split", "train") != "train":
            raise ValueError("Validation/test export cannot enter train/dev")
        image, target = _path(row.get("image")), _path(row.get("target"))
        if image in images:
            raise ValueError("Repeated image path; preserve/audit it before splitting")
        images.add(image)
        base = _base_group(row)
        ds = row["dataset_id"]
        if ds == "vkitti2_0_3":
            scene, drive = row["source_scene"], row["source_video"]
            if scene_to_drive.setdefault(scene, drive) != drive or drive_to_scene.setdefault(drive, scene) != scene:
                raise ValueError("VKITTI scene-to-source-drive mapping is inconsistent")
        datasets[ds] += 1
        base_keys.append(base)
        exact_groups[ds].add(row["group_id"])
        item_keys = [("base", base), ("scene", ds, row["group_id"]), ("origin", row["origin_id"]), ("image", image)]
        if row.get("source_image"):
            item_keys.append(("image", _path(row["source_image"])))
        for name, category in (("image_sha256", "byte"), ("image_rgb_sha256", "rgb"), ("source_image_sha256", "byte")):
            value = row.get(name)
            if name == "source_image_sha256" and value is None:
                continue
            if not isinstance(value, str) or not SHA256.fullmatch(value):
                raise ValueError("Verified byte and decoded-RGB hashes required: " + name)
            item_keys.append((category, value))
        # The same image/source hash in one row is not a duplicate observation.
        item_keys = list(dict.fromkeys(item_keys))
        for key in item_keys:
            if key in identities:
                union(index, identities[key])
                if key[0] in {"byte", "rgb"}:
                    duplicate_links[key[0]] += 1
            else:
                identities[key] = index
        keys.append(item_keys)
    grouped = defaultdict(list)
    for i in range(len(rows)):
        grouped[find(i)].append(i)
    components = []
    for indices in grouped.values():
        bases = sorted({base_keys[i] for i in indices})
        component_id = "group:" + _hash(bases)
        counts = Counter(rows[i]["dataset_id"] for i in indices)
        components.append({"group_id": component_id, "base_group_keys": bases, "indices": indices,
                           "rows": len(indices), "by_dataset": dict(sorted(counts.items()))})
    components.sort(key=lambda x: (_hash([seed, x["group_id"]]), x["group_id"]))
    effective_groups = Counter(ds for component in components for ds in component["by_dataset"])
    if any(effective_groups[ds] < 2 for ds in datasets):
        raise ValueError("Every dataset needs at least two independent grouping components for nonempty train/dev")
    chosen, development = set(), Counter()
    def loss(counts):
        return sum((counts[ds] / datasets[ds] - fraction) ** 2 for ds in sorted(datasets))
    def can_add(component):
        return all(development[ds] + n < datasets[ds] for ds, n in component["by_dataset"].items())
    for component in components:
        trial = development + Counter(component["by_dataset"])
        if can_add(component) and loss(trial) < loss(development) - 1e-15:
            chosen.add(component["group_id"])
            development = trial
    # Rare, very large components can prevent a nominal 10% dev; retain whole groups.
    while missing := {ds for ds in datasets if not development[ds]}:
        possible = [component for component in components if component["group_id"] not in chosen
                    and missing.intersection(component["by_dataset"]) and can_add(component)]
        if not possible:
            raise ValueError("Cannot cover all datasets without splitting a group or emptying training")
        component = min(possible, key=lambda x: (loss(development + Counter(x["by_dataset"])), _hash([seed, x["group_id"]])))
        chosen.add(component["group_id"])
        development.update(component["by_dataset"])
    assignment = {}
    for component in components:
        for index in component["indices"]:
            assignment[index] = ("dev" if component["group_id"] in chosen else "train", component["group_id"])
    key_sets = {split: {key for i in assignment if assignment[i][0] == split for key in keys[i]} for split in ("train", "dev")}
    if key_sets["train"] & key_sets["dev"]:
        raise RuntimeError("Group/origin/path/byte/RGB leakage after partition")
    outputs = {"train": [], "dev": []}
    for index, row in enumerate(rows):
        split, group = assignment[index]
        copied = dict(row)
        copied.update(split=split, development_group_id=group)
        outputs[split].append(copied)
    for result in outputs.values():
        result.sort(key=lambda row: (row["dataset_id"], row["group_id"], row.get("sample_id", ""), _path(row["image"])))
    if len(outputs["train"]) + len(outputs["dev"]) != len(rows):
        raise RuntimeError("Partition did not preserve every input row")
    report = {"schema_version": 1, "complete": True, "operation": "metadata_only_train_dev_partition",
              "seed": seed, "target_dev_fraction": fraction,
              "selection_rule": "SHA256(seed, component ID) order; greedily reduce equally weighted per-dataset squared fraction error; preserve nonempty train; then cover any missing dev dataset using the least-error whole component. No claim of globally optimal fraction.",
              "grouping_policy": {"bdd": "Original full clip group/origin, never the first filename token",
                                  "vkitti": "Whole original scene/source drive; all variants, cameras and frames stay together",
                                  "aihub": "Filename first device token + final YYMMDD token across AIHub13/14/15; whole clips remain together",
                                  "duplicates": "Connected components of original group/origin/path and exact byte/RGB hashes; retain all rows in one split, never silently discard duplicates"},
              "metadata_limits": ["AIHub equipment/date are filename tokens with valid calendar/time syntax, not verified physical vehicle or independent-drive metadata.",
                                  "BDD one-frame-per-clip identity comes from existing preparation metadata; separate sessions/routes are not established.",
                                  "Exact byte/RGB detection does not establish perceptual near-duplicate independence.",
                                  "Hashes are caller-provided export metadata. This function performs no file reads or content rehash; callers must bind inputs to their source export evidence."],
              "counts": {"input": len(rows), "train": len(outputs["train"]), "dev": len(outputs["dev"])},
              "actual_dev_fraction": len(outputs["dev"]) / len(rows),
              "by_dataset": {ds: {"input": total, "train": total-development[ds], "dev": development[ds],
                                       "actual_dev_fraction": development[ds]/total, "original_groups": len(exact_groups[ds]),
                                       "effective_components": effective_groups[ds]} for ds, total in sorted(datasets.items())},
              "total_components": len(components), "dev_components": len(chosen),
              "selected_groups": [{k: v for k, v in component.items() if k != "indices"}
                                  for component in components if component["group_id"] in chosen],
              "duplicate_hash_links": dict(duplicate_links),
              "split_identity_checks": {"rows_preserved": True, "nonempty_each_dataset_train_dev": True,
                                        "group_origin_path_byte_rgb_overlap": 0},
              "initializer_exposure": "These TRAINING rows were already used by completed local models/ancestors. The new dev split is for future development/checkpoint selection; it is not an unseen independent test for those initializers.",
              "official_validation_modified": False, "target_contents_used_for_selection": False,
              "model_or_evaluation_scores_used_for_selection": False, "training_performed": False}
    return outputs["train"], outputs["dev"], report
