import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from camera_rc.prepare import (make_paired_manifest, prepare_bdd_detection, split_manifest,
                               prepare_bdd_drivable, prepare_kitti_depth)
from camera_rc.detection import run_detection


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="camera-rc-한글-")
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.val = self.root / "val"
        self.train.mkdir()
        self.val.mkdir()
        Image.new("RGB", (100, 50)).save(self.train / "train.jpg")
        Image.new("RGB", (100, 50)).save(self.val / "val.jpg")
        self.train_json, self.val_json = self.root / "train.json", self.root / "val.json"
        self.write_labels()

    def tearDown(self):
        self.temp.cleanup()

    def write_labels(self, category="person", box=None):
        if box is None:
            box = {"x1": -5, "y1": 0, "x2": 49, "y2": 49}
        self.train_json.write_text(json.dumps([{"name": "train.jpg", "labels": [
            {"category": category, "box2d": box},
            {"category": "lane", "poly2d": []}]}]), encoding="utf-8")
        self.val_json.write_text(json.dumps([{"name": "val.jpg", "labels": []}]), encoding="utf-8")

    def convert(self, **kwargs):
        return prepare_bdd_detection(self.train_json, self.val_json, self.train,
                                     self.val, self.root / "prepared", **kwargs)

    def test_conversion_alias_clip_normalize_and_true_negative(self):
        report = self.convert()
        text = (self.root / "prepared/labels/train/train.txt").read_text()
        self.assertEqual(text.split(), ["0", "0.25000000", "0.50000000", "0.50000000", "1.00000000"])
        self.assertEqual((self.root / "prepared/labels/val/val.txt").read_text(), "")
        self.assertEqual(report["train"]["clipped_boxes"], 1)
        self.assertEqual(report["train"]["skipped_non_box:lane"], 1)
        self.assertEqual(report["val"]["negative_images"], 1)
        yaml = json.loads((self.root / "prepared/dataset.yaml").read_text())
        self.assertEqual(yaml["names"][0], "pedestrian")

    def test_unknown_category_and_bad_coordinates_refused_before_output(self):
        self.write_labels(category="trailer")
        with self.assertRaisesRegex(ValueError, "Unknown detection"):
            self.convert()
        self.assertFalse((self.root / "prepared").exists())
        self.write_labels(box={"x1": 10, "y1": 10, "x2": 3, "y2": 30})
        with self.assertRaisesRegex(ValueError, "Reversed box"):
            self.convert()

    def test_missing_labels_is_not_a_negative(self):
        self.train_json.write_text('[{"name":"train.jpg"}]')
        with self.assertRaisesRegex(ValueError, "Missing labels"):
            self.convert()

    def test_missing_image_refused(self):
        (self.train / "train.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            self.convert()

    def test_train_val_stem_leakage_refused(self):
        Image.new("RGB", (100, 50)).save(self.val / "train.jpg")
        self.val_json.write_text('[{"name":"train.jpg","labels":[]}]')
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.convert()

    def test_output_refused_and_explicit_overwrite_preserves_backup(self):
        self.convert()
        marker = self.root / "prepared/custom-note.txt"
        marker.write_text("keep me")
        with self.assertRaises(FileExistsError):
            self.convert()
        report = self.convert(overwrite=True)
        self.assertEqual((Path(report["backup"]) / marker.name).read_text(), "keep me")

    def test_relative_paths_cannot_escape_image_root(self):
        self.train_json.write_text('[{"name":"../val/val.jpg","labels":[]}]')
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.convert()

    def test_split_is_grouped_reproducible_and_resolves_paths(self):
        rows = []
        for group in range(5):
            for frame in range(2):
                image = self.root / f"g{group}-{frame}.jpg"
                Image.new("RGB", (8, 8)).save(image)
                rows.append({"image": image.name, "group_id": f"g{group}", "target": [[1, 2, 3]]})
        manifest = self.root / "all.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        first = split_manifest(manifest, self.root / "split1", seed=17)
        second = split_manifest(manifest, self.root / "split2", seed=17)
        self.assertEqual(first, second)
        self.assertFalse(set(first["train_groups"]) & set(first["val_groups"]))
        output_rows = [json.loads(row) for row in (self.root / "split1/train.jsonl").read_text().splitlines()]
        self.assertTrue(all(Path(row["image"]).is_absolute() for row in output_rows))
        self.assertEqual(output_rows[0]["target"], [[1, 2, 3]])

    def test_split_needs_explicit_groups(self):
        manifest = self.root / "all.jsonl"
        manifest.write_text(json.dumps({"image": "train/train.jpg", "target": [[0, 0, 0]]}))
        with self.assertRaisesRegex(ValueError, "group_id"):
            split_manifest(manifest, self.root / "split")

    def test_pairing_requires_sequence_capture_and_all_targets(self):
        targets = self.root / "targets"
        targets.mkdir()
        Image.new("L", (100, 50), 2).save(targets / "train.png")
        output = self.root / "pairs.jsonl"
        with self.assertRaisesRegex(ValueError, "group_pattern"):
            make_paired_manifest(self.train, targets, output, "segmentation")
        result = make_paired_manifest(self.train, targets, output, "segmentation", r"(train)")
        self.assertEqual(result["samples"], 1)
        row = json.loads(output.read_text())
        self.assertEqual(row["group_id"], "train")
        self.assertEqual(Path(row["target"]).name, "train.png")


class DetectionGuardTests(unittest.TestCase):
    def test_invalid_local_data_never_loads_model(self):
        with self.assertRaises(FileNotFoundError):
            run_detection({"data": "a-certainly-absent-dataset.yaml"})

    def test_resume_requires_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "Resume"):
            run_detection({"resume": True})

    def test_validation_and_export_require_trained_checkpoint(self):
        for mode in ("val", "export"):
            with self.assertRaisesRegex(ValueError, "trained"):
                run_detection({}, mode=mode)


class SourcePairedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="camera-rc-pairs-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def bdd_fixture(self, color=False):
        dirs = []
        for split in ("train", "val"):
            image_dir, mask_dir = self.root / split / "images", self.root / split / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir()
            Image.new("RGB", (8, 8)).save(image_dir / f"{split}-clip.jpg")
            Image.new("RGB" if color else "L", (8, 8), 2).save(mask_dir / f"{split}-clip_drivable_id.png")
            dirs.append((image_dir, mask_dir))
        return dirs

    def test_bdd_drivable_suffix_and_ids(self):
        dirs = self.bdd_fixture()
        report = prepare_bdd_drivable(dirs[0][0], dirs[1][0], dirs[0][1], dirs[1][1], self.root / "out")
        self.assertEqual(report["train"]["samples"], 1)
        row = json.loads((self.root / "out/train.jsonl").read_text())
        self.assertEqual(row["group_id"], "train-clip")
        self.assertTrue(row["target"].endswith("train-clip_drivable_id.png"))

    def test_bdd_visualization_rejected(self):
        dirs = self.bdd_fixture(color=True)
        with self.assertRaisesRegex(ValueError, "color visualization"):
            prepare_bdd_drivable(dirs[0][0], dirs[1][0], dirs[0][1], dirs[1][1], self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def kitti_fixture(self, overlap=False):
        raw, depth = self.root / "raw", self.root / "depth"
        for number, split in enumerate(("train", "val"), 1):
            drive = f"2011_09_26_drive_{1 if overlap else number:04d}_sync"
            for camera in ("image_02", "image_03"):
                image = raw / "2011_09_26" / drive / camera / "data/0000000000.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(image)
                target = depth / split / drive / "proj_depth/groundtruth" / camera / "0000000000.png"
                target.parent.mkdir(parents=True)
                Image.new("I;16", (8, 8), 256).save(target)
        return raw, depth

    def test_kitti_exact_pairing_and_stereo_grouping(self):
        raw, depth = self.kitti_fixture()
        report = prepare_kitti_depth(raw, depth, self.root / "out")
        self.assertEqual(report["train"], {"samples": 2, "groups": 1})
        rows = [json.loads(line) for line in (self.root / "out/train.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertNotEqual(rows[0]["image"], rows[1]["image"])

    def test_kitti_drive_overlap_rejected(self):
        raw, depth = self.kitti_fixture(overlap=True)
        with self.assertRaisesRegex(ValueError, "group IDs overlap"):
            prepare_kitti_depth(raw, depth, self.root / "out")

    def test_kitti_ambiguous_raw_layout_rejected(self):
        raw, depth = self.kitti_fixture()
        duplicate = raw / "2011_09_26_drive_0001_sync/image_02/data/0000000000.png"
        duplicate.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(duplicate)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            prepare_kitti_depth(raw, depth, self.root / "out")

    def test_kitti_missing_rgb_rejected(self):
        raw, depth = self.kitti_fixture()
        next(raw.rglob("*.png")).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing raw RGB"):
            prepare_kitti_depth(raw, depth, self.root / "out")


if __name__ == "__main__":
    unittest.main()
