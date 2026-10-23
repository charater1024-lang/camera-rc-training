import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from camera_rc.config import load_config
from camera_rc.data import ManifestDataset, read_manifest, validate_splits
from camera_rc.rc_targets import prepare_rc


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "한글 image.png"
        Image.new("RGB", (32, 32)).save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def row(self, target):
        return {"image": str(self.image), "target": target, "group_id": "drive1"}

    def test_group_leakage_even_with_distinct_images(self):
        other = dict(self.row("unused"), image=str(self.root / "other.png"))
        with self.assertRaisesRegex(ValueError, "scene leakage"):
            validate_splits([self.row("unused")], [other])

    def test_depth_units_and_invalid_pixels(self):
        target = self.root / "depth.png"
        pixels = np.full((32, 32), 2560, np.uint16)
        pixels[0, 0] = 0
        Image.fromarray(pixels).save(target)
        ds = ManifestDataset([self.row(str(target))], {"task": "depth", "data": {"image_size": [32, 32]}})
        tensor = ds[0]["target"]
        self.assertEqual(float(tensor[0, 1, 1]), 10.0)
        self.assertEqual(float(tensor[0, 0, 0]), 0.0)

    def test_mask_rgb_visualization_rejected(self):
        ds = ManifestDataset([self.row(str(self.image))], {"task": "segmentation", "data": {"image_size": [32, 32]}})
        with self.assertRaisesRegex(ValueError, "single-channel"):
            ds[0]

    def test_offsets_cannot_be_silently_mixed(self):
        row = self.row([[1, 0, 1]])
        row.update(state=[1, 0], future_offsets_s=[2.0])
        ds = ManifestDataset([row], {"task": "trajectory", "model": {"horizon": 1},
                                    "data": {"image_size": [32, 32], "future_offsets_s": [0.5]}})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ds[0]

    def test_flat_detection_config_paths(self):
        p = self.root / "config.json"
        p.write_text(json.dumps({"task": "detection", "data": "data.yaml", "project": "runs"}))
        cfg = load_config(p)
        self.assertEqual(cfg["data"], str(self.root / "data.yaml"))

    def test_rc_world_to_ego_rotation_and_gap(self):
        csv_path = self.root / "poses.csv"
        fields = ["image", "group_id", "timestamp_s", "x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rps"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            for i in range(5):
                writer.writerow(dict(zip(fields, [str(self.image), "north", i * 0.1, 0, i * 0.1, math.pi / 2, 1, 0])))
        out = self.root / "trajectory.jsonl"
        report = prepare_rc(csv_path, out, offsets=[0.2], max_gap=0.15)
        self.assertGreater(report["samples"], 0)
        row = json.loads(out.read_text().splitlines()[0])
        self.assertAlmostEqual(row["target"][0][0], 0.2)
        self.assertAlmostEqual(row["target"][0][1], 0.0)
        self.assertEqual(row["state"], [1, 0])
        with self.assertRaisesRegex(ValueError, "No complete"):
            prepare_rc(csv_path, self.root / "gaps.jsonl", offsets=[0.2], max_gap=0.05)


if __name__ == "__main__":
    unittest.main()
