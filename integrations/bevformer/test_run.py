"""Offline tests for config/path/launch handling; no torch or nuScenes required."""

import json
from pathlib import Path
import runpy
import tempfile
import unittest

from integrations.bevformer.run import CAMERAS, VARIANTS, make_plan, parser, write_plan
from integrations.bevformer.prepare_infos import make_command, parser as converter_parser


class PreparationTests(unittest.TestCase):
    def setUp(self):
        # Keep test-owned temporary paths within the checkout, including in
        # restricted Windows sandboxes where the system temp root is not usable.
        self.temp = tempfile.TemporaryDirectory(
            prefix="temp_한글 경로_", dir=str(Path(__file__).resolve().parent)
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream with space"
        self.data = self.root / "data's 한글"
        for base, trainer in VARIANTS.values():
            self.write(self.upstream / base, "# upstream placeholder\n")
            self.write(self.upstream / trainer, "# upstream placeholder\n")
        for path in ("projects/configs/datasets/custom_nus-3d.py",
                     "projects/configs/_base_/default_runtime.py",
                     "projects/mmdet3d_plugin/__init__.py", "LICENSE"):
            self.write(self.upstream / path, "# placeholder\n")
        for camera in CAMERAS:
            (self.data / "samples" / camera).mkdir(parents=True)
        for name in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sample_annotation"):
            self.write(self.data / "v1.0-mini" / (name + ".json"), "[]")
        for split in ("train", "val"):
            self.write(self.data / ("nuscenes_infos_temporal_" + split + ".pkl"), "placeholder; not loaded")

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def arguments(self, *extra):
        return parser().parse_args([
            "--upstream", str(self.upstream), "--data-root", str(self.data),
            "--dataset-version", "v1.0-mini", "--output-dir", str(self.root / "출력 ' 폴더"),
            "--lr", "0.000025", "--allow-unpinned", "--random-init",
        ] + list(extra))

    def test_unicode_paths_single_gpu_and_explicit_lr(self):
        args = self.arguments()
        plan = make_plan(args)
        write_plan(args, plan)
        config = runpy.run_path(plan["config"])
        self.assertEqual(config["data"]["train"]["ann_file"], plan["train_info"])
        self.assertEqual(config["optimizer"]["lr"], 0.000025)
        self.assertEqual(config["data"]["samples_per_gpu"], 1)
        self.assertEqual(config["data"]["workers_per_gpu"], 2)
        self.assertEqual(config["runner"]["max_epochs"], 24)
        self.assertIsNone(config["model"]["pretrained"])
        self.assertIn("--nproc_per_node=1", plan["argv"])
        self.assertNotIn("--autoscale-lr", plan["argv"])
        self.assertEqual(plan["argv"][-4:-2], ["--work-dir", plan["work_dir"]])
        stored = json.loads(Path(plan["config"]).with_name("launch_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["argv"], plan["argv"])
        # Repeating preparation with unchanged settings is safe.
        write_plan(args, plan)

    def test_no_silent_override(self):
        args = self.arguments()
        plan = make_plan(args)
        write_plan(args, plan)
        args.epochs = 2
        with self.assertRaisesRegex(ValueError, "output differs"):
            write_plan(args, make_plan(args))
        args.overwrite = True
        write_plan(args, make_plan(args))

    def test_missing_annotations_and_split_leakage(self):
        args = self.arguments()
        args.train_info = self.root / "missing.pkl"
        with self.assertRaisesRegex(ValueError, "training annotations"):
            make_plan(args)
        args.train_info = self.data / "nuscenes_infos_temporal_val.pkl"
        with self.assertRaisesRegex(ValueError, "different files"):
            make_plan(args)

    def test_missing_camera(self):
        (self.data / "samples" / CAMERAS[0]).rmdir()
        with self.assertRaisesRegex(ValueError, "camera directory"):
            make_plan(self.arguments())

    def test_pin_required(self):
        args = self.arguments()
        args.allow_unpinned = False
        with self.assertRaisesRegex(ValueError, "clean checkout"):
            make_plan(args)

    def test_fp16_uses_its_own_upstream_entrypoint(self):
        args = self.arguments("--precision", "fp16", "--epochs", "1", "--workers", "0")
        plan = make_plan(args)
        write_plan(args, plan)
        self.assertIn(str(self.upstream / "tools/fp16/train.py"), plan["argv"])
        self.assertIn("bevformer_tiny_fp16.py", plan["base_config"])
        config = runpy.run_path(plan["config"])
        # Runner type is inherited so FP16's EpochBasedRunner_video survives.
        self.assertNotIn("type", config["runner"])
        self.assertEqual(config["data"]["workers_per_gpu"], 0)

    def test_local_initialization_and_resume(self):
        weights = self.root / "weights 한글.pth"
        self.write(weights, "placeholder")
        args = self.arguments()
        args.random_init = False
        args.backbone_weights = weights
        plan = make_plan(args)
        write_plan(args, plan)
        self.assertEqual(runpy.run_path(plan["config"])["model"]["pretrained"], {"img": str(weights)})
        args.backbone_weights = None
        args.resume_from = weights
        plan = make_plan(args)
        self.assertEqual(plan["argv"][-2:], ["--resume-from", str(weights)])

    def test_conversion_command_preserves_paths_and_split(self):
        self.write(self.upstream / "tools/data_converter/nuscenes_converter.py", "# placeholder\n")
        (self.data / "samples/LIDAR_TOP").mkdir()
        canbus_parent = self.root / "CAN bus 한글"
        (canbus_parent / "can_bus").mkdir(parents=True)
        output = self.root / "new infos"
        args = converter_parser().parse_args([
            "--upstream", str(self.upstream), "--data-root", str(self.data),
            "--canbus-root", str(canbus_parent), "--version", "v1.0-mini",
            "--output-dir", str(output), "--allow-unpinned",
        ])
        cwd, destination, command = make_command(args)
        self.assertEqual(cwd, self.upstream)
        self.assertEqual(destination, output)
        self.assertEqual(command[-4:], [str(self.data), str(output), str(canbus_parent), "v1.0-mini"])
        self.assertFalse(output.exists())
        self.write(output / "nuscenes_infos_temporal_train.pkl", "existing annotation")
        with self.assertRaisesRegex(ValueError, "already exist"):
            make_command(args)

    def test_lr_is_required_and_must_be_finite_positive(self):
        # Argument validation is exercised without invoking a model/environment.
        from integrations.bevformer.run import positive_float
        import argparse
        for value in ("nan", "inf", "0", "-0.1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_float(value)


if __name__ == "__main__":
    unittest.main()
