"""Plot verified, public aggregate accuracy only; never read datasets or weights.

The input schema is reports/baseline_accuracy.json. An optional model named
corrected_labels_384x640 is included only under the same frozen conditions.
All axes start at zero; the general nonroad FPR panel has an explicitly labelled
independent range. Percent labels show two decimals; JSON retains full precision.
Requires the existing [preview] extra (matplotlib); no torch/model imports.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re


MODELS = (("tinyunet_384x640", "TinyUNet", "#64748b"),
          ("yolo_v2_384x640", "YOLO26x-sem v2", "#2563eb"),
          ("aihub_384x640", "AIHub fine-tune", "#d97706"))
CANDIDATE = ("corrected_labels_384x640", "Corrected candidate", "#16826b")
METRICS = (("road_iou", "A  Road IoU", "Higher is better"),
           ("road_recall", "B  Road recall", "Higher is better"),
           ("false_road_rate", "C  Nonroad predicted as road", "Lower is better"),
           ("obstacle_fpr", "D  Obstacle pixels predicted as road", "Lower is better"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, description, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), f"Invalid count: {description}")
    return value


def digest(value, description):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Invalid SHA256: {description}")
    return value


def ratio_matches(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual) and 0 <= actual <= 1
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"Stored {name} disagrees with its confusion/labelled-pixel counts")


def matrix_metrics(value):
    require(isinstance(value, list) and len(value) == 2
            and all(isinstance(row, list) and len(row) == 2 for row in value), "Require 2x2 confusion matrix")
    tp, fn, fp, tn = [count(x, "confusion matrix") for row in value for x in row]
    require(tp + fn > 0 and fp + tn > 0 and tp + fp > 0, "Undefined road/nonroad denominator")
    return {"road_iou": tp / (tp + fn + fp), "road_recall": tp / (tp + fn),
            "road_precision": tp / (tp + fp), "false_road_rate": fp / (fp + tn)}, (tp + fn, fp + tn)


def validate_native(model):
    pooled = model["pooled_native"]
    metrics, denominators = matrix_metrics(pooled["confusion_matrix"])
    for name, expected in metrics.items():
        ratio_matches(pooled[name], expected, name)
    images = count(pooled["images"], "pooled images", positive=True)
    matrix = [[0, 0], [0, 0]]
    source_conditions, total_images = {}, 0
    require(isinstance(model.get("native_sources"), dict) and model["native_sources"], "Source breakdown required")
    for name, source in model["native_sources"].items():
        current = source["runtime_threshold_metrics"]
        current_metrics, sizes = matrix_metrics(current["confusion_matrix"])
        for metric, expected in current_metrics.items():
            ratio_matches(current[metric], expected, f"{name}.{metric}")
        n = count(source["images"], "source images", positive=True)
        source_conditions[name] = {"images": n, "gt_road_pixels": sizes[0], "gt_nonroad_pixels": sizes[1]}
        total_images += n
        for i in range(2):
            for j in range(2):
                matrix[i][j] += current["confusion_matrix"][i][j]
    require(total_images == images and matrix == pooled["confusion_matrix"], "Pooled confusion/count differs from source sum")
    for key, value in zip(("road_tp_pixels", "road_fn_pixels", "road_fp_pixels", "road_tn_pixels"),
                          [x for row in matrix for x in row]):
        require(type(pooled[key]) is int and pooled[key] == value, f"Pooled {key} differs from matrix")
    return metrics, {"images": images, "road_gt_pixels": denominators[0], "nonroad_gt_pixels": denominators[1],
                     "sources": source_conditions}


def validate_public(model):
    tracks = model["public_reused_regression"]
    require(set(tracks) == {"obstacle", "anomaly"}, "Reused public40 requires obstacle30 and anomaly10 tracks")
    conditions, rates = {}, {}
    for name, expected_images in (("obstacle", 30), ("anomaly", 10)):
        aggregate = tracks[name]["aggregate"]
        decision = aggregate["decisions"]["runtime_road"]
        require(type(decision["threshold"]) in (int, float) and decision["threshold"] == .65
                and decision["comparison"] == ">=", "Public decision must be road probability >= 0.65")
        require(count(aggregate["images"], "public images") == expected_images
                and count(decision["images_with_labelled_obstacles"], "labelled public images") == expected_images,
                "Unexpected reused public track size")
        denominator = count(decision["labelled_obstacle_pixels"], "labelled obstacle pixels", positive=True)
        numerator = count(decision["obstacle_pixels_predicted_road"], "obstacle false road pixels")
        require(numerator <= denominator, "Obstacle false-road count exceeds labelled pixels")
        rate = numerator / denominator
        ratio_matches(decision["micro_obstacle_as_road_rate"], rate, f"{name}.micro_false_road_rate")
        conditions[name] = {"images": expected_images, "labelled_obstacle_pixels": denominator,
                            "ignored_pixels": count(aggregate["ignored_pixels"], "public ignored pixels"),
                            "usual_pixels_not_used_as_road_gt": count(aggregate["usual_pixels_not_used_as_road_gt"], "public usual pixels")}
        rates[name] = rate
    return rates, conditions


def validate_report(report):
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1, "Expected public aggregate schema 1")
    require(report.get("real_vehicle_validated") is False, "Plot does not support real-vehicle validation claims")
    require(report.get("speed_comparison") == "cancelled; not measured", "Plot requires explicit speed-not-measured status")
    models = report["models"]
    require(isinstance(models, dict), "Model mapping required")
    chosen = [*MODELS, *([CANDIDATE] if CANDIDATE[0] in models else [])]
    unrecognized = {key for key in models if key.endswith("_384x640")} - {item[0] for item in chosen}
    require(not unrecognized, f"Unrecognized 384x640 model keys: {sorted(unrecognized)}")
    values, reference, report_hashes, checkpoint_hashes = [], None, set(), set()
    for key, label, color in chosen:
        require(key in models, f"Required matched model is absent: {key}")
        model = models[key]
        require(model["input_size"] == [384, 640] and model["native_gt_size"] == [192, 320]
                and type(model["runtime_threshold"]) in (int, float) and model["runtime_threshold"] == .65,
                "Required input 384x640, native GT 192x320, threshold 0.65")
        require(model["precision_condition"] == "autocast_fp16_logits_fp32_metrics", "Required matched FP16 autocast / FP32 metrics")
        for field in ("source_report_sha256", "checkpoint_sha256", "evaluation_contract_sha256"):
            digest(model[field], key + "." + field)
        require(model["source_report_sha256"] not in report_hashes and model["checkpoint_sha256"] not in checkpoint_hashes,
                "Distinct model reports and checkpoints required")
        report_hashes.add(model["source_report_sha256"])
        checkpoint_hashes.add(model["checkpoint_sha256"])
        native, native_conditions = validate_native(model)
        public, public_conditions = validate_public(model)
        conditions = {"input_size": model["input_size"], "native_gt_size": model["native_gt_size"],
                      "threshold": model["runtime_threshold"], "precision": model["precision_condition"],
                      "evaluation_contract_sha256": model["evaluation_contract_sha256"],
                      "native": native_conditions, "public": public_conditions}
        require(reference is None or conditions == reference, "Unmatched evaluation contract, source counts, GT denominators, precision or thresholds")
        reference = conditions
        values.append({"key": key, "label": label, "color": color, **native, "obstacle_fpr": public["obstacle"]})
    return values, reference


def _delta_line(before, after, label):
    delta = {key: 100 * (after[key] - before[key]) for key, _, _ in METRICS}
    def direction(value, lower_better=False):
        if value == 0:
            return "unchanged"
        return "better" if ((value < 0) if lower_better else (value > 0)) else "worse"
    return (f"{label}: IoU {delta['road_iou']:+.2f} pp ({direction(delta['road_iou'])}); "
            f"recall {delta['road_recall']:+.2f} pp ({direction(delta['road_recall'])}); "
            f"nonroad FPR {delta['false_road_rate']:+.2f} pp ({direction(delta['false_road_rate'], True)}); "
            f"obstacle FPR {delta['obstacle_fpr']:+.2f} pp ({direction(delta['obstacle_fpr'], True)}).")


def render_figure(values, conditions, source_sha256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.fonttype": "none", "svg.hashsalt": source_sha256,
                                "axes.spines.top": False, "axes.spines.right": False,
                                "axes.spines.left": False, "axes.edgecolor": "#cbd5e1"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.6), facecolor="white")
    fig.subplots_adjust(left=.145, right=.98, top=.815, bottom=.275, wspace=.39, hspace=.70)
    fig.text(.05, .955, "Road segmentation: accuracy trade-offs", fontsize=23, fontweight="bold", color="#0f172a")
    fig.text(.05, .915, "Same model input: 384 x 640  |  Native GT: 192 x 320  |  Road probability >= 0.65", fontsize=13, color="#334155")
    fig.text(.05, .885, "FP16 autocast / FP32 metrics. All axes start at 0; panel C has an independent range. Labels rounded to two decimals; full precision in JSON.",
             fontsize=10.5, color="#475569")
    for ax, (metric, title, direction) in zip(axes.flat, METRICS):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
        maximum = max(5, math.ceil(max(value[metric] for value in values) * 100 / 5) * 5) if metric == "false_road_rate" else 100
        ticks = [maximum * i / 5 for i in range(6)] if metric == "false_road_rate" else [0, 25, 50, 75, 100]
        ax.set_xlim(0, maximum)
        ax.set_xticks(ticks, [f"{value:g}%" for value in ticks])
        ax.tick_params(axis="y", length=0, pad=12)
        ax.tick_params(axis="x", colors="#64748b", length=3)
        ax.set_yticks(range(len(values)), [value["label"] for value in values])
        ax.set_ylim(len(values) - .45, -.55)
        ax.set_title(title, loc="left", pad=34, fontsize=13.5, fontweight="bold", color="#0f172a")
        ax.text(0, 1.055, direction + (f"  |  Independent axis: 0-{maximum:g}%" if metric == "false_road_rate" else "  |  Axis: 0-100%"),
                transform=ax.transAxes, fontsize=11, color="#475569")
        for index, value in enumerate(values):
            percent = value[metric] * 100
            ax.barh(index, percent, height=.52, color=value["color"], zorder=3)
            # Keep labels inside the panel range even for near-perfect scores.
            inside = percent >= .79 * maximum
            offset = maximum * .012
            ax.text(percent - offset if inside else percent + offset, index, f"{percent:.2f}%",
                    ha="right" if inside else "left", va="center", fontsize=11,
                    color="white" if inside else "#0f172a", fontweight="bold", zorder=4)
        if metric == "obstacle_fpr":
            n = conditions["public"]["obstacle"]
            note = f"Micro pixel rate: {n['images']} images / {n['labelled_obstacle_pixels']:,} obstacle GT pixels.\nOriginal-resolution GT; not the 192 x 320 native grid."
        elif metric == "false_road_rate":
            note = f"FP / (FP + TN); {conditions['native']['nonroad_gt_pixels']:,} labelled nonroad GT pixels."
        else:
            note = f"Pooled pixel counts across {conditions['native']['images']:,} frames; source sizes are unequal."
        ax.text(0, -.22, note, transform=ax.transAxes, va="top", fontsize=9.2, color="#475569")
    fig.text(.05, .168, _delta_line(values[1], values[2], "YOLO v2 -> AIHub"), fontsize=10.5, color="#0f172a", fontweight="bold")
    if len(values) == 4:
        fig.text(.05, .139, _delta_line(values[2], values[3], "AIHub -> corrected candidate"), fontsize=10.5, color="#0f172a")
    fig.text(.05, .103, "Public40 (30 obstacle + 10 anomaly images) is reused development regression, NOT a blind test; panel D shows obstacle30 only.",
             fontsize=10.5, color="#9a3412")
    fig.text(.05, .074, "Speed comparison cancelled / not measured. Descriptive pixel scores are not collision probabilities or real-vehicle safety evidence.",
             fontsize=10.5, color="#475569")
    fig.text(.05, .042, "Native scores use one frozen reporting cohort previously used for checkpoint selection. No independent generalization claim. "
             f"Source JSON SHA256: {source_sha256[:16]}...", fontsize=9.3, color="#64748b")
    return fig


def generate(input_path, output_path, preview_png=None):
    source, output = Path(input_path).resolve(strict=True), Path(output_path).resolve()
    require(source.suffix.lower() == ".json" and output.suffix.lower() == ".svg", "Use JSON input and SVG output")
    require(source != output, "Do not overwrite the source report")
    payload = source.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    values, conditions = validate_report(json.loads(payload.decode("utf-8-sig")))
    preview = Path(preview_png).resolve() if preview_png else None
    if preview:
        require(preview.suffix.lower() == ".png" and not preview.is_relative_to(Path(__file__).resolve().parents[1]),
                "Visual-review PNG must be outside the public source tree")
    figure = render_figure(values, conditions, sha)
    try:
        svg = BytesIO()
        figure.savefig(svg, format="svg", metadata={"Title": "Matched road segmentation accuracy trade-offs",
                        "Creator": "plot_accuracy_tradeoff.py / matplotlib Agg",
                        "Description": f"Public aggregate source SHA256={sha}. Same frozen evaluation conditions and denominators verified. No inference performed.",
                        "Date": None})
        png = BytesIO()
        if preview:
            figure.savefig(png, format="png", dpi=140, metadata={"SourceJSONSHA256": sha})
        require(hashlib.sha256(source.read_bytes()).hexdigest() == sha, "Source report changed while plotting")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(svg.getvalue())
        if preview:
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(png.getvalue())
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return {"source_json_sha256": sha, "models": [value["key"] for value in values],
            "conditions": conditions, "values": [{key: value[key] for key in ("key", *[item[0] for item in METRICS])} for value in values]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Public aggregate JSON only")
    parser.add_argument("--output", type=Path, required=True, help="Reproducible .svg output")
    parser.add_argument("--preview-png", type=Path, help="Optional local review PNG outside public source tree")
    args = parser.parse_args(argv)
    result = generate(args.input, args.output, args.preview_png)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
