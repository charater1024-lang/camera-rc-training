"""Task losses and metrics with explicit masking and physical units."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def task_loss(task: str, prediction: torch.Tensor, target: torch.Tensor,
              segmentation_class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a masked loss; optional class weights affect segmentation only.

    Cross entropy divides by the sum of valid target-class weights, matching
    PyTorch's weighted mean. A larger weight increases a class's relative
    contribution; it is not a calibrated risk or collision probability.
    """
    if segmentation_class_weights is not None and task != "segmentation":
        raise ValueError("segmentation_class_weights require task=segmentation")
    if task == "segmentation":
        weight = segmentation_class_weights
        if weight is not None:
            if not isinstance(weight, torch.Tensor) or weight.ndim != 1 or len(weight) != prediction.shape[1]:
                raise ValueError("segmentation_class_weights must be a tensor with num_classes entries")
            weight = weight.to(device=prediction.device, dtype=prediction.dtype)
            if not torch.all(torch.isfinite(weight) & (weight > 0)):
                raise ValueError("segmentation_class_weights must be finite and positive in the prediction dtype")
        if target.shape != (prediction.shape[0], *prediction.shape[2:]):
            raise ValueError("Segmentation target must have shape [B,H,W]")
        valid = target != 255
        if torch.any(valid & ((target < 0) | (target >= prediction.shape[1]))):
            raise ValueError("Segmentation labels must be 0..num_classes-1, or ignore=255")
        if not torch.any(valid):
            return prediction.sum() * 0.0
        return F.cross_entropy(prediction, target.long(), weight=weight, ignore_index=255)
    if target.shape != prediction.shape:
        raise ValueError(f"{task} target shape {tuple(target.shape)} != prediction shape {tuple(prediction.shape)}")
    if task == "depth":
        valid = torch.isfinite(target) & (target > 0)
        if not torch.any(valid):
            return prediction.sum() * 0.0
        # Log-depth error balances nearby and faraway valid depth measurements.
        return F.smooth_l1_loss(prediction[valid].clamp_min(1e-3).log(), target[valid].log())
    if task == "trajectory":
        if not torch.all(torch.isfinite(target)) or torch.any(target[..., 2] < 0):
            raise ValueError("Trajectory labels must be finite with nonnegative target speed")
        return F.smooth_l1_loss(prediction[..., :2], target[..., :2]) + F.smooth_l1_loss(prediction[..., 2], target[..., 2])
    raise ValueError(f"Unknown task: {task}")


class MetricAccumulator:
    def __init__(self, task: str, num_classes: int = 3):
        self.task = task
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.count = 0
        self.sum_squared = self.sum_relative = 0.0
        self.sum_ade = self.sum_fde = self.sum_speed = 0.0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction, target = prediction.detach().float(), target.detach()
        if self.task == "segmentation":
            labels = prediction.argmax(1)
            valid = target != 255
            encoded = target[valid].long() * self.num_classes + labels[valid]
            self.confusion += torch.bincount(encoded, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes).cpu()
            self.count += int(valid.sum().item())
        elif self.task == "depth":
            valid = torch.isfinite(target) & (target > 0)
            difference = prediction[valid] - target[valid]
            self.sum_squared += float(difference.square().sum().item())
            self.sum_relative += float((difference.abs() / target[valid]).sum().item())
            self.count += int(valid.sum().item())
        elif self.task == "trajectory":
            distances = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
            self.sum_ade += float(distances.mean(dim=1).sum().item())
            self.sum_fde += float(distances[:, -1].sum().item())
            self.sum_speed += float((prediction[..., 2] - target[..., 2]).abs().mean(dim=1).sum().item())
            self.count += int(target.shape[0])
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def compute(self) -> dict:
        if self.count == 0:
            raise ValueError("No valid validation labels found; inspect masks/depth values and the dataset split")
        if self.task == "segmentation":
            matrix = self.confusion.double()
            intersection = matrix.diag()
            union = matrix.sum(0) + matrix.sum(1) - intersection
            present = union > 0
            return {
                "miou": float((intersection[present] / union[present]).mean()),
                "pixel_accuracy": float(intersection.sum() / matrix.sum()),
                "valid_pixels": self.count,
                "class_iou": [float(intersection[i] / union[i]) if present[i] else None
                              for i in range(self.num_classes)],
                # Rows are ground truth; columns are predictions.
                "confusion_matrix": self.confusion.tolist(),
            }
        if self.task == "depth":
            return {"rmse_m": (self.sum_squared / self.count) ** 0.5, "abs_rel": self.sum_relative / self.count, "valid_pixels": self.count}
        return {"ade_m": self.sum_ade / self.count, "fde_m": self.sum_fde / self.count, "speed_mae_mps": self.sum_speed / self.count, "samples": self.count}
