"""Joint multi-camera temporal waypoint research baseline, without actuation.

One shared image encoder supplies tokens for every view and past frame. Rays,
camera origins and relative capture ages locate tokens in the current anchor
frame (x forward, y left, z up). This is not a BEV or 3D occupancy model. Camera
coordinates are optical (x right, y down, z forward); images must be rectified.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_activations

from ..models import ConvBlock


MODEL_TYPE = "joint_camera_temporal_waypoints"


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


class CrossAttentionBlock(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.SiLU(),
                                 nn.Linear(dimension * 2, dimension))

    def forward(self, queries, memory):
        context = self.memory_norm(memory)
        attended, _ = self.attention(self.query_norm(queries), context, context, need_weights=False)
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class JointCameraTemporalPolicy(nn.Module):
    """Predict [B,K,3] forward metres, left metres, nonnegative speed m/s.

    Future offsets are fixed semantic metadata, not future observed inputs.
    Every supplied view is required at each past time. Input order has no camera
    identity embedding: permuting images and their geometry together preserves
    the result. The current anchor must remain unchanged under that permutation.
    """
    def __init__(self, *, width=16, embed_dim=64, token_grid=(3, 5), horizon=5,
                 encoder_chunk_size=1, checkpoint_encoder=False, heads=4,
                 max_history_s=1., expected_views=None, expected_history=None,
                 image_size=None, future_dt_s=None):
        super().__init__()
        self.width = _integer(width, "model.width", 4, 64)
        self.embed_dim = _integer(embed_dim, "model.embed_dim", 16, 256)
        self.horizon = _integer(horizon, "model.horizon", 1, 100)
        heads = _integer(heads, "model.heads", 1, 16)
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by attention heads")
        if not isinstance(token_grid, (tuple, list)) or len(token_grid) != 2:
            raise ValueError("model.token_grid requires height,width")
        self.token_grid = tuple(_integer(v, "token_grid", 1, 16) for v in token_grid)
        self.encoder_chunk_size = _integer(encoder_chunk_size, "encoder_chunk_size", 1, 64)
        if not isinstance(checkpoint_encoder, bool):
            raise ValueError("checkpoint_encoder must be boolean")
        self.checkpoint_encoder = checkpoint_encoder
        if isinstance(max_history_s, bool) or not isinstance(max_history_s, (int, float)) or not math.isfinite(max_history_s) or max_history_s <= 0:
            raise ValueError("max_history_s must be finite and positive")
        self.max_history_s = float(max_history_s)
        self.expected_views = None if expected_views is None else _integer(expected_views, "expected_views", 1, 8)
        self.expected_history = None if expected_history is None else _integer(expected_history, "expected_history", 2, 4)
        if image_size is not None:
            if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
                raise ValueError("image_size requires height,width")
            image_size = tuple(_integer(v, "image_size", 16, 4096) for v in image_size)
        self.image_size = image_size
        if future_dt_s is None:
            future_dt_s = [.5 * (i + 1) for i in range(horizon)]
        if (not isinstance(future_dt_s, (list, tuple)) or len(future_dt_s) != horizon
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in future_dt_s)
                or any(b <= a for a, b in zip(future_dt_s, future_dt_s[1:]))):
            raise ValueError("future_dt_s must contain horizon strictly increasing positive seconds")
        self.register_buffer("future_dt_s", torch.tensor(future_dt_s, dtype=torch.float32))
        # Keep names and shapes identical to the native TinyUNet encoder.
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.visual_projection = nn.Linear(width * 4, embed_dim)
        self.geometry_embedding = nn.Sequential(nn.Linear(7, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.state_embedding = nn.Sequential(nn.Linear(2, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.future_embedding = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.queries = nn.Parameter(torch.randn(horizon, embed_dim) * .02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(embed_dim, heads) for _ in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
                                    nn.SiLU(), nn.Linear(embed_dim, 3))

    def _validate(self, images, intrinsics, camera_to_anchor, time_offsets, state):
        tensors = {"images": images, "intrinsics": intrinsics, "camera_to_anchor": camera_to_anchor,
                   "time_offsets": time_offsets, "state": state}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating tensor")
            if value.device != images.device or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and share the image device")
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError("images must have shape [B,T,V,3,H,W]")
        b, t, v, _, h, w = images.shape
        if b < 1 or not 2 <= t <= 4 or not 1 <= v <= 8 or min(h, w) < 16:
            raise ValueError("Requires B>=1, history T=2..4, views V=1..8 and image size>=16")
        if self.expected_views is not None and v != self.expected_views:
            raise ValueError("Every configured camera view is mandatory")
        if self.expected_history is not None and t != self.expected_history:
            raise ValueError("History count differs from configured expected_history")
        if self.image_size is not None and (h, w) != self.image_size:
            raise ValueError("Image size differs from the configured resized calibration resolution")
        if intrinsics.shape != (b, t, v, 3, 3) or camera_to_anchor.shape != (b, t, v, 4, 4):
            raise ValueError("Calibration shape must match every batch, time and view")
        if time_offsets.shape != (b, t) or state.shape != (b, 2):
            raise ValueError("time_offsets must be [B,T] and current state [B,2]")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("RGB images must be in [0,1]")
        if (time_offsets > 0).any() or (time_offsets < -self.max_history_s - 1e-6).any():
            raise ValueError("Future or stale camera timestamps are forbidden")
        if not torch.allclose(time_offsets[:, -1], torch.zeros_like(time_offsets[:, -1]), atol=1e-7, rtol=0):
            raise ValueError("History must end at the current anchor time zero")
        if not (torch.diff(time_offsets, dim=1) > 0).all():
            raise ValueError("History times must be strictly increasing")
        if (state[:, 0] < 0).any():
            raise ValueError("Current speed must be nonnegative; reverse operation is unsupported")
        k = intrinsics.float()
        expected_row = k.new_tensor([0, 0, 1]).expand_as(k[..., 2, :])
        if (not torch.allclose(k[..., 2, :], expected_row, atol=1e-5, rtol=0)
                or (k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any()
                or (k[..., 1, 0].abs() > 1e-5).any()):
            raise ValueError("Intrinsics must be a nonsingular rectified pinhole matrix with positive focal lengths")
        transform = camera_to_anchor.float()
        if not torch.allclose(transform[..., 3, :], transform.new_tensor([0, 0, 0, 1]).expand_as(transform[..., 3, :]), atol=1e-5, rtol=0):
            raise ValueError("camera_to_anchor must be homogeneous SE(3)")
        rotation = transform[..., :3, :3]
        eye = torch.eye(3, device=rotation.device).expand_as(rotation)
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            orthogonality = rotation.transpose(-2, -1) @ rotation
            determinant = torch.linalg.det(rotation)
        if (not torch.allclose(orthogonality, eye, atol=1e-3, rtol=0)
                or not torch.allclose(determinant, torch.ones_like(rotation[..., 0, 0]), atol=1e-3, rtol=0)):
            raise ValueError("camera_to_anchor rotation must be orthonormal with determinant +1")
        return b, t, v, h, w

    def _encode(self, images):
        x = self.enc1(images * 2 - 1)
        x = self.enc2(F.avg_pool2d(x, 2))
        x = self.enc3(F.avg_pool2d(x, 2))
        return F.adaptive_avg_pool2d(x, self.token_grid).flatten(2).transpose(1, 2)

    def geometry_tokens(self, intrinsics, camera_to_anchor, time_offsets, height, width):
        """Unit rays, metric origins and relative age for each pooled token."""
        rows, columns = self.token_grid
        y = (torch.arange(rows, device=intrinsics.device, dtype=torch.float32) + .5) * height / rows - .5
        x = (torch.arange(columns, device=intrinsics.device, dtype=torch.float32) + .5) * width / columns - .5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        pixels = torch.stack((xx.flatten(), yy.flatten(), torch.ones_like(xx.flatten())))
        pixels = pixels.expand(*intrinsics.shape[:3], 3, rows * columns)
        # Calibration arithmetic stays in float32 even inside mixed-precision training.
        with torch.autocast(device_type=intrinsics.device.type, enabled=False):
            rays = torch.linalg.solve(intrinsics.float(), pixels)
            directions = (camera_to_anchor[..., :3, :3].float() @ rays).transpose(-2, -1)
            directions = F.normalize(directions, dim=-1)
            origins = camera_to_anchor[..., :3, 3].float().unsqueeze(-2).expand_as(directions)
            ages = time_offsets.float()[:, :, None, None, None].expand(*directions.shape[:-1], 1)
            return torch.cat((directions, origins, ages), dim=-1)

    def forward(self, images, intrinsics, camera_to_anchor, time_offsets, state):
        b, t, v, height, width = self._validate(images, intrinsics, camera_to_anchor, time_offsets, state)
        flat = images.float().reshape(b * t * v, 3, height, width)
        chunks = []
        for image_chunk in flat.split(self.encoder_chunk_size):
            encoded = (checkpoint_activations(self._encode, image_chunk, use_reentrant=False)
                       if self.training and self.checkpoint_encoder and torch.is_grad_enabled()
                       else self._encode(image_chunk))
            chunks.append(encoded)
        visual = self.visual_projection(torch.cat(chunks)).reshape(b, t, v, -1, self.embed_dim)
        geometry = self.geometry_tokens(intrinsics, camera_to_anchor, time_offsets, height, width)
        memory = (visual + self.geometry_embedding(geometry)).reshape(b, -1, self.embed_dim)
        queries = (self.queries + self.future_embedding(self.future_dt_s[:, None])).unsqueeze(0)
        queries = queries.expand(b, -1, -1) + self.state_embedding(state.float()).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, memory)
        prediction = self.output(queries)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite joint waypoint prediction")
        return torch.cat((prediction[..., :2], F.softplus(prediction[..., 2:3])), dim=-1)


def build_joint_policy(config):
    if not isinstance(config, dict):
        raise ValueError("Joint policy configuration must be a dictionary")
    options, data = config.get("model", {}), config.get("data", {})
    names = data.get("camera_names")
    expected_views = data.get("expected_views", options.get("expected_views"))
    if names is not None:
        if not isinstance(names, list) or not 1 <= len(names) <= 8 or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("camera_names must contain one to eight unique nonempty names")
        if expected_views is not None and expected_views != len(names):
            raise ValueError("camera_names and expected_views disagree")
        expected_views = len(names)
    return JointCameraTemporalPolicy(
        width=options.get("width", 16), embed_dim=options.get("embed_dim", 64),
        token_grid=options.get("token_grid", [3, 5]), horizon=options.get("horizon", 5),
        encoder_chunk_size=options.get("encoder_chunk_size", 1), checkpoint_encoder=options.get("checkpoint_encoder", False),
        heads=options.get("heads", 4), max_history_s=data.get("max_history_s", options.get("max_history_s", 1.)),
        expected_views=expected_views, expected_history=data.get("expected_history"),
        image_size=data.get("image_size"), future_dt_s=data.get("future_dt_s"))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def init_encoder(model, checkpoint, allow_diagnostic=False):
    """Copy only matching native segmentation encoder tensors, transactionally."""
    path = Path(checkpoint).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stored = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(stored, dict) or stored.get("format_version") != 1:
        raise ValueError("Expected a native segmentation checkpoint format_version=1")
    config = stored.get("config", {})
    if config.get("task") != "segmentation":
        raise ValueError("Only a native segmentation encoder may initialize this model")
    training = config.get("training", {})
    diagnostic = bool(stored.get("diagnostic") or stored.get("diagnostic_only") or config.get("diagnostic")
                      or config.get("diagnostic_only") or training.get("max_train_steps") or training.get("max_val_steps")
                      or stored.get("epoch_complete") is False)
    if diagnostic and not allow_diagnostic:
        raise ValueError("Diagnostic or incomplete segmentation initialization requires allow_diagnostic=True")
    current = model.state_dict()
    names = sorted(name for name in current if name.startswith(("enc1.", "enc2.", "enc3.")))
    source = stored.get("model")
    if not isinstance(source, dict) or config.get("model", {}).get("width", 16) != model.width:
        raise ValueError("Segmentation encoder width or model state is incompatible")
    for name in names:
        value = source.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                or value.dtype != current[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Missing, incompatible or nonfinite encoder tensor: {name}")
    with torch.no_grad():
        for name in names:
            current[name].copy_(source[name].to(current[name].device))
    return {"checkpoint": str(path), "sha256": digest.hexdigest(), "matched_tensors": len(names),
            "matched_names": names, "matched_values": sum(current[name].numel() for name in names),
            "source_diagnostic": diagnostic, "scope": "enc1/enc2/enc3 only; waypoint attention and output are newly initialized"}
