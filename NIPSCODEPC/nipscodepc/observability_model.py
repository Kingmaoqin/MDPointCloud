#!/usr/bin/env python3
"""
Learned Observability Field for SCSU (Support-Calibrated Structured Update).

V2: Hybrid mode — MLP learns a correction/blending of hand-crafted support.
    Features now include S_handcrafted (14 dims total).
    Output blended with hand-crafted support for stability.
"""

import numpy as np
import torch
import torch.nn as nn


class ObservabilityMLP(nn.Module):
    """
    Per-vertex MLP that predicts a multiplicative gate on hand-crafted support.

    Output: gate in [gate_lo, gate_hi] (default [0.5, 1.5])
    Final support: S_final = clamp(S_hc * gate, 0, 1)

    This preserves the hand-crafted support's structure (0-1 dynamic range)
    while learning vertex-specific corrections.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 gate_lo: float = 0.5, gate_hi: float = 1.5):
        super().__init__()
        self.gate_lo = gate_lo
        self.gate_hi = gate_hi
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor,
                support_hc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, d] per-vertex features
            support_hc: [N] hand-crafted support from Kernel B
        Returns:
            S_final: [N] gated support in [0, 1]
        """
        raw = self.net(features).squeeze(-1)  # [N]
        # Sigmoid -> scale to [gate_lo, gate_hi]
        gate = self.gate_lo + (self.gate_hi - self.gate_lo) * torch.sigmoid(raw)
        return (support_hc * gate).clamp(0.0, 1.0)


def build_frame_blend_stats(
    residuals: torch.Tensor,
    s_obs: torch.Tensor,
    obs_weight_sum: torch.Tensor,
    support_hat: torch.Tensor,
    support_hc: torch.Tensor,
    lesion_band_mask: torch.Tensor,
) -> torch.Tensor:
    """Compact frame-level stats for learned blend prediction."""
    residual_mag = residuals.norm(dim=1)
    lesion_mask = lesion_band_mask.float()
    nonlesion_mask = 1.0 - lesion_mask

    def masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (x * w).sum() / w.sum().clamp(min=1e-6)

    stats = torch.stack([
        support_hat.mean(),
        support_hat.std(),
        support_hc.mean(),
        residual_mag.mean(),
        torch.quantile(residual_mag, 0.95),
        obs_weight_sum.mean(),
        s_obs.mean(),
        masked_mean(support_hat, lesion_mask),
        masked_mean(support_hat, nonlesion_mask),
    ], dim=0)
    return stats


def build_frame_safety_stats(
    residual_mean: torch.Tensor,
    s_obs_mean: torch.Tensor,
    support_hat: torch.Tensor,
    support_hc: torch.Tensor,
    solver_drift: torch.Tensor,
    fusion_gap: torch.Tensor,
    blend_scalar: torch.Tensor,
    frame_idx: int,
    seq_len: int,
) -> torch.Tensor:
    """Frame-level stats for learned conservative fallback scaling."""
    frame_norm = torch.as_tensor(
        float(frame_idx) / max(seq_len - 1, 1),
        device=support_hat.device,
        dtype=support_hat.dtype,
    )
    stats = torch.stack([
        residual_mean,
        s_obs_mean,
        support_hat.mean(),
        support_hat.std(),
        support_hc.mean(),
        solver_drift,
        fusion_gap,
        blend_scalar,
        frame_norm,
    ], dim=0)
    return stats


class FrameBlendHead(nn.Module):
    """Predict a small frame-level blend correction around a base blend."""

    def __init__(self, stats_dim: int = 9, hidden_dim: int = 32,
                 delta_range: float = 0.08, blend_min: float = 0.0, blend_max: float = 0.35):
        super().__init__()
        self.delta_range = delta_range
        self.blend_min = blend_min
        self.blend_max = blend_max
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Start exactly at base blend until training moves it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, stats: torch.Tensor, base_blend: float) -> torch.Tensor:
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        raw = self.net(stats).squeeze()
        delta = self.delta_range * torch.tanh(raw)
        blend = torch.as_tensor(base_blend, dtype=stats.dtype, device=stats.device) + delta
        return blend.clamp(self.blend_min, self.blend_max)


class FrameSafetyHead(nn.Module):
    """Predict a conservative scaling factor in [min_scale, 1]."""

    def __init__(self, stats_dim: int = 9, hidden_dim: int = 32, min_scale: float = 0.6):
        super().__init__()
        self.min_scale = min_scale
        self.max_drop = 1.0 - min_scale
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        raw = self.net(stats).squeeze()
        # zero-init => scale = 1.0, only learns conservative damping
        damp = self.max_drop * torch.relu(torch.tanh(raw))
        return (1.0 - damp).clamp(self.min_scale, 1.0)


class FrameSelectorHead(nn.Module):
    """Predict a conservative-candidate selection probability in [0, 1]."""

    def __init__(self, stats_dim: int = 9, hidden_dim: int = 32, init_bias: float = -2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)
        raw = self.net(stats).squeeze()
        return torch.sigmoid(raw)


class HybridSupportBlendModel(nn.Module):
    """
    Learned support + learned frame-level blend.

    Support path is initialized from the original A8/A9 observability weights.
    Blend head starts as a no-op around the configured base blend.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 gate_lo: float = 0.3, gate_hi: float = 2.0,
                 blend_hidden_dim: int = 32, blend_delta_range: float = 0.08,
                 safety_hidden_dim: int = 32, safety_min_scale: float = 0.6,
                 selector_hidden_dim: int = 32):
        super().__init__()
        self.support_net = ObservabilityMLP(input_dim, hidden_dim, gate_lo, gate_hi)
        self.blend_head = FrameBlendHead(
            stats_dim=9,
            hidden_dim=blend_hidden_dim,
            delta_range=blend_delta_range,
        )
        self.safety_head = FrameSafetyHead(
            stats_dim=9,
            hidden_dim=safety_hidden_dim,
            min_scale=safety_min_scale,
        )
        self.selector_head = FrameSelectorHead(
            stats_dim=9,
            hidden_dim=selector_hidden_dim,
        )

    def forward(self, features: torch.Tensor, support_hc: torch.Tensor) -> torch.Tensor:
        return self.support_net(features, support_hc)

    def predict_support(self, features: torch.Tensor, support_hc: torch.Tensor) -> torch.Tensor:
        return self.support_net(features, support_hc)

    def predict_blend(self, stats: torch.Tensor, base_blend: float) -> torch.Tensor:
        return self.blend_head(stats, base_blend)

    def predict_safety(self, stats: torch.Tensor) -> torch.Tensor:
        return self.safety_head(stats)

    def predict_selector(self, stats: torch.Tensor) -> torch.Tensor:
        return self.selector_head(stats)


def compute_boundary_flag_vectorized(
    faces: torch.Tensor,
    component_ids: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Vectorized boundary flag computation (no Python loop)."""
    dev = faces.device
    faces_cpu = faces.cpu().numpy()
    comp_cpu = component_ids.cpu().numpy()
    edges = np.vstack([faces_cpu[:, [0, 1]], faces_cpu[:, [1, 2]], faces_cpu[:, [2, 0]]])
    cross = comp_cpu[edges[:, 0]] != comp_cpu[edges[:, 1]]
    boundary_ids = np.unique(edges[cross])
    flag = torch.zeros(N, device=dev)
    if len(boundary_ids) > 0:
        flag[torch.from_numpy(boundary_ids).long().to(dev)] = 1.0
    return flag


def build_observability_features(
    residuals: torch.Tensor,        # [N, 3]
    s_obs: torch.Tensor,            # [N]
    obs_count: torch.Tensor,        # [N]
    obs_weight_sum: torch.Tensor,   # [N]
    support_prev: torch.Tensor,     # [N]
    support_handcrafted: torch.Tensor,  # [N] hand-crafted S from Kernel B
    vertices: torch.Tensor,         # [N, 3]
    vertices_preop: torch.Tensor,   # [N, 3]
    vertices_prev: torch.Tensor | None,
    normals: torch.Tensor,          # [N, 3]
    boundary_flag: torch.Tensor,    # [N]
    lesion_band_mask: torch.Tensor, # [N]
    component_ids: torch.Tensor,    # [N]
    n_components: int,
    curvature: torch.Tensor,        # [N] pre-computed curvature proxy
) -> torch.Tensor:
    """
    Build per-vertex feature tensor. V2: 14 features (added S_handcrafted).
    Curvature is pre-computed and passed in to avoid redundant sparse matmuls.
    """
    N = residuals.shape[0]
    dev = residuals.device

    residual_mag = residuals.norm(dim=1)
    drift = (vertices - vertices_preop).norm(dim=1)
    if vertices_prev is not None:
        prev_motion = (vertices - vertices_prev).norm(dim=1)
    else:
        prev_motion = torch.zeros(N, device=dev)
    support_change = (support_prev - s_obs).abs()
    comp_norm = component_ids.float() / max(n_components - 1, 1)
    res_norm = residual_mag.clamp(min=1e-8)
    cos_rn = ((residuals * normals).sum(dim=1) / res_norm).clamp(-1, 1)

    # Stack: 14 features
    feature_tensor = torch.stack([
        residual_mag,           # 0: observation evidence
        s_obs,                  # 1: instantaneous baked support
        obs_count.float(),      # 2: obs contribution count
        obs_weight_sum,         # 3: gaussian weight sum
        support_prev,           # 4: previous support
        support_change,         # 5: |S_prev - s_obs|
        drift,                  # 6: drift from prior
        prev_motion,            # 7: previous step motion
        boundary_flag.float(),  # 8: semantic boundary
        lesion_band_mask.float(), # 9: lesion band
        curvature,              # 10: curvature proxy
        comp_norm,              # 11: normalized component index
        cos_rn,                 # 12: residual-normal cosine
        support_handcrafted,    # 13: hand-crafted support (new in V2)
    ], dim=1)  # [N, 14]

    # Per-feature normalization
    feat_mean = feature_tensor.mean(dim=0, keepdim=True)
    feat_std = feature_tensor.std(dim=0, keepdim=True).clamp(min=1e-6)
    feature_tensor = (feature_tensor - feat_mean) / feat_std

    return feature_tensor


def compute_curvature_proxy(L_phi_csr: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Pre-compute curvature proxy: ||L_Phi @ V|| per vertex."""
    Lv = torch.stack([torch.mv(L_phi_csr, vertices[:, d]) for d in range(3)], dim=1)
    return Lv.norm(dim=1)


# V2: 14 features (added S_handcrafted)
OBSERVABILITY_FEATURE_DIM = 14


# ============================================================
# Architecture variants for structural bias ablation (Exp 4)
# ============================================================

class AdditiveObservabilityMLP(nn.Module):
    """Per-vertex additive correction: S_final = clamp(S_hc + delta, 0, 1)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 delta_range: float = 0.5):
        super().__init__()
        self.delta_range = delta_range
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor,
                support_hc: torch.Tensor) -> torch.Tensor:
        raw = self.net(features).squeeze(-1)
        delta = self.delta_range * torch.tanh(raw)  # [-delta_range, +delta_range]
        return (support_hc + delta).clamp(0.0, 1.0)


class GlobalScalarGate(nn.Module):
    """Single scalar gate per frame: S_final = clamp(S_hc * g, 0, 1)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 gate_lo: float = 0.3, gate_hi: float = 2.0):
        super().__init__()
        self.gate_lo = gate_lo
        self.gate_hi = gate_hi
        # Pool per-vertex features to frame-level, then predict one scalar
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor,
                support_hc: torch.Tensor) -> torch.Tensor:
        # Mean-pool across vertices
        pooled = features.mean(dim=0, keepdim=True)  # [1, d]
        raw = self.net(pooled).squeeze()  # scalar
        gate = self.gate_lo + (self.gate_hi - self.gate_lo) * torch.sigmoid(raw)
        return (support_hc * gate).clamp(0.0, 1.0)


class MultGateNoSHC(nn.Module):
    """Multiplicative gate but without S_hc in input features (13-dim)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 gate_lo: float = 0.3, gate_hi: float = 2.0):
        super().__init__()
        self.gate_lo = gate_lo
        self.gate_hi = gate_hi
        # input_dim - 1 because we drop S_hc from features
        self.net = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor,
                support_hc: torch.Tensor) -> torch.Tensor:
        # Drop the last feature (S_hc, index 13)
        features_no_shc = features[:, :-1]  # [N, 13]
        raw = self.net(features_no_shc).squeeze(-1)
        gate = self.gate_lo + (self.gate_hi - self.gate_lo) * torch.sigmoid(raw)
        return (support_hc * gate).clamp(0.0, 1.0)


def create_observability_model(arch: str, input_dim: int, hidden_dim: int = 64,
                               gate_lo: float = 0.3, gate_hi: float = 2.0):
    """Factory for observability model variants."""
    if arch == "multiplicative" or arch == "mult_gate_v3":
        return ObservabilityMLP(input_dim, hidden_dim, gate_lo, gate_hi)
    elif arch == "additive":
        return AdditiveObservabilityMLP(input_dim, hidden_dim, delta_range=0.5)
    elif arch == "global_scalar":
        return GlobalScalarGate(input_dim, hidden_dim, gate_lo, gate_hi)
    elif arch == "mult_no_shc":
        return MultGateNoSHC(input_dim, hidden_dim, gate_lo, gate_hi)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
