"""Main structured safe-update algorithm, separated from experiment code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .anisotropic_residuals import kernel_a_anisotropic
from .lattice import (
    apply_grid_displacement,
    build_grid_laplacian,
    build_vertex_to_grid_weights,
    project_vertex_values_to_grid,
    solve_grid_update,
)
from .observability_model import (
    OBSERVABILITY_FEATURE_DIM,
    HybridSupportBlendModel,
    build_frame_blend_stats,
    build_observability_features,
    compute_boundary_flag_vectorized,
    compute_curvature_proxy,
)
from .sscu_engine_gpu import (
    GPUSSCUParams,
    GPUTwinState,
    compute_vertex_normals_gpu,
    init_twin_gpu,
    kernel_b_gpu,
    kernel_c_gpu,
)


@dataclass
class SafeUpdateConfig:
    """Algorithm hyperparameters for one online safe-update stream."""

    sigma_mm: float = 10.0
    gamma: float = 0.30
    diffusion_eta: float = 0.20
    diffusion_iters: int = 8
    lambda_sem: float = 1.0
    lambda_prior: float = 0.1
    lambda_les: float = 1.0
    alpha: float = 0.8
    pcg_tol: float = 1e-5
    pcg_maxiter: int = 50
    tau_vis: float = 0.5

    beta_tangential: float = 1.5
    gate_tau: float = 0.40
    gate_min: float = 0.0
    gate_signal: str = "delta_v"
    lattice_decay: float = 0.5
    preop_pullback: float = 0.6
    hard_cutoff_low: float = 0.0
    hard_cutoff_high: float = 0.0
    per_vertex_gate: bool = False
    exclude_lesion_from_pullback: bool = True

    grid_nx: int = 7
    grid_ny: int = 7
    grid_nz: int = 7
    bbox_margin_mm: float = 3.0
    lattice_lambda_elastic: float = 1.0
    lattice_lambda_prior: float = 0.1
    lattice_lambda_lesion: float = 2.0
    lattice_alpha: float = 0.8
    lattice_prior_mix: float = 0.3
    final_lattice_blend: float = 0.12

    hidden_dim: int = 64
    gate_lo: float = 0.3
    gate_hi: float = 2.0
    blend_hidden_dim: int = 32
    blend_delta_range: float = 0.03
    safety_hidden_dim: int = 32
    safety_min_scale: float = 0.6
    selector_hidden_dim: int = 32

    @classmethod
    def from_training_config(cls, cfg: dict) -> "SafeUpdateConfig":
        """Create a runtime config from a training/evaluation config dict."""

        base = cls()
        for key, value in cfg.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    def solver_params(self) -> GPUSSCUParams:
        return GPUSSCUParams(
            sigma_mm=self.sigma_mm,
            gamma=self.gamma,
            diffusion_eta=self.diffusion_eta,
            diffusion_iters=self.diffusion_iters,
            lambda_sem=self.lambda_sem,
            lambda_prior=self.lambda_prior,
            lambda_les=self.lambda_les,
            alpha=self.alpha,
            pcg_tol=self.pcg_tol,
            pcg_maxiter=self.pcg_maxiter,
            tau_vis=self.tau_vis,
            use_learned_observability=True,
        )


def load_hybrid_model(
    checkpoint_or_run_dir: str | Path,
    device: str | torch.device,
    config: SafeUpdateConfig | None = None,
) -> tuple[HybridSupportBlendModel, SafeUpdateConfig]:
    """
    Load the learned trust/blend model.

    checkpoint_or_run_dir may be either a run directory containing
    config.json and checkpoints/best.pt, or a direct checkpoint file.
    """

    path = Path(checkpoint_or_run_dir)
    if path.is_dir():
        cfg_path = path / "config.json"
        ckpt_path = path / "checkpoints" / "best.pt"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    else:
        ckpt_path = path
        cfg = {}

    runtime_cfg = config or SafeUpdateConfig.from_training_config(cfg)
    model = HybridSupportBlendModel(
        input_dim=OBSERVABILITY_FEATURE_DIM,
        hidden_dim=int(runtime_cfg.hidden_dim),
        gate_lo=float(runtime_cfg.gate_lo),
        gate_hi=float(runtime_cfg.gate_hi),
        blend_hidden_dim=int(runtime_cfg.blend_hidden_dim),
        blend_delta_range=float(runtime_cfg.blend_delta_range),
        safety_hidden_dim=int(runtime_cfg.safety_hidden_dim),
        safety_min_scale=float(runtime_cfg.safety_min_scale),
        selector_hidden_dim=int(runtime_cfg.selector_hidden_dim),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, runtime_cfg


class StructuredSafeUpdater:
    """
    Online anatomical state updater.

    The step method consumes one partial observation frame and returns the
    updated mesh state plus intermediate algorithm fields. It does not compute
    benchmark metrics, load test cases, or write experiment outputs.
    """

    def __init__(
        self,
        state: GPUTwinState,
        model: HybridSupportBlendModel,
        config: SafeUpdateConfig | None = None,
    ):
        self.state = state
        self.model = model
        self.config = config or SafeUpdateConfig()
        self.params = self.config.solver_params()

        self.boundary_flag = compute_boundary_flag_vectorized(
            self.state.faces,
            self.state.component_ids,
            self.state.N,
        )
        self.n_components = int(self.state.component_ids.max().item()) + 1
        self.vertices_prev: torch.Tensor | None = None

        self.vertices_preop_np = self.state.vertices_preop.detach().cpu().numpy()
        self.lesion_band_np = self.state.lesion_band_mask.detach().cpu().numpy().astype(np.float32)

        _, self.node_ids, self.interp_w = build_vertex_to_grid_weights(
            self.vertices_preop_np,
            self.config.grid_nx,
            self.config.grid_ny,
            self.config.grid_nz,
            self.config.bbox_margin_mm,
        )
        self.n_nodes = self.config.grid_nx * self.config.grid_ny * self.config.grid_nz
        self.grid_laplacian = build_grid_laplacian(
            self.config.grid_nx,
            self.config.grid_ny,
            self.config.grid_nz,
        )
        lesion_scalar, lesion_weight_nodes = project_vertex_values_to_grid(
            self.lesion_band_np[:, None],
            np.ones(len(self.vertices_preop_np), dtype=np.float32),
            self.node_ids,
            self.interp_w,
            self.n_nodes,
        )
        self.lesion_weight_nodes = np.clip(
            lesion_weight_nodes * lesion_scalar[:, 0],
            0.0,
            1.0,
        ).astype(np.float32)
        self.node_disp = np.zeros((self.n_nodes, 3), dtype=np.float32)

    @classmethod
    def from_phi(
        cls,
        phi_dir: str | Path,
        model: HybridSupportBlendModel,
        device: str | torch.device = "cuda:0",
        config: SafeUpdateConfig | None = None,
    ) -> "StructuredSafeUpdater":
        """Initialize the updater from a precomputed phi directory."""

        state = init_twin_gpu(str(phi_dir), device=str(device))
        return cls(state=state, model=model, config=config)

    def step(
        self,
        obs_points: torch.Tensor | np.ndarray,
        obs_normals: torch.Tensor | np.ndarray,
        obs_confidences: torch.Tensor | np.ndarray,
    ) -> tuple[GPUTwinState, dict[str, torch.Tensor | float]]:
        """Run one trust-controlled safe-update step."""

        dev = self.state.device
        obs_points_t = self._as_tensor(obs_points, dev)
        obs_normals_t = self._as_tensor(obs_normals, dev)
        obs_conf_t = self._as_tensor(obs_confidences, dev).reshape(-1)

        ka_out = kernel_a_anisotropic(
            self.state,
            obs_points_t,
            obs_normals_t,
            obs_conf_t,
            self.params,
            beta_tangential=self.config.beta_tangential,
        )
        residuals = ka_out["residuals"]
        s_obs = ka_out["s_obs"]
        support_hc = kernel_b_gpu(self.state, s_obs, self.params)

        curvature = compute_curvature_proxy(self.state.L_phi_csr, self.state.vertices)
        features = build_observability_features(
            residuals=residuals,
            s_obs=s_obs,
            obs_count=ka_out["obs_count"],
            obs_weight_sum=ka_out["obs_weight_sum"],
            support_prev=self.state.support,
            support_handcrafted=support_hc.detach(),
            vertices=self.state.vertices,
            vertices_preop=self.state.vertices_preop,
            vertices_prev=self.vertices_prev,
            normals=self.state.normals,
            boundary_flag=self.boundary_flag,
            lesion_band_mask=self.state.lesion_band_mask,
            component_ids=self.state.component_ids,
            n_components=self.n_components,
            curvature=curvature,
        )
        support_hat = self.model.predict_support(features, support_hc.detach())

        v_lattice = self._update_lattice(residuals, ka_out["obs_weight_sum"], support_hat)
        prior_weight = torch.full_like(support_hat, float(self.config.lattice_prior_mix))
        prior_target = self.state.vertices_preop + prior_weight.unsqueeze(1) * (
            v_lattice - self.state.vertices_preop
        )

        delta_v, _ = kernel_c_gpu(
            self.state,
            residuals,
            self.params,
            support_override=support_hat,
            prior_target_override=prior_target,
        )
        gate_scale, gate_scale_pv = self._robustness_gate(
            delta_v=delta_v,
            residuals=residuals,
            support_hat=support_hat,
            obs_points=obs_points_t,
        )

        alpha_min = self.params.alpha * 0.3
        alpha_per_vertex = alpha_min + (self.params.alpha - alpha_min) * support_hat
        if gate_scale_pv is not None:
            alpha_per_vertex = alpha_per_vertex * gate_scale_pv
        else:
            alpha_per_vertex = alpha_per_vertex * gate_scale

        if self.config.lattice_decay > 0.0 and gate_scale < 1.0:
            decay_strength = self.config.lattice_decay * (1.0 - gate_scale)
            self.node_disp *= 1.0 - decay_strength

        v_new = self.state.vertices + alpha_per_vertex.unsqueeze(1) * delta_v
        v_new = self._apply_preop_pullback(v_new, support_hat, gate_scale, gate_scale_pv)

        blend_stats = build_frame_blend_stats(
            residuals=residuals,
            s_obs=s_obs,
            obs_weight_sum=ka_out["obs_weight_sum"],
            support_hat=support_hat,
            support_hc=support_hc.detach(),
            lesion_band_mask=self.state.lesion_band_mask,
        )
        blend_scalar = self.model.predict_blend(blend_stats, float(self.config.final_lattice_blend))
        v_pred = (1.0 - blend_scalar) * v_new + blend_scalar * v_lattice
        normals_new = compute_vertex_normals_gpu(v_pred, self.state.faces)

        tumor_centroid = None
        if self.state.tumor_bary_ids is not None and self.state.tumor_bary_weights is not None:
            tumor_vertices = (
                v_pred[self.state.tumor_bary_ids]
                * self.state.tumor_bary_weights.unsqueeze(-1)
            ).sum(dim=1)
            tumor_centroid = tumor_vertices.mean(dim=0)

        self.vertices_prev = self.state.vertices.clone()
        self.state = GPUTwinState(
            vertices=v_pred.detach(),
            faces=self.state.faces,
            normals=normals_new.detach(),
            component_ids=self.state.component_ids,
            L_phi_csr=self.state.L_phi_csr,
            lesion_band_mask=self.state.lesion_band_mask,
            vertices_preop=self.state.vertices_preop,
            support=support_hat.detach(),
            tumor_bary_ids=self.state.tumor_bary_ids,
            tumor_bary_weights=self.state.tumor_bary_weights,
            tumor_centroid=None if tumor_centroid is None else tumor_centroid.detach(),
        )

        info: dict[str, torch.Tensor | float] = {
            "residuals": residuals.detach(),
            "support_hc": support_hc.detach(),
            "support_hat": support_hat.detach(),
            "delta_v": delta_v.detach(),
            "v_lattice": v_lattice.detach(),
            "blend_scalar": float(blend_scalar.detach().cpu().item()),
            "gate_scale": float(gate_scale),
        }
        if tumor_centroid is not None:
            info["tumor_centroid"] = tumor_centroid.detach()
        return self.state, info

    def _update_lattice(
        self,
        residuals: torch.Tensor,
        obs_weight_sum: torch.Tensor,
        support_hat: torch.Tensor,
    ) -> torch.Tensor:
        residuals_np = residuals.detach().cpu().numpy()
        obs_gate = support_hat.detach().cpu().numpy()
        obs_w_np = obs_weight_sum.detach().cpu().numpy() * obs_gate
        residual_nodes, obs_weight_nodes = project_vertex_values_to_grid(
            residuals_np,
            obs_w_np,
            self.node_ids,
            self.interp_w,
            self.n_nodes,
        )
        delta_nodes = solve_grid_update(
            self.node_disp,
            residual_nodes,
            obs_weight_nodes,
            self.grid_laplacian,
            self.lesion_weight_nodes,
            lambda_elastic=float(self.config.lattice_lambda_elastic),
            lambda_prior=float(self.config.lattice_lambda_prior),
            lambda_lesion=float(self.config.lattice_lambda_lesion),
        )
        self.node_disp = self.node_disp + float(self.config.lattice_alpha) * delta_nodes
        lattice_np = self.vertices_preop_np + apply_grid_displacement(
            self.node_disp,
            self.node_ids,
            self.interp_w,
        )
        return torch.from_numpy(lattice_np).float().to(self.state.device)

    def _robustness_gate(
        self,
        delta_v: torch.Tensor,
        residuals: torch.Tensor,
        support_hat: torch.Tensor,
        obs_points: torch.Tensor,
    ) -> tuple[float, torch.Tensor | None]:
        gate_scale = 1.0
        gate_scale_pv = None

        if self.config.gate_tau <= 0.0 and self.config.hard_cutoff_low <= 0.0:
            return gate_scale, gate_scale_pv

        if self.config.gate_signal == "delta_v":
            signal = float(delta_v.norm(dim=1).mean().item())
        elif self.config.gate_signal == "obs_to_preop":
            with torch.no_grad():
                signal = float(torch.cdist(obs_points, self.state.vertices_preop).min(dim=1).values.mean().item())
        elif self.config.gate_signal == "res_x_supp":
            signal = float((residuals.norm(dim=1) * support_hat).mean().item())
        else:
            raise ValueError(f"Unknown gate_signal: {self.config.gate_signal}")

        if self.config.hard_cutoff_low > 0.0 and signal < self.config.hard_cutoff_low:
            gate_scale = 0.0
        elif self.config.hard_cutoff_high > 0.0 and signal > self.config.hard_cutoff_high:
            gate_scale = 1.0
        elif self.config.gate_tau > 0.0:
            gate_scale = max(self.config.gate_min, min(1.0, signal / self.config.gate_tau))

        if self.config.per_vertex_gate and self.config.gate_tau > 0.0:
            sig_v = delta_v.norm(dim=1)
            local = torch.clamp(sig_v / self.config.gate_tau, min=self.config.gate_min, max=1.0)
            gate_scale_pv = local * gate_scale

        return gate_scale, gate_scale_pv

    def _apply_preop_pullback(
        self,
        vertices: torch.Tensor,
        support_hat: torch.Tensor,
        gate_scale: float,
        gate_scale_pv: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.preop_pullback <= 0.0:
            return vertices

        if gate_scale_pv is not None:
            pullback = self.config.preop_pullback * (1.0 - gate_scale_pv)
        elif gate_scale < 1.0:
            pullback = torch.full_like(support_hat, self.config.preop_pullback * (1.0 - gate_scale))
        else:
            return vertices

        if self.config.exclude_lesion_from_pullback:
            pullback = pullback * (~self.state.lesion_band_mask).float()

        pullback = pullback.clamp(0.0, 1.0).unsqueeze(1)
        return (1.0 - pullback) * vertices + pullback * self.state.vertices_preop

    @staticmethod
    def _as_tensor(array: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array.float().to(device)
        return torch.from_numpy(np.asarray(array)).float().to(device)
