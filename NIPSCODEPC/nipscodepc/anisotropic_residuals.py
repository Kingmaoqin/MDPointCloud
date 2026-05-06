"""Observation residual baking with optional tangential correction."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .sscu_engine_gpu import GPUSSCUParams, GPUTwinState


def kernel_a_anisotropic(
    state: GPUTwinState,
    obs_points: torch.Tensor,
    obs_normals: torch.Tensor,
    obs_confs: torch.Tensor,
    params: GPUSSCUParams,
    beta_tangential: float = 0.0,
    k_neighbors: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Bake partial observations into per-vertex residuals and support.

    beta_tangential=0 uses the original normal-direction projection.
    beta_tangential>0 preserves part of the tangential component, gated by
    observation confidence.
    """

    n_obs = obs_points.shape[0]
    dev = state.device

    vertices_cpu = state.vertices.detach().cpu().numpy()
    points_cpu = obs_points.detach().cpu().numpy()
    comp_cpu = state.component_ids.detach().cpu().numpy()
    tree = cKDTree(vertices_cpu)
    dists_np, knn_ids_np = tree.query(points_cpu, k=k_neighbors)

    nearest_comp = comp_cpu[knn_ids_np[:, 0]]
    knn_comp = comp_cpu[knn_ids_np]
    same_comp_mask = knn_comp == nearest_comp[:, np.newaxis]

    knn_ids_t = torch.from_numpy(knn_ids_np.astype(np.int64)).to(dev)
    dists_t = torch.from_numpy(dists_np.astype(np.float32)).to(dev)
    comp_mask_t = torch.from_numpy(same_comp_mask).to(dev)

    gauss_w = torch.exp(-(dists_t**2) / (params.sigma_mm**2))
    gauss_w = gauss_w * comp_mask_t.float()
    weights = obs_confs.unsqueeze(1) * gauss_w

    flat_ids = knn_ids_t.reshape(-1)
    v_knn = state.vertices[flat_ids].reshape(n_obs, k_neighbors, 3)
    n_knn = state.normals[flat_ids].reshape(n_obs, k_neighbors, 3)
    diffs = obs_points.unsqueeze(1) - v_knn

    proj_scalar = (diffs * n_knn).sum(dim=2, keepdim=True)
    proj_normal = proj_scalar * n_knn

    if beta_tangential > 0.0:
        proj_tangential = diffs - proj_normal
        conf_gate = obs_confs.unsqueeze(1).unsqueeze(2)
        residual_3d = proj_normal + beta_tangential * conf_gate * proj_tangential
        weighted_res = weights.unsqueeze(2) * residual_3d
    else:
        weighted_projs = weights * proj_scalar.squeeze(2)
        weighted_res = weighted_projs.unsqueeze(2) * n_knn

    residual_acc = torch.zeros(state.N, 3, device=dev, dtype=torch.float64)
    weight_acc = torch.zeros(state.N, device=dev, dtype=torch.float64)
    flat_ids_3 = flat_ids.unsqueeze(1).expand(-1, 3)
    residual_acc.scatter_add_(0, flat_ids_3, weighted_res.reshape(-1, 3).double())
    weight_acc.scatter_add_(0, flat_ids, weights.reshape(-1).double())
    residuals = (residual_acc / (weight_acc.unsqueeze(1) + 1e-8)).float()

    contrib = weights.reshape(-1).clamp(max=1.0 - 1e-10)
    log_comp = torch.log(1.0 - contrib)
    log_complement = torch.zeros(state.N, device=dev, dtype=torch.float64)
    log_complement.scatter_add_(0, flat_ids, log_comp.double())
    s_obs = (1.0 - torch.exp(log_complement)).float().clamp(0, 1)

    obs_count = torch.zeros(state.N, device=dev)
    obs_count.scatter_add_(0, flat_ids, (weights.reshape(-1) > 1e-10).float())

    obs_weight_sum = torch.zeros(state.N, device=dev)
    obs_weight_sum.scatter_add_(0, flat_ids, weights.reshape(-1).float())

    return {
        "residuals": residuals,
        "s_obs": s_obs,
        "obs_count": obs_count,
        "obs_weight_sum": obs_weight_sum,
    }
