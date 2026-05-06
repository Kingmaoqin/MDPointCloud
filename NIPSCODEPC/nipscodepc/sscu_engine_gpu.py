#!/usr/bin/env python3
"""
SSCU GPU Engine: All four kernels accelerated via PyTorch CUDA.

GPU acceleration strategy:
    Kernel A: Vectorized scatter on GPU (torch.scatter_add)
    Kernel B: Sparse CSR matmul for Laplacian diffusion
    Kernel C: Custom PCG solver using torch sparse CSR SpMV
    Kernel D: Batched barycentric transport + BVH distance via torch

Targets: 10-30 ms/step on A100 (vs ~1300 ms CPU)

Requires: PyTorch 2.6+ with CUDA, scipy (for CSR conversion at init)
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _maybe_cuda_synchronize(device: torch.device | str | None = None) -> None:
    """Synchronize only when running on CUDA."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


# ---------------------------------------------------------------------------
# GPU Twin State
# ---------------------------------------------------------------------------

@dataclass
class GPUTwinState:
    """Twin state T_t on GPU."""
    # Geometry (GPU tensors)
    vertices: torch.Tensor         # [N, 3] float32
    faces: torch.Tensor            # [M, 3] int64
    normals: torch.Tensor          # [N, 3] float32

    # Partition (GPU, fixed)
    component_ids: torch.Tensor    # [N] int32
    L_phi_csr: torch.Tensor        # sparse CSR [N, N]
    lesion_band_mask: torch.Tensor # [N] bool

    # Preop reference (GPU, fixed)
    vertices_preop: torch.Tensor   # [N, 3] float32

    # Support field
    support: torch.Tensor          # [N] float32

    # Lesion transport (GPU, fixed)
    tumor_bary_ids: torch.Tensor | None = None       # [K, n_nb] int64
    tumor_bary_weights: torch.Tensor | None = None   # [K, n_nb] float32

    # Lesion state (updated each step)
    tumor_centroid: torch.Tensor | None = None        # [3]
    tumor_vertices: torch.Tensor | None = None        # [K, 3]
    tumor_uncertainty: float = 1.0
    tumor_to_surface_dist: float = float('inf')

    @property
    def device(self):
        return self.vertices.device

    @property
    def N(self):
        return self.vertices.shape[0]


@dataclass
class GPUSSCUParams:
    """SSCU hyperparameters."""
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
    # Learned observability extension (SCSU)
    use_learned_observability: bool = False


@dataclass
class GPUStepTimings:
    """Per-step timing breakdown (ms)."""
    kernel_a_ms: float = 0.0
    kernel_b_ms: float = 0.0
    kernel_c_ms: float = 0.0
    kernel_d_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Utility: scipy CSR → torch sparse CSR on GPU
# ---------------------------------------------------------------------------

def scipy_csr_to_torch(mat: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert scipy CSR matrix to PyTorch sparse CSR tensor on device."""
    mat = mat.astype(np.float32)
    crow = torch.from_numpy(mat.indptr.astype(np.int64)).to(device)
    col = torch.from_numpy(mat.indices.astype(np.int64)).to(device)
    val = torch.from_numpy(mat.data.astype(np.float32)).to(device)
    return torch.sparse_csr_tensor(crow, col, val, size=mat.shape, device=device)


def compute_vertex_normals_gpu(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute area-weighted vertex normals on GPU."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)  # [M, 3]

    vn = torch.zeros_like(verts)
    vn.scatter_add_(0, faces[:, 0:1].expand(-1, 3), fn)
    vn.scatter_add_(0, faces[:, 1:2].expand(-1, 3), fn)
    vn.scatter_add_(0, faces[:, 2:3].expand(-1, 3), fn)

    norms = vn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return vn / norms


# ---------------------------------------------------------------------------
# Kernel A: Residual Baking (GPU)
# ---------------------------------------------------------------------------

def kernel_a_gpu(state: GPUTwinState, obs_points: torch.Tensor,
                 obs_normals: torch.Tensor, obs_confs: torch.Tensor,
                 params: GPUSSCUParams,
                 k_neighbors: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU Kernel A: Bake observations into per-vertex residuals.

    Uses CPU KD-tree for k-nearest-neighbor, then GPU scatter for accumulation.
    Each observation spreads support/residuals to k neighbors via Gaussian weighting.
    Component-aware: neighbors in different components from the nearest are masked out
    to prevent cross-boundary leakage (Theorem 1 compliance).
    """
    N = state.N
    dev = state.device
    N_obs = obs_points.shape[0]

    # K-nearest-neighbor on CPU (KD-tree)
    V_cpu = state.vertices.detach().cpu().numpy()
    P_cpu = obs_points.cpu().numpy()
    comp_cpu = state.component_ids.cpu().numpy()
    tree = cKDTree(V_cpu)
    dists_np, knn_ids_np = tree.query(P_cpu, k=k_neighbors)  # [N_obs, k]

    # Component-aware masking: keep only neighbors in same component as nearest
    nearest_comp = comp_cpu[knn_ids_np[:, 0]]                          # [N_obs]
    knn_comp = comp_cpu[knn_ids_np]                                    # [N_obs, k]
    same_comp_mask = (knn_comp == nearest_comp[:, np.newaxis])         # [N_obs, k] bool

    knn_ids_t = torch.from_numpy(knn_ids_np.astype(np.int64)).to(dev)  # [N_obs, k]
    dists_t = torch.from_numpy(dists_np.astype(np.float32)).to(dev)    # [N_obs, k]
    comp_mask_t = torch.from_numpy(same_comp_mask).to(dev)             # [N_obs, k] bool

    # Gaussian distance weights with component mask
    d2 = dists_t ** 2                                                   # [N_obs, k]
    gauss_w = torch.exp(-d2 / (params.sigma_mm ** 2))                  # [N_obs, k]
    gauss_w = gauss_w * comp_mask_t.float()                            # zero out cross-component
    weights = obs_confs.unsqueeze(1) * gauss_w                         # [N_obs, k]

    # Residual computation per (obs, neighbor) pair
    flat_ids = knn_ids_t.reshape(-1)                                   # [N_obs*k]
    V_knn = state.vertices[flat_ids].reshape(N_obs, k_neighbors, 3)    # [N_obs, k, 3]
    N_knn = state.normals[flat_ids].reshape(N_obs, k_neighbors, 3)     # [N_obs, k, 3]
    diffs = obs_points.unsqueeze(1) - V_knn                            # [N_obs, k, 3]

    projs = (diffs * N_knn).sum(dim=2)                                 # [N_obs, k]
    weighted_projs = weights * projs                                   # [N_obs, k]
    weighted_res = weighted_projs.unsqueeze(2) * N_knn                 # [N_obs, k, 3]

    # Scatter accumulate R, W
    R = torch.zeros(N, 3, device=dev, dtype=torch.float64)
    W = torch.zeros(N, device=dev, dtype=torch.float64)
    flat_ids_3 = flat_ids.unsqueeze(1).expand(-1, 3)                   # [N_obs*k, 3]
    R.scatter_add_(0, flat_ids_3, weighted_res.reshape(-1, 3).double())
    W.scatter_add_(0, flat_ids, weights.reshape(-1).double())

    residuals = (R / (W.unsqueeze(1) + 1e-8)).float()

    # Support: s_obs via scatter of log(1 - w)
    contrib = weights.reshape(-1).clamp(max=1.0 - 1e-10)              # [N_obs*k]
    log_comp = torch.log(1.0 - contrib)

    log_complement = torch.zeros(N, device=dev, dtype=torch.float64)
    log_complement.scatter_add_(0, flat_ids, log_comp.double())
    s_obs = (1.0 - torch.exp(log_complement)).float().clamp(0, 1)

    return residuals, s_obs


def kernel_a_gpu_extended(state: GPUTwinState, obs_points: torch.Tensor,
                          obs_normals: torch.Tensor, obs_confs: torch.Tensor,
                          params: GPUSSCUParams,
                          k_neighbors: int = 8) -> dict:
    """
    Extended Kernel A that also returns obs_count and obs_weight_sum
    needed by the learned observability feature builder.

    Returns dict with keys: residuals, s_obs, obs_count, obs_weight_sum
    """
    N = state.N
    dev = state.device
    N_obs = obs_points.shape[0]

    V_cpu = state.vertices.detach().cpu().numpy()
    P_cpu = obs_points.cpu().numpy()
    comp_cpu = state.component_ids.cpu().numpy()
    tree = cKDTree(V_cpu)
    dists_np, knn_ids_np = tree.query(P_cpu, k=k_neighbors)

    nearest_comp = comp_cpu[knn_ids_np[:, 0]]
    knn_comp = comp_cpu[knn_ids_np]
    same_comp_mask = (knn_comp == nearest_comp[:, np.newaxis])

    knn_ids_t = torch.from_numpy(knn_ids_np.astype(np.int64)).to(dev)
    dists_t = torch.from_numpy(dists_np.astype(np.float32)).to(dev)
    comp_mask_t = torch.from_numpy(same_comp_mask).to(dev)

    d2 = dists_t ** 2
    gauss_w = torch.exp(-d2 / (params.sigma_mm ** 2))
    gauss_w = gauss_w * comp_mask_t.float()
    weights = obs_confs.unsqueeze(1) * gauss_w

    flat_ids = knn_ids_t.reshape(-1)
    V_knn = state.vertices[flat_ids].reshape(N_obs, k_neighbors, 3)
    N_knn = state.normals[flat_ids].reshape(N_obs, k_neighbors, 3)
    diffs = obs_points.unsqueeze(1) - V_knn
    projs = (diffs * N_knn).sum(dim=2)
    weighted_projs = weights * projs
    weighted_res = weighted_projs.unsqueeze(2) * N_knn

    R = torch.zeros(N, 3, device=dev, dtype=torch.float64)
    W = torch.zeros(N, device=dev, dtype=torch.float64)
    flat_ids_3 = flat_ids.unsqueeze(1).expand(-1, 3)
    R.scatter_add_(0, flat_ids_3, weighted_res.reshape(-1, 3).double())
    W.scatter_add_(0, flat_ids, weights.reshape(-1).double())
    residuals = (R / (W.unsqueeze(1) + 1e-8)).float()

    # Support: s_obs
    contrib = weights.reshape(-1).clamp(max=1.0 - 1e-10)
    log_comp = torch.log(1.0 - contrib)
    log_complement = torch.zeros(N, device=dev, dtype=torch.float64)
    log_complement.scatter_add_(0, flat_ids, log_comp.double())
    s_obs = (1.0 - torch.exp(log_complement)).float().clamp(0, 1)

    # Extra features for observability model
    # obs_count: how many (obs, neighbor) pairs contribute to each vertex
    obs_count = torch.zeros(N, device=dev)
    ones_flat = (weights.reshape(-1) > 1e-10).float()
    obs_count.scatter_add_(0, flat_ids, ones_flat)

    # obs_weight_sum: sum of gaussian weights per vertex
    obs_weight_sum = torch.zeros(N, device=dev)
    obs_weight_sum.scatter_add_(0, flat_ids, weights.reshape(-1).float())

    return {
        "residuals": residuals,
        "s_obs": s_obs,
        "obs_count": obs_count,
        "obs_weight_sum": obs_weight_sum,
    }


# ---------------------------------------------------------------------------
# Kernel B: Support Diffusion (GPU)
# ---------------------------------------------------------------------------

def kernel_b_gpu(state: GPUTwinState, s_obs: torch.Tensor,
                 params: GPUSSCUParams) -> torch.Tensor:
    """
    GPU Kernel B: EMA update + within-component Laplacian diffusion.
    Uses sparse CSR matmul for diffusion: S ← S - η * L_Φ @ S
    """
    # EMA
    S = (params.gamma * state.support + (1 - params.gamma) * s_obs).clamp(0, 1)

    # Diffusion via sparse CSR matmul
    for _ in range(params.diffusion_iters):
        LS = torch.mv(state.L_phi_csr, S)  # sparse CSR @ dense vector
        S = (S - params.diffusion_eta * LS).clamp(0, 1)

    return S


# ---------------------------------------------------------------------------
# Kernel C: PCG Solver (GPU)
# ---------------------------------------------------------------------------

def _build_system_matrix_gpu(state: GPUTwinState,
                             params: GPUSSCUParams,
                             support_override: torch.Tensor | None = None) -> torch.Tensor:
    """
    Build A_t = diag(S) + λ_sem * L_Φ + λ_prior * diag(1-S) + λ_les * diag(1_B) + εI
    as a sparse CSR tensor on GPU.
    """
    N = state.N
    dev = state.device

    # Start from L_Φ scaled by λ_sem
    # We need to add diagonal terms to the existing CSR
    # Strategy: convert L_Φ to COO, add diagonal entries, convert back to CSR

    L = state.L_phi_csr

    # Extract L_Φ as COO
    L_coo = L.to_sparse_coo().coalesce()
    L_indices = L_coo.indices()  # [2, nnz]
    L_values = L_coo.values()    # [nnz]

    # Scale L_Φ by λ_sem
    scaled_values = params.lambda_sem * L_values

    # Diagonal entries: S + λ_prior*(1-S) + λ_les*1_B + ε
    S = support_override if support_override is not None else state.support
    diag_vals = (S
                 + params.lambda_prior * (1.0 - S)
                 + params.lambda_les * state.lesion_band_mask.float()
                 + 1e-6)

    # Add diagonal to COO
    diag_indices = torch.arange(N, device=dev).unsqueeze(0).expand(2, -1)
    all_indices = torch.cat([L_indices, diag_indices], dim=1)
    all_values = torch.cat([scaled_values, diag_vals])

    # Build COO and coalesce (sums duplicate indices)
    A_coo = torch.sparse_coo_tensor(all_indices, all_values, (N, N)).coalesce()

    # Convert to CSR for efficient SpMV
    A_csr = A_coo.to_sparse_csr()
    return A_csr


def _pcg_gpu_batched(A_csr: torch.Tensor, B: torch.Tensor,
                     diag_A: torch.Tensor,
                     tol: float = 1e-5, maxiter: int = 50,
                     X0: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """
    Batched PCG: solve A @ X = B for X where B is [N, D].

    Solves D systems simultaneously, reducing Python loop overhead by D×.
    Uses Jacobi preconditioner M = diag(A)^{-1}.

    Returns: (X [N, D], max_iterations_used)
    """
    N, D = B.shape
    dev = B.device

    M_inv = 1.0 / diag_A.clamp(min=1e-8)  # [N]

    B_norms = B.norm(dim=0)  # [D]
    if B_norms.max() < 1e-15:
        return torch.zeros_like(B), 0

    if X0 is not None:
        X = X0.clone()
        R = B - torch.sparse.mm(A_csr, X)  # [N, D]
    else:
        X = torch.zeros_like(B)
        R = B.clone()

    Z = M_inv.unsqueeze(1) * R  # [N, D]
    P = Z.clone()
    RZ = (R * Z).sum(dim=0)  # [D]

    converged = B_norms < 1e-15  # [D] bool
    max_iter_used = 0

    for k in range(maxiter):
        AP = torch.sparse.mm(A_csr, P)  # [N, D]
        PAP = (P * AP).sum(dim=0)  # [D]
        PAP = PAP.clamp(min=1e-30)
        alpha = RZ / PAP  # [D]
        X = X + alpha.unsqueeze(0) * P
        R = R - alpha.unsqueeze(0) * AP

        R_norms = R.norm(dim=0)  # [D]
        newly_converged = (R_norms / B_norms.clamp(min=1e-15)) < tol
        converged = converged | newly_converged

        if converged.all():
            return X, k + 1

        Z = M_inv.unsqueeze(1) * R
        RZ_new = (R * Z).sum(dim=0)
        beta = RZ_new / RZ.clamp(min=1e-30)
        P = Z + beta.unsqueeze(0) * P
        RZ = RZ_new
        max_iter_used = k + 1

    return X, max_iter_used


def _pcg_gpu(A_csr: torch.Tensor, b: torch.Tensor,
             diag_A: torch.Tensor,
             tol: float = 1e-5, maxiter: int = 50,
             x0: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """
    Preconditioned Conjugate Gradient on GPU.

    Solves Ax = b where A is sparse CSR, with Jacobi preconditioner M = diag(A)^{-1}.
    Supports warm-start via x0 (initial guess).

    All operations are GPU tensor ops — no CPU roundtrip.

    Returns: (x, n_iterations)
    """
    N = b.shape[0]
    dev = b.device

    # Jacobi preconditioner
    M_inv = 1.0 / diag_A.clamp(min=1e-8)

    b_norm = b.norm()
    if b_norm < 1e-15:
        return torch.zeros(N, device=dev), 0

    if x0 is not None:
        x = x0.clone()
        r = b - torch.mv(A_csr, x)        # r = b - A@x0
    else:
        x = torch.zeros(N, device=dev)
        r = b.clone()

    z = M_inv * r
    p = z.clone()
    rz = torch.dot(r, z)

    # Check if already converged (warm-start case)
    if r.norm() / b_norm < tol:
        return x, 0

    for k in range(maxiter):
        Ap = torch.mv(A_csr, p)        # SpMV: A @ p
        pAp = torch.dot(p, Ap)
        if pAp.abs() < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        if r.norm() / b_norm < tol:
            return x, k + 1

        z = M_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz.clamp(min=1e-30)
        p = z + beta * p
        rz = rz_new

    return x, maxiter


@dataclass
class KernelCCache:
    """Pre-computed data for fast kernel_c calls within a case."""
    L_scaled_indices: torch.Tensor   # [2, nnz_L] — L_phi COO indices
    L_scaled_values: torch.Tensor    # [nnz_L] — lambda_sem * L_phi values
    L_diag: torch.Tensor             # [N] — L_phi diagonal contribution
    diag_indices: torch.Tensor       # [2, N] — diagonal COO indices
    N: int


def build_kernel_c_cache(state: GPUTwinState, params: GPUSSCUParams) -> KernelCCache:
    """Pre-compute fixed structure for kernel_c (call once per case)."""
    N = state.N
    dev = state.device

    L_coo = state.L_phi_csr.to_sparse_coo().coalesce()
    L_indices = L_coo.indices()
    L_values = L_coo.values()

    # Scaled L values
    L_scaled_values = params.lambda_sem * L_values

    # L diagonal
    L_diag = torch.zeros(N, device=dev)
    diag_mask = L_indices[0] == L_indices[1]
    if diag_mask.any():
        L_diag.scatter_add_(0, L_indices[0][diag_mask], L_scaled_values[diag_mask])

    # Diagonal COO indices (reusable)
    diag_indices = torch.arange(N, device=dev).unsqueeze(0).expand(2, -1)

    return KernelCCache(
        L_scaled_indices=L_indices,
        L_scaled_values=L_scaled_values,
        L_diag=L_diag,
        diag_indices=diag_indices,
        N=N,
    )


def _build_system_matrix_fast(S: torch.Tensor, state: GPUTwinState,
                               params: GPUSSCUParams,
                               cache: KernelCCache) -> torch.Tensor:
    """Build A_t using cached L_phi structure (avoids repeated COO conversion)."""
    diag_vals = (S + params.lambda_prior * (1.0 - S)
                 + params.lambda_les * state.lesion_band_mask.float() + 1e-6)
    all_indices = torch.cat([cache.L_scaled_indices, cache.diag_indices], dim=1)
    all_values = torch.cat([cache.L_scaled_values, diag_vals])
    A_coo = torch.sparse_coo_tensor(all_indices, all_values,
                                     (cache.N, cache.N)).coalesce()
    return A_coo.to_sparse_csr()


def _compute_diag_and_L_cache(state: GPUTwinState, params: GPUSSCUParams):
    """Pre-compute L_phi diagonal contribution (fixed per case, reusable)."""
    N = state.N
    dev = state.device
    L_diag = torch.zeros(N, device=dev)
    L_coo = state.L_phi_csr.to_sparse_coo().coalesce()
    L_idx = L_coo.indices()
    L_val = L_coo.values()
    diag_mask = L_idx[0] == L_idx[1]
    if diag_mask.any():
        L_diag.scatter_add_(0, L_idx[0][diag_mask], params.lambda_sem * L_val[diag_mask])
    return L_diag


def kernel_c_gpu(state: GPUTwinState, residuals: torch.Tensor,
                 params: GPUSSCUParams,
                 support_override: torch.Tensor | None = None,
                 prior_target_override: torch.Tensor | None = None,
                 delta_V_prev: torch.Tensor | None = None,
                 L_diag_cache: torch.Tensor | None = None,
                 kc_cache: KernelCCache | None = None,
                 ) -> tuple[torch.Tensor, dict]:
    """
    GPU Kernel C: Solve SSCU system A_t ΔV = b_t via PCG.

    Solves 3 independent systems (one per coordinate axis).
    If kc_cache is provided, uses cached L_phi structure (much faster).
    """
    N = state.N
    dev = state.device
    S = support_override if support_override is not None else state.support

    # Build system matrix (use cache if available)
    if kc_cache is not None:
        A_csr = _build_system_matrix_fast(S, state, params, kc_cache)
        L_diag = kc_cache.L_diag
    else:
        A_csr = _build_system_matrix_gpu(state, params, support_override=support_override)
        L_diag = L_diag_cache if L_diag_cache is not None else _compute_diag_and_L_cache(state, params)

    # Diagonal for preconditioner
    diag_A = S + params.lambda_prior * (1.0 - S) + \
             params.lambda_les * state.lesion_band_mask.float() + 1e-6 + L_diag

    # RHS: b = diag(S)*r + λ_prior * diag(1-S) * (V_prior - V_{t-1})
    prior_target = prior_target_override if prior_target_override is not None else state.vertices_preop
    prior_pull = prior_target - state.vertices
    b = (S.unsqueeze(1) * residuals
         + params.lambda_prior * (1.0 - S).unsqueeze(1) * prior_pull)

    # Batched solve: all 3 axes simultaneously
    delta_V, max_iters = _pcg_gpu_batched(
        A_csr, b, diag_A,
        tol=params.pcg_tol, maxiter=params.pcg_maxiter,
        X0=delta_V_prev,
    )

    solve_info = {"total_pcg_iters": max_iters * 3, "avg_iters": max_iters}
    return delta_V, solve_info


class _ImplicitPCGSolve(torch.autograd.Function):
    """
    Custom autograd for PCG solve with implicit differentiation.

    Forward: solve A(S) x = b(S) via PCG  (no graph recorded for iterations)
    Backward: uses implicit function theorem:
        dx/dS = A^{-1} (db/dS - dA/dS · x)
    This replaces backprop through N PCG iterations with ONE extra PCG solve.
    """

    @staticmethod
    def forward(ctx, S, residuals, vertices_preop, vertices, L_phi_csr,
                lesion_band_mask, lambda_sem, lambda_prior, lambda_les,
                pcg_tol, pcg_maxiter, delta_V_prev=None):
        N = S.shape[0]
        dev = S.device

        # Build A_csr from S (no grad tracking needed in forward)
        with torch.no_grad():
            L_coo = L_phi_csr.to_sparse_coo().coalesce()
            L_indices = L_coo.indices()
            L_values = L_coo.values()

            diag_vals = (S + lambda_prior * (1.0 - S)
                         + lambda_les * lesion_band_mask.float() + 1e-6)
            diag_indices = torch.arange(N, device=dev).unsqueeze(0).expand(2, -1)
            all_indices = torch.cat([L_indices, diag_indices], dim=1)
            all_values = torch.cat([lambda_sem * L_values, diag_vals])
            A_coo = torch.sparse_coo_tensor(all_indices, all_values, (N, N)).coalesce()
            A_csr = A_coo.to_sparse_csr()

            # Diagonal for preconditioner
            L_diag = torch.zeros(N, device=dev)
            diag_mask = L_indices[0] == L_indices[1]
            if diag_mask.any():
                L_diag.scatter_add_(0, L_indices[0][diag_mask],
                                    lambda_sem * L_values[diag_mask])
            diag_A = diag_vals + L_diag

            # RHS
            prior_pull = vertices_preop - vertices
            b = (S.unsqueeze(1) * residuals
                 + lambda_prior * (1.0 - S).unsqueeze(1) * prior_pull)

            # Batched solve: all 3 axes simultaneously
            delta_V, total_iters = _pcg_gpu_batched(
                A_csr, b, diag_A,
                tol=pcg_tol, maxiter=pcg_maxiter,
                X0=delta_V_prev,
            )
            total_iters = total_iters * 3  # for compatibility

        # Save for backward
        ctx.save_for_backward(S, delta_V, residuals, vertices_preop, vertices,
                              lesion_band_mask, diag_A)
        ctx.L_phi_csr = L_phi_csr
        ctx.lambda_sem = lambda_sem
        ctx.lambda_prior = lambda_prior
        ctx.lambda_les = lambda_les
        ctx.pcg_tol = pcg_tol
        ctx.pcg_maxiter = pcg_maxiter
        ctx.total_iters = total_iters

        return delta_V

    @staticmethod
    def backward(ctx, grad_delta_V):
        (S, delta_V, residuals, vertices_preop, vertices,
         lesion_band_mask, diag_A) = ctx.saved_tensors
        L_phi_csr = ctx.L_phi_csr
        lambda_sem = ctx.lambda_sem
        lambda_prior = ctx.lambda_prior
        lambda_les = ctx.lambda_les

        N = S.shape[0]
        dev = S.device

        # Rebuild A_csr (needed for the adjoint solve)
        L_coo = L_phi_csr.to_sparse_coo().coalesce()
        L_indices = L_coo.indices()
        L_values = L_coo.values()
        diag_vals = (S + lambda_prior * (1.0 - S)
                     + lambda_les * lesion_band_mask.float() + 1e-6)
        diag_indices = torch.arange(N, device=dev).unsqueeze(0).expand(2, -1)
        all_indices = torch.cat([L_indices, diag_indices], dim=1)
        all_values = torch.cat([lambda_sem * L_values, diag_vals.detach()])
        A_coo = torch.sparse_coo_tensor(all_indices, all_values, (N, N)).coalesce()
        A_csr = A_coo.to_sparse_csr()

        # Adjoint solve: A^T lambda = grad_delta_V  (A is symmetric, so A^T = A)
        # Batched solve for all 3 axes at once
        adjoint, _ = _pcg_gpu_batched(A_csr, grad_delta_V, diag_A.detach(),
                                       tol=ctx.pcg_tol, maxiter=ctx.pcg_maxiter)

        # Implicit diff: dL/dS = -lambda^T (dA/dS x) + lambda^T (db/dS)
        #
        # dA/dS: A = diag(S) + lambda_prior*diag(1-S) + ... + lambda_sem*L_phi
        #   dA/dS_i = diag(e_i) - lambda_prior*diag(e_i) = (1 - lambda_prior)*diag(e_i)
        #   So (dA/dS) x  for each i:  (1 - lambda_prior) * x_i * e_i
        #   => sum over coords: (1-lambda_prior) * sum_d(adjoint_d * delta_V_d)
        #
        # db/dS: b = diag(S)*r + lambda_prior*diag(1-S)*(V0-V)
        #   db/dS_i = r_i - lambda_prior*(V0-V)_i
        #   => lambda^T db/dS = sum_d adjoint_d * (r_d - lambda_prior*(V0-V)_d)

        prior_pull = vertices_preop - vertices  # [N, 3]

        # db/dS contribution: for each vertex i, sum over 3 coords
        db_dS = residuals - lambda_prior * prior_pull  # [N, 3]
        term_b = (adjoint * db_dS).sum(dim=1)  # [N]

        # dA/dS contribution
        dA_coeff = 1.0 - lambda_prior  # scalar
        term_A = dA_coeff * (adjoint * delta_V).sum(dim=1)  # [N]

        grad_S = term_b - term_A  # [N]

        # Return grads: S gets grad, everything else None (12 args total including delta_V_prev)
        return grad_S, None, None, None, None, None, None, None, None, None, None, None


def kernel_c_gpu_implicit(state: GPUTwinState, residuals: torch.Tensor,
                          params: GPUSSCUParams,
                          support_override: torch.Tensor | None = None,
                          delta_V_prev: torch.Tensor | None = None,
                          ) -> tuple[torch.Tensor, dict]:
    """
    GPU Kernel C with implicit differentiation.

    Same interface as kernel_c_gpu, but uses implicit function theorem
    for backward pass (1 extra PCG solve instead of backprop through N iterations).
    Supports warm-start via delta_V_prev.
    """
    S = support_override if support_override is not None else state.support

    delta_V = _ImplicitPCGSolve.apply(
        S, residuals.detach(), state.vertices_preop.detach(),
        state.vertices.detach(), state.L_phi_csr,
        state.lesion_band_mask, params.lambda_sem, params.lambda_prior,
        params.lambda_les, params.pcg_tol, params.pcg_maxiter,
        delta_V_prev.detach() if delta_V_prev is not None else None,
    )

    solve_info = {"total_pcg_iters": _ImplicitPCGSolve.total_iters_cache
                  if hasattr(_ImplicitPCGSolve, 'total_iters_cache') else 0,
                  "avg_iters": 0, "implicit_diff": True}
    return delta_V, solve_info


# ---------------------------------------------------------------------------
# Kernel D: Lesion Transport + Distance (GPU)
# ---------------------------------------------------------------------------

def kernel_d_gpu(state: GPUTwinState, delta_V: torch.Tensor,
                 params: GPUSSCUParams) -> dict:
    """
    GPU Kernel D: Barycentric lesion transport + distance query.
    """
    result = {
        "tumor_centroid": None,
        "tumor_vertices": None,
        "tumor_to_surface_dist": float('inf'),
        "tumor_uncertainty": 1.0,
    }

    if state.tumor_bary_ids is None:
        return result

    V_new = state.vertices + params.alpha * delta_V

    # Barycentric transport: vectorized gather + weighted sum
    # ids: [K, n_nb], weights: [K, n_nb]
    gathered = V_new[state.tumor_bary_ids]            # [K, n_nb, 3]
    wts = state.tumor_bary_weights.unsqueeze(-1)      # [K, n_nb, 1]
    tumor_V_new = (gathered * wts).sum(dim=1)          # [K, 3]
    tumor_centroid_new = tumor_V_new.mean(dim=0)       # [3]

    # Distance: tumor boundary to supported surface
    supported_mask = state.support > params.tau_vis
    if supported_mask.sum() > 0:
        sup_verts = V_new[supported_mask]              # [S, 3]
        # Pairwise distance: for moderate K, S this is fast on GPU
        # Use chunked approach if memory is concern
        K = tumor_V_new.shape[0]
        S_count = sup_verts.shape[0]

        if K * S_count < 50_000_000:  # < 200MB at float32
            dists = torch.cdist(tumor_V_new.unsqueeze(0),
                                sup_verts.unsqueeze(0)).squeeze(0)  # [K, S]
            d_t = float(dists.min().item())
        else:
            # Chunked for large meshes
            min_dist = float('inf')
            chunk = 4096
            for i in range(0, K, chunk):
                chunk_t = tumor_V_new[i:i+chunk]
                d = torch.cdist(chunk_t.unsqueeze(0),
                                sup_verts.unsqueeze(0)).squeeze(0)
                min_dist = min(min_dist, float(d.min().item()))
            d_t = min_dist
    else:
        d_t = float('inf')

    # Uncertainty from lesion band support
    lesion_support = state.support[state.lesion_band_mask]
    uncertainty = 1.0 - float(lesion_support.mean().item()) if len(lesion_support) > 0 else 1.0

    result["tumor_centroid"] = tumor_centroid_new
    result["tumor_vertices"] = tumor_V_new
    result["tumor_to_surface_dist"] = d_t
    result["tumor_uncertainty"] = uncertainty

    return result


# ---------------------------------------------------------------------------
# Full GPU SSCU Update Step
# ---------------------------------------------------------------------------

def sscu_update_gpu(state: GPUTwinState, obs_points: torch.Tensor,
                    obs_normals: torch.Tensor, obs_confs: torch.Tensor,
                    params: GPUSSCUParams) -> tuple[GPUTwinState, GPUStepTimings, dict]:
    """
    One SSCU update step, fully on GPU (except KD-tree in Kernel A).

    Returns: (new_state, timings, metadata)
    """
    timings = GPUStepTimings()
    meta = {}

    _maybe_cuda_synchronize(state.device)

    # --- Kernel A ---
    t0 = time.perf_counter()
    residuals, s_obs = kernel_a_gpu(state, obs_points, obs_normals, obs_confs, params)
    _maybe_cuda_synchronize(state.device)
    timings.kernel_a_ms = (time.perf_counter() - t0) * 1000
    meta["residual_norm_mean"] = float(residuals.norm(dim=1).mean().item())
    meta["s_obs_mean"] = float(s_obs.mean().item())

    # --- Kernel B ---
    t0 = time.perf_counter()
    S_new = kernel_b_gpu(state, s_obs, params)
    _maybe_cuda_synchronize(state.device)
    timings.kernel_b_ms = (time.perf_counter() - t0) * 1000
    meta["support_mean"] = float(S_new.mean().item())

    state.support = S_new

    # --- Kernel C ---
    t0 = time.perf_counter()
    delta_V, solve_info = kernel_c_gpu(state, residuals, params)
    _maybe_cuda_synchronize(state.device)
    timings.kernel_c_ms = (time.perf_counter() - t0) * 1000
    meta["solve_info"] = solve_info
    meta["delta_V_norm_mean"] = float(delta_V.norm(dim=1).mean().item())

    # --- Kernel D ---
    t0 = time.perf_counter()
    lesion_result = kernel_d_gpu(state, delta_V, params)
    _maybe_cuda_synchronize(state.device)
    timings.kernel_d_ms = (time.perf_counter() - t0) * 1000
    meta["tumor_to_surface_dist"] = lesion_result["tumor_to_surface_dist"]
    meta["tumor_uncertainty"] = lesion_result["tumor_uncertainty"]

    # --- Apply vertex update with support-aware alpha ---
    # High-support vertices get full alpha; low-support get damped (reduces overshoot)
    alpha_min = params.alpha * 0.3  # minimum alpha for unsupported vertices
    alpha_per_vertex = alpha_min + (params.alpha - alpha_min) * S_new  # [N]
    V_new = state.vertices + alpha_per_vertex.unsqueeze(1) * delta_V
    N_new = compute_vertex_normals_gpu(V_new, state.faces)

    timings.total_ms = (timings.kernel_a_ms + timings.kernel_b_ms
                        + timings.kernel_c_ms + timings.kernel_d_ms)

    new_state = GPUTwinState(
        vertices=V_new,
        faces=state.faces,
        normals=N_new,
        component_ids=state.component_ids,
        L_phi_csr=state.L_phi_csr,
        lesion_band_mask=state.lesion_band_mask,
        vertices_preop=state.vertices_preop,
        support=S_new,
        tumor_bary_ids=state.tumor_bary_ids,
        tumor_bary_weights=state.tumor_bary_weights,
        tumor_centroid=lesion_result["tumor_centroid"],
        tumor_vertices=lesion_result["tumor_vertices"],
        tumor_uncertainty=lesion_result["tumor_uncertainty"],
        tumor_to_surface_dist=lesion_result["tumor_to_surface_dist"],
    )

    return new_state, timings, meta


# ---------------------------------------------------------------------------
# SCSU Update Step: with learned observability field
# ---------------------------------------------------------------------------

def scsu_update_gpu(state: GPUTwinState, obs_points: torch.Tensor,
                    obs_normals: torch.Tensor, obs_confs: torch.Tensor,
                    params: GPUSSCUParams,
                    obs_model: "torch.nn.Module",
                    vertices_prev: torch.Tensor | None = None,
                    ) -> tuple[GPUTwinState, GPUStepTimings, dict]:
    """
    One SCSU update step with learned observability field.

    Uses the observability MLP to predict hat_S_t instead of hand-crafted
    EMA+diffusion. The predicted hat_S_t replaces S_t in A_t, b_t, and alpha.

    Args:
        state: current twin state
        obs_points/normals/confs: observation data
        params: SSCU hyperparameters
        obs_model: ObservabilityMLP instance
        vertices_prev: previous-frame vertices for temporal feature (None if t=0)

    Returns: (new_state, timings, metadata)
             metadata includes 'support_hat' tensor for loss computation
    """
    from .observability_model import (
        build_observability_features, compute_boundary_flag_vectorized,
        compute_curvature_proxy,
    )

    timings = GPUStepTimings()
    meta = {}
    dev = state.device

    _maybe_cuda_synchronize(dev)

    # --- Kernel A (extended) ---
    t0 = time.perf_counter()
    ka_out = kernel_a_gpu_extended(state, obs_points, obs_normals, obs_confs, params)
    residuals = ka_out["residuals"]
    s_obs = ka_out["s_obs"]
    obs_count = ka_out["obs_count"]
    obs_weight_sum = ka_out["obs_weight_sum"]
    _maybe_cuda_synchronize(dev)
    timings.kernel_a_ms = (time.perf_counter() - t0) * 1000
    meta["residual_norm_mean"] = float(residuals.norm(dim=1).mean().item())
    meta["s_obs_mean"] = float(s_obs.mean().item())

    # --- Kernel B (hand-crafted, used as feature + blending anchor) ---
    t0 = time.perf_counter()
    S_handcrafted = kernel_b_gpu(state, s_obs, params)
    _maybe_cuda_synchronize(dev)
    timings.kernel_b_ms = (time.perf_counter() - t0) * 1000

    # --- Compute boundary flags (vectorized) ---
    boundary_flag = compute_boundary_flag_vectorized(
        state.faces, state.component_ids, state.N)

    # --- Build features (V2: includes S_handcrafted + pre-computed curvature) ---
    n_components = int(state.component_ids.max().item()) + 1
    curvature = compute_curvature_proxy(state.L_phi_csr, state.vertices)
    features = build_observability_features(
        residuals=residuals, s_obs=s_obs,
        obs_count=obs_count, obs_weight_sum=obs_weight_sum,
        support_prev=state.support,
        support_handcrafted=S_handcrafted,
        vertices=state.vertices,
        vertices_preop=state.vertices_preop,
        vertices_prev=vertices_prev, normals=state.normals,
        boundary_flag=boundary_flag,
        lesion_band_mask=state.lesion_band_mask,
        component_ids=state.component_ids,
        n_components=n_components, curvature=curvature,
    )

    # --- Predict gated support ---
    support_hat = obs_model(features, S_handcrafted)  # [N] in [0,1]
    meta["support_mean"] = float(support_hat.mean().item())
    meta["support_hat"] = support_hat
    meta["support_handcrafted_mean"] = float(S_handcrafted.mean().item())

    # --- Kernel C with learned support ---
    t0 = time.perf_counter()
    delta_V, solve_info = kernel_c_gpu(state, residuals, params,
                                        support_override=support_hat)
    _maybe_cuda_synchronize(dev)
    timings.kernel_c_ms = (time.perf_counter() - t0) * 1000
    meta["solve_info"] = solve_info
    meta["delta_V_norm_mean"] = float(delta_V.norm(dim=1).mean().item())

    # --- Support-aware alpha using learned support ---
    alpha_min = params.alpha * 0.3
    alpha_per_vertex = alpha_min + (params.alpha - alpha_min) * support_hat
    V_new = state.vertices + alpha_per_vertex.unsqueeze(1) * delta_V
    N_new = compute_vertex_normals_gpu(V_new, state.faces)

    # --- Kernel D (using V_new directly since we used per-vertex alpha) ---
    # Create a temporary state with V_new as vertices, pass zero delta_V and alpha=1
    state_for_d = GPUTwinState(
        vertices=V_new, faces=state.faces, normals=N_new,
        component_ids=state.component_ids, L_phi_csr=state.L_phi_csr,
        lesion_band_mask=state.lesion_band_mask,
        vertices_preop=state.vertices_preop, support=support_hat.detach(),
        tumor_bary_ids=state.tumor_bary_ids,
        tumor_bary_weights=state.tumor_bary_weights,
    )
    t0 = time.perf_counter()
    # Pass zero delta_V: kernel_d will compute V_new = state.vertices + alpha*0 = V_new
    lesion_result = kernel_d_gpu(state_for_d, torch.zeros_like(delta_V), params)
    _maybe_cuda_synchronize(dev)
    timings.kernel_d_ms = (time.perf_counter() - t0) * 1000
    meta["tumor_to_surface_dist"] = lesion_result["tumor_to_surface_dist"]
    meta["tumor_uncertainty"] = lesion_result["tumor_uncertainty"]

    timings.total_ms = (timings.kernel_a_ms + timings.kernel_b_ms
                        + timings.kernel_c_ms + timings.kernel_d_ms)

    # Build new state with learned support
    new_state = GPUTwinState(
        vertices=V_new,
        faces=state.faces,
        normals=N_new,
        component_ids=state.component_ids,
        L_phi_csr=state.L_phi_csr,
        lesion_band_mask=state.lesion_band_mask,
        vertices_preop=state.vertices_preop,
        support=support_hat.detach(),  # detach for state storage
        tumor_bary_ids=state.tumor_bary_ids,
        tumor_bary_weights=state.tumor_bary_weights,
        tumor_centroid=lesion_result["tumor_centroid"],
        tumor_vertices=lesion_result["tumor_vertices"],
        tumor_uncertainty=lesion_result["tumor_uncertainty"],
        tumor_to_surface_dist=lesion_result["tumor_to_surface_dist"],
    )

    # Store residuals and V_new in meta for loss computation
    meta["residuals"] = residuals
    meta["V_new"] = V_new
    meta["delta_V"] = delta_V

    return new_state, timings, meta


# ---------------------------------------------------------------------------
# Initialization: load Φ → GPU state
# ---------------------------------------------------------------------------

def init_twin_gpu(phi_dir: str, device: str = "cuda:0") -> GPUTwinState:
    """
    Load Φ outputs and build GPU twin state.
    """
    from pathlib import Path

    dev = torch.device(device)
    phi_path = Path(phi_dir)

    phi = np.load(phi_path / "phi_mesh.npz")
    L_phi_scipy = sparse.load_npz(phi_path / "L_phi.npz")

    V0 = torch.from_numpy(phi["vertices"].astype(np.float32)).to(dev)
    faces = torch.from_numpy(phi["faces"].astype(np.int64)).to(dev)
    normals = torch.from_numpy(phi["normals"].astype(np.float32)).to(dev)
    comp_ids = torch.from_numpy(phi["component_ids"].astype(np.int32)).to(dev)
    lesion_mask = torch.from_numpy(phi["lesion_band_mask"].astype(bool)).to(dev)

    L_phi_csr = scipy_csr_to_torch(L_phi_scipy, dev)

    # Tumor transport
    tumor_path = phi_path / "tumor_transport.npz"
    if tumor_path.exists():
        td = np.load(tumor_path)
        t_bary_ids = torch.from_numpy(td["barycentric_ids"].astype(np.int64)).to(dev)
        t_bary_wts = torch.from_numpy(td["barycentric_weights"].astype(np.float32)).to(dev)
        t_centroid = torch.from_numpy(td["tumor_centroid"].astype(np.float32)).to(dev)
    else:
        t_bary_ids = None
        t_bary_wts = None
        t_centroid = None

    return GPUTwinState(
        vertices=V0.clone(),
        faces=faces,
        normals=normals,
        component_ids=comp_ids,
        L_phi_csr=L_phi_csr,
        lesion_band_mask=lesion_mask,
        vertices_preop=V0,
        support=torch.zeros(len(V0), device=dev),
        tumor_bary_ids=t_bary_ids,
        tumor_bary_weights=t_bary_wts,
        tumor_centroid=t_centroid,
    )


# ---------------------------------------------------------------------------
# Run SSCU on benchmark sequence (GPU)
# ---------------------------------------------------------------------------

def run_sscu_sequence_gpu(phi_dir: str, benchmark_dir: str, coverage: float,
                          params: GPUSSCUParams | None = None,
                          device: str = "cuda:0",
                          obs_model: "torch.nn.Module | None" = None) -> dict:
    """
    Run GPU SSCU on a full benchmark sequence.
    """
    import json
    from pathlib import Path
    from .sscu_engine import compute_leakage_rate

    if params is None:
        params = GPUSSCUParams()

    dev = torch.device(device)
    bench = Path(benchmark_dir)
    cov_str = f"cov{int(coverage * 100):02d}"
    cov_dir = bench / cov_str

    with open(cov_dir / "sequence_meta.json") as f:
        seq_meta = json.load(f)
    n_frames = seq_meta["n_frames"]

    log.info(f"[GPU] Initializing twin from {phi_dir}")
    state = init_twin_gpu(phi_dir, device=device)
    log.info(f"  Vertices: {state.N}, device: {state.device}")

    # Warmup: one dummy SpMV to JIT-compile sparse kernels
    _ = torch.mv(state.L_phi_csr, state.support)
    _maybe_cuda_synchronize(dev)

    V_preop = state.vertices_preop.clone()

    all_metrics = []
    vertices_prev = None
    for t in range(n_frames):
        frame = np.load(cov_dir / f"frame_{t:03d}.npz")

        obs_pts = torch.from_numpy(frame["obs_points"]).float().to(dev)
        obs_nrm = torch.from_numpy(frame["obs_normals"]).float().to(dev)
        obs_conf = torch.from_numpy(frame["obs_confidences"]).float().to(dev)

        if params.use_learned_observability:
            if obs_model is None:
                raise ValueError(
                    "params.use_learned_observability=True requires obs_model in "
                    "run_sscu_sequence_gpu()."
                )
            state, timings, meta = scsu_update_gpu(
                state,
                obs_pts,
                obs_nrm,
                obs_conf,
                params,
                obs_model=obs_model,
                vertices_prev=vertices_prev,
            )
        else:
            state, timings, meta = sscu_update_gpu(state, obs_pts, obs_nrm, obs_conf, params)

        # Ground truth
        V_gt = torch.from_numpy(frame["vertices_gt"]).float().to(dev)

        # --- Comprehensive diagnostics ---
        vertices_eval = state.vertices.detach()
        support_eval = state.support.detach()
        disp = vertices_eval - V_preop
        surface_chamfer = float((vertices_eval - V_gt).norm(dim=1).mean().item())
        prior_only_chamfer = float((V_preop - V_gt).norm(dim=1).mean().item())
        relative_improvement = (prior_only_chamfer - surface_chamfer) / (prior_only_chamfer + 1e-12)
        disp_norm = disp.norm(dim=1)
        support_gt0 = float((support_eval > 0.01).float().mean().item())
        support_gt50 = float((support_eval > 0.5).float().mean().item())
        prior_data_ratio = float(
            (params.lambda_prior * (1.0 - support_eval.mean().item())) /
            (support_eval.mean().item() + 1e-12)
        )

        frame_metrics = {
            "frame": t,
            "surface_chamfer_mm": surface_chamfer,
            "prior_only_chamfer_mm": prior_only_chamfer,
            "relative_improvement": relative_improvement,
            "support_mean": meta["support_mean"],
            "support_max": float(support_eval.max().item()),
            "support_gt001_frac": support_gt0,
            "support_gt050_frac": support_gt50,
            "prior_data_ratio": prior_data_ratio,
            "delta_V_norm_mean": meta["delta_V_norm_mean"],
            "cumulative_disp_mean_mm": float(disp_norm.mean().item()),
            "cumulative_disp_max_mm": float(disp_norm.max().item()),
            "residual_norm_mean": meta["residual_norm_mean"],
            "s_obs_mean": meta["s_obs_mean"],
            "kernel_a_ms": timings.kernel_a_ms,
            "kernel_b_ms": timings.kernel_b_ms,
            "kernel_c_ms": timings.kernel_c_ms,
            "kernel_d_ms": timings.kernel_d_ms,
            "total_ms": timings.total_ms,
        }
        if "support_handcrafted_mean" in meta:
            frame_metrics["support_handcrafted_mean"] = meta["support_handcrafted_mean"]

        if "tumor_centroid_gt" in frame:
            tc_gt = torch.from_numpy(frame["tumor_centroid_gt"]).float().to(dev)
            if state.tumor_centroid is not None:
                tc_err = float((state.tumor_centroid.detach() - tc_gt).norm().item())
                frame_metrics["tumor_centroid_error_mm"] = tc_err
            frame_metrics["tumor_to_surface_dist"] = meta["tumor_to_surface_dist"]
            frame_metrics["tumor_uncertainty"] = meta["tumor_uncertainty"]

        # Leakage (CPU)
        disp_cpu = disp.cpu().numpy()
        faces_cpu = state.faces.detach().cpu().numpy()
        comp_cpu = state.component_ids.detach().cpu().numpy()
        frame_metrics["leakage_rate"] = compute_leakage_rate(disp_cpu, faces_cpu, comp_cpu)

        all_metrics.append(frame_metrics)

        log.info(
            f"  t={t:2d}: Chamfer={surface_chamfer:.3f} prior={prior_only_chamfer:.3f} "
            f"imp={relative_improvement*100:+.1f}% "
            f"S={meta['support_mean']:.3f}(max={support_eval.max().item():.3f}) "
            f"|ΔV|={meta['delta_V_norm_mean']:.4f} "
            f"|disp|={disp_norm.mean().item():.4f} "
            f"P/D={prior_data_ratio:.1f} "
            f"leak={frame_metrics['leakage_rate']:.4f} "
            f"{timings.total_ms:.1f}ms"
            + (f" T_err={frame_metrics.get('tumor_centroid_error_mm',0):.3f}"
               if "tumor_centroid_error_mm" in frame_metrics else "")
        )
        vertices_prev = state.vertices.clone()

    return {
        "coverage": coverage,
        "n_frames": n_frames,
        "device": device,
        "params": {k: getattr(params, k) for k in [
            "lambda_sem", "lambda_prior", "lambda_les", "alpha",
            "sigma_mm", "gamma", "diffusion_eta", "diffusion_iters", "tau_vis",
        ]},
        "frames": all_metrics,
    }
