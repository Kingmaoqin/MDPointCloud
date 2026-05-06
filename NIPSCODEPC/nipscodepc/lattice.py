"""Coarse volumetric lattice stabilizer used by the structured safe updater."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def build_grid_laplacian(nx: int, ny: int, nz: int) -> sparse.csr_matrix:
    """Build a 6-neighbor graph Laplacian over an nx-by-ny-by-nz lattice."""

    def idx(ix: int, iy: int, iz: int) -> int:
        return ix + nx * (iy + ny * iz)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros(nx * ny * nz, dtype=np.float32)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                i = idx(ix, iy, iz)
                for dx, dy, dz in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ):
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                        j = idx(jx, jy, jz)
                        rows.append(i)
                        cols.append(j)
                        data.append(-1.0)
                        degree[i] += 1.0

    rows.extend(np.arange(nx * ny * nz).tolist())
    cols.extend(np.arange(nx * ny * nz).tolist())
    data.extend(degree.tolist())
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(nx * ny * nz, nx * ny * nz),
        dtype=np.float32,
    )


def build_vertex_to_grid_weights(
    vertices: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    margin_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lattice node positions plus trilinear node ids/weights per vertex."""

    vmin = vertices.min(axis=0) - margin_mm
    vmax = vertices.max(axis=0) + margin_mm
    extent = np.maximum(vmax - vmin, 1e-6)

    grid_pos = np.stack(
        np.meshgrid(
            np.linspace(vmin[0], vmax[0], nx, dtype=np.float32),
            np.linspace(vmin[1], vmax[1], ny, dtype=np.float32),
            np.linspace(vmin[2], vmax[2], nz, dtype=np.float32),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 3)

    scaled = (vertices - vmin[None, :]) / extent[None, :]
    gx = scaled[:, 0] * (nx - 1)
    gy = scaled[:, 1] * (ny - 1)
    gz = scaled[:, 2] * (nz - 1)

    ix0 = np.clip(np.floor(gx).astype(np.int64), 0, nx - 2)
    iy0 = np.clip(np.floor(gy).astype(np.int64), 0, ny - 2)
    iz0 = np.clip(np.floor(gz).astype(np.int64), 0, nz - 2)
    tx = (gx - ix0).astype(np.float32)
    ty = (gy - iy0).astype(np.float32)
    tz = (gz - iz0).astype(np.float32)

    def nid(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        return ix + nx * (iy + ny * iz)

    node_ids = np.stack(
        [
            nid(ix0, iy0, iz0),
            nid(ix0 + 1, iy0, iz0),
            nid(ix0, iy0 + 1, iz0),
            nid(ix0 + 1, iy0 + 1, iz0),
            nid(ix0, iy0, iz0 + 1),
            nid(ix0 + 1, iy0, iz0 + 1),
            nid(ix0, iy0 + 1, iz0 + 1),
            nid(ix0 + 1, iy0 + 1, iz0 + 1),
        ],
        axis=1,
    ).astype(np.int64)

    weights = np.stack(
        [
            (1 - tx) * (1 - ty) * (1 - tz),
            tx * (1 - ty) * (1 - tz),
            (1 - tx) * ty * (1 - tz),
            tx * ty * (1 - tz),
            (1 - tx) * (1 - ty) * tz,
            tx * (1 - ty) * tz,
            (1 - tx) * ty * tz,
            tx * ty * tz,
        ],
        axis=1,
    ).astype(np.float32)

    return grid_pos, node_ids, weights


def apply_grid_displacement(
    node_disp: np.ndarray,
    node_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Interpolate lattice node displacements back to vertices."""

    return np.sum(node_disp[node_ids] * weights[..., None], axis=1)


def project_vertex_values_to_grid(
    values: np.ndarray,
    vertex_weights: np.ndarray,
    node_ids: np.ndarray,
    interp_w: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter weighted per-vertex vectors to lattice nodes."""

    acc_v = np.zeros((n_nodes, values.shape[1]), dtype=np.float64)
    acc_w = np.zeros(n_nodes, dtype=np.float64)
    weighted = vertex_weights.astype(np.float64)

    for k in range(8):
        ids = node_ids[:, k]
        w = weighted * interp_w[:, k]
        np.add.at(acc_v, ids, w[:, None] * values)
        np.add.at(acc_w, ids, w)

    out = acc_v / (acc_w[:, None] + 1e-8)
    return out.astype(np.float32), acc_w.astype(np.float32)


def solve_grid_update(
    node_disp: np.ndarray,
    residual_nodes: np.ndarray,
    obs_weight_nodes: np.ndarray,
    grid_laplacian: sparse.csr_matrix,
    lesion_weight_nodes: np.ndarray,
    lambda_elastic: float,
    lambda_prior: float,
    lambda_lesion: float,
) -> np.ndarray:
    """Solve the low-frequency lattice update."""

    n = len(node_disp)
    system = (
        sparse.diags(obs_weight_nodes + lambda_prior + lambda_lesion * lesion_weight_nodes, format="csr")
        + lambda_elastic * grid_laplacian
    )
    rhs = obs_weight_nodes[:, None] * residual_nodes - lambda_prior * node_disp
    delta = np.zeros_like(node_disp, dtype=np.float32)

    for d in range(3):
        x, info = cg(system, rhs[:, d].astype(np.float64), rtol=1e-5, atol=0.0, maxiter=100)
        if info != 0:
            x = np.zeros(n, dtype=np.float64)
        delta[:, d] = x.astype(np.float32)

    return delta
