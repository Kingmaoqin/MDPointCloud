#!/usr/bin/env python3
"""
Step 1: Build medical semantic partition Φ from KiTS23 kidney+tumor segmentation.

Converts a preoperative CT segmentation into the structured twin state:
    Φ = (C, E, y)
where:
    C = set of surface components (6-12 sextants + tumor-adjacent band + optional hilum)
    E = within-component adjacency edges
    y : V → C  maps each mesh vertex to its component ID

Pipeline:
    1. Load KiTS23 segmentation (kidney=1, tumor=2, cyst=3)
    2. Marching cubes → surface mesh M_0 = (V, F)
    3. PCA alignment along kidney principal axes
    4. Sextant partition: split along 3 PCA axes → 6-12 components
    5. Geodesic distance → lesion-adjacent band (d_les mm ring around tumor)
    6. Build semantic Laplacian L_Φ (CSR, within-component edges only)
    7. Export Φ as JSON + NPZ

Reference: SSCU_Implementation_Plan.md, Step 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import sparse
from skimage import measure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KiTS23 label conventions
# ---------------------------------------------------------------------------
KITS_KIDNEY = 1
KITS_TUMOR = 2
KITS_CYST = 3


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------

def extract_mesh(mask: np.ndarray, spacing_mm: np.ndarray, label_ids: list[int],
                 step_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes on binary mask → vertices (mm), faces."""
    binary = np.isin(mask, label_ids).astype(np.uint8)
    if binary.sum() < 50:
        raise ValueError(f"Mask too small ({binary.sum()} voxels) for marching cubes")
    verts_vox, faces, normals, _ = measure.marching_cubes(
        binary, level=0.5, spacing=spacing_mm.tolist(), step_size=step_size
    )
    return verts_vox.astype(np.float32), faces.astype(np.int32)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from mesh."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    face_normals /= norms

    vertex_normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vertex_normals, faces[:, k], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    vertex_normals /= norms
    return vertex_normals.astype(np.float32)


# ---------------------------------------------------------------------------
# PCA alignment
# ---------------------------------------------------------------------------

def pca_align(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-align vertices.
    Returns: (aligned_verts, centroid, principal_axes [3×3, rows=axes])
    """
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending by eigenvalue
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # [3, 3], rows = principal axes
    aligned = centered @ axes.T
    return aligned.astype(np.float32), centroid.astype(np.float32), axes.astype(np.float32)


# ---------------------------------------------------------------------------
# Sextant partition
# ---------------------------------------------------------------------------

def sextant_partition(aligned_verts: np.ndarray, n_divisions: int = 2) -> np.ndarray:
    """
    Partition aligned vertices into sextants along PCA axes.

    n_divisions=2 → 2^3 = 8 octants (sextants).
    n_divisions=3 → 3^3 = 27 fine blocks.

    Each vertex gets component_id = ix * n^2 + iy * n + iz.
    """
    n_verts = len(aligned_verts)
    comp_ids = np.zeros(n_verts, dtype=np.int32)

    for axis in range(3):
        coords = aligned_verts[:, axis]
        # Use quantile-based splits for balanced partition
        quantiles = np.linspace(0, 1, n_divisions + 1)[1:-1]
        thresholds = np.quantile(coords, quantiles)
        axis_bin = np.digitize(coords, thresholds)  # 0..n_divisions-1
        comp_ids = comp_ids * n_divisions + axis_bin

    return comp_ids


# ---------------------------------------------------------------------------
# Tumor proximity and lesion band
# ---------------------------------------------------------------------------

def compute_tumor_vertices(verts_mm: np.ndarray, mask: np.ndarray,
                           spacing_mm: np.ndarray, origin_mm: np.ndarray) -> np.ndarray:
    """Find mesh vertices that fall within (or near) the tumor region."""
    # Convert world coords to voxel coords
    vox_coords = (verts_mm - origin_mm) / spacing_mm
    vox_idx = np.round(vox_coords).astype(int)

    # Clamp to valid range
    for d in range(3):
        vox_idx[:, d] = np.clip(vox_idx[:, d], 0, mask.shape[d] - 1)

    is_tumor = (mask[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] == KITS_TUMOR)
    return is_tumor


def geodesic_distance_approx(verts: np.ndarray, faces: np.ndarray,
                             source_mask: np.ndarray,
                             max_dist: float = 30.0) -> np.ndarray:
    """
    Approximate geodesic distance from source vertices via BFS on mesh edges.
    Uses edge-length-weighted Dijkstra (approximation to true geodesic).

    Returns per-vertex distance (mm). Non-reached vertices get max_dist.
    """
    import heapq

    n_verts = len(verts)

    # Build adjacency list with edge lengths
    adj = [[] for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            d = float(np.linalg.norm(verts[a] - verts[b]))
            adj[a].append((b, d))
            adj[b].append((a, d))

    # Dijkstra from all source vertices
    dist = np.full(n_verts, max_dist, dtype=np.float32)
    heap = []
    source_indices = np.where(source_mask)[0]
    for s in source_indices:
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, int(s)))

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue
        if d_u > max_dist:
            break
        for v, w in adj[u]:
            d_v = d_u + w
            if d_v < dist[v]:
                dist[v] = d_v
                heapq.heappush(heap, (d_v, v))

    return dist


def assign_lesion_band(comp_ids: np.ndarray, geodesic_dist: np.ndarray,
                       d_les: float = 15.0,
                       lesion_band_id: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign vertices within d_les mm of tumor to a dedicated lesion-adjacent band.

    Returns:
        comp_ids_updated: component IDs with lesion band
        lesion_band_mask: boolean mask of lesion-adjacent vertices
    """
    comp_ids_updated = comp_ids.copy()
    lesion_band_mask = geodesic_dist <= d_les

    # Assign a unique component ID for the lesion band
    if lesion_band_id < 0:
        lesion_band_id = comp_ids.max() + 1
    comp_ids_updated[lesion_band_mask] = lesion_band_id

    return comp_ids_updated, lesion_band_mask


# ---------------------------------------------------------------------------
# Mesh adjacency and semantic Laplacian
# ---------------------------------------------------------------------------

def build_mesh_adjacency(faces: np.ndarray, n_verts: int) -> sparse.csr_matrix:
    """Build symmetric adjacency matrix from faces."""
    rows, cols = [], []
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            rows.extend([a, b])
            cols.extend([b, a])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    # Remove duplicates
    adj.data[:] = 1.0
    return adj


def build_semantic_laplacian(faces: np.ndarray, comp_ids: np.ndarray,
                             n_verts: int) -> sparse.csr_matrix:
    """
    Build the semantic Laplacian L_Φ where edges only connect vertices
    within the same component (y(i) == y(k)).

    L_Φ = D - W  (graph Laplacian, restricted to within-component edges)

    This is the key mathematical structure enabling Theorem 1 (no cross-region leakage):
    when ordered by component, L_Φ becomes block-diagonal.
    """
    rows, cols, vals = [], [], []

    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            if comp_ids[a] == comp_ids[b]:
                rows.extend([a, b])
                cols.extend([b, a])
                vals.extend([1.0, 1.0])

    if not rows:
        return sparse.csr_matrix((n_verts, n_verts), dtype=np.float32)

    # Adjacency (within-component)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_verts, n_verts), dtype=np.float32)
    W.data[:] = 1.0  # deduplicate

    # Degree
    D = sparse.diags(np.array(W.sum(axis=1)).ravel(), format="csr", dtype=np.float32)

    # Laplacian
    L = D - W
    return L


# ---------------------------------------------------------------------------
# Tumor mesh extraction (for lesion transport, Kernel D)
# ---------------------------------------------------------------------------

def extract_tumor_mesh(mask: np.ndarray, spacing_mm: np.ndarray,
                       step_size: int = 2) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract tumor surface mesh for lesion boundary B_0."""
    tumor_mask = (mask == KITS_TUMOR).astype(np.uint8)
    if tumor_mask.sum() < 30:
        log.warning("Tumor region too small for mesh extraction")
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            tumor_mask, level=0.5, spacing=spacing_mm.tolist(), step_size=step_size
        )
        return verts.astype(np.float32), faces.astype(np.int32)
    except Exception as e:
        log.warning(f"Tumor marching cubes failed: {e}")
        return None


def compute_barycentric_weights(tumor_verts: np.ndarray,
                                kidney_verts: np.ndarray,
                                k_neighbors: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    For each tumor boundary point, compute barycentric-style weights
    w.r.t. nearest kidney surface vertices (for lesion transport, Theorem 3).

    Returns:
        neighbor_ids: [n_tumor, k] int array of kidney vertex indices
        weights: [n_tumor, k] float array, sum=1 per row
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(kidney_verts)
    dists, ids = tree.query(tumor_verts, k=k_neighbors)

    # Inverse-distance weights, normalized
    eps = 1e-6
    inv_dists = 1.0 / (dists + eps)
    weights = inv_dists / inv_dists.sum(axis=1, keepdims=True)

    return ids.astype(np.int32), weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Main Φ builder
# ---------------------------------------------------------------------------

def build_phi(seg_path: str, output_dir: str,
              n_divisions: int = 2,
              d_les: float = 15.0,
              mesh_step: int = 1,
              max_geodesic: float = 30.0) -> dict:
    """
    Build semantic partition Φ from a KiTS23 segmentation volume.

    Args:
        seg_path: path to segmentation NIfTI (labels: 1=kidney, 2=tumor, 3=cyst)
        output_dir: directory to save outputs
        n_divisions: PCA axis divisions (2→8 sextants, 3→27 blocks)
        d_les: lesion band radius in mm
        mesh_step: marching cubes step size (1=fine, 2=coarse)
        max_geodesic: max geodesic distance for Dijkstra

    Returns:
        dict with summary statistics
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load segmentation ----
    log.info(f"Loading segmentation: {seg_path}")
    nii = nib.load(seg_path)
    mask = np.asarray(nii.dataobj).astype(np.int16)
    spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
    affine = nii.affine

    # Origin in mm (translation column of affine)
    origin_mm = affine[:3, 3].astype(np.float32)

    log.info(f"  Volume shape: {mask.shape}, spacing: {spacing} mm")
    log.info(f"  Kidney voxels: {(mask == KITS_KIDNEY).sum()}")
    log.info(f"  Tumor voxels: {(mask == KITS_TUMOR).sum()}")
    log.info(f"  Cyst voxels: {(mask == KITS_CYST).sum()}")

    has_tumor = (mask == KITS_TUMOR).sum() > 30

    # ---- Extract kidney surface mesh ----
    log.info("Extracting kidney surface mesh (marching cubes)...")
    kidney_labels = [KITS_KIDNEY, KITS_TUMOR, KITS_CYST]  # combined kidney region
    verts, faces = extract_mesh(mask, spacing, kidney_labels, step_size=mesh_step)
    # Shift to world coordinates
    verts_mm = verts + origin_mm
    log.info(f"  Mesh: {len(verts)} vertices, {len(faces)} faces")

    # ---- Vertex normals ----
    normals = compute_vertex_normals(verts_mm, faces)

    # ---- PCA alignment ----
    log.info("PCA-aligning kidney mesh...")
    aligned, centroid, pca_axes = pca_align(verts_mm)
    log.info(f"  Centroid: {centroid}")

    # ---- Sextant partition ----
    log.info(f"Partitioning into sextants (n_divisions={n_divisions})...")
    comp_ids = sextant_partition(aligned, n_divisions=n_divisions)
    n_comps_base = len(np.unique(comp_ids))
    log.info(f"  Base components: {n_comps_base}")

    # ---- Tumor identification ----
    if has_tumor:
        log.info("Identifying tumor vertices on mesh...")
        is_tumor = compute_tumor_vertices(verts_mm, mask, spacing, origin_mm)
        n_tumor_verts = is_tumor.sum()
        log.info(f"  Tumor-touched vertices: {n_tumor_verts}")

        if n_tumor_verts > 0:
            # ---- Geodesic distance from tumor ----
            log.info(f"Computing geodesic distance from tumor (max={max_geodesic}mm)...")
            geo_dist = geodesic_distance_approx(verts_mm, faces, is_tumor,
                                                max_dist=max_geodesic)
            log.info(f"  Vertices within {d_les}mm of tumor: {(geo_dist <= d_les).sum()}")

            # ---- Assign lesion band ----
            lesion_band_id = n_comps_base  # next available ID
            comp_ids, lesion_band_mask = assign_lesion_band(
                comp_ids, geo_dist, d_les=d_les, lesion_band_id=lesion_band_id
            )
            log.info(f"  Lesion band (comp {lesion_band_id}): {lesion_band_mask.sum()} vertices")
        else:
            geo_dist = np.full(len(verts_mm), max_geodesic, dtype=np.float32)
            lesion_band_mask = np.zeros(len(verts_mm), dtype=bool)
    else:
        log.info("No tumor in this case, skipping lesion band")
        geo_dist = np.full(len(verts_mm), max_geodesic, dtype=np.float32)
        lesion_band_mask = np.zeros(len(verts_mm), dtype=bool)
        is_tumor = np.zeros(len(verts_mm), dtype=bool)

    # ---- Semantic Laplacian L_Φ ----
    log.info("Building semantic Laplacian L_Φ...")
    L_phi = build_semantic_laplacian(faces, comp_ids, len(verts_mm))
    log.info(f"  L_Φ shape: {L_phi.shape}, nnz: {L_phi.nnz}")

    # Verify block-diagonality
    unique_comps = np.unique(comp_ids)
    n_cross_edges = 0
    adj = build_mesh_adjacency(faces, len(verts_mm))
    rows, cols = adj.nonzero()
    for r, c in zip(rows, cols):
        if comp_ids[r] != comp_ids[c]:
            n_cross_edges += 1
    log.info(f"  Cross-component edges in full adj: {n_cross_edges // 2}")
    log.info(f"  L_Φ has ZERO cross-component edges (by construction)")

    # ---- Tumor mesh for lesion transport ----
    tumor_bary_ids, tumor_bary_weights = None, None
    tumor_verts, tumor_faces = None, None
    if has_tumor:
        log.info("Extracting tumor boundary mesh...")
        result = extract_tumor_mesh(mask, spacing, step_size=max(mesh_step, 2))
        if result is not None:
            tumor_verts, tumor_faces = result
            tumor_verts_mm = tumor_verts + origin_mm
            log.info(f"  Tumor mesh: {len(tumor_verts)} verts, {len(tumor_faces)} faces")

            # Barycentric weights for lesion transport
            log.info("Computing barycentric weights for lesion transport...")
            tumor_bary_ids, tumor_bary_weights = compute_barycentric_weights(
                tumor_verts_mm, verts_mm, k_neighbors=4
            )
            tumor_centroid = tumor_verts_mm.mean(axis=0)
            log.info(f"  Tumor centroid: {tumor_centroid}")
        else:
            tumor_verts_mm = None

    # ---- Save outputs ----
    log.info(f"Saving Φ to {out}/...")

    # 1. Main NPZ: mesh + partition + preop state V_0
    np.savez_compressed(
        out / "phi_mesh.npz",
        vertices=verts_mm,          # V_0 (preop surface, mm) [N, 3]
        faces=faces,                # F [M, 3]
        normals=normals,            # vertex normals [N, 3]
        component_ids=comp_ids,     # y: V → C [N]
        lesion_band_mask=lesion_band_mask,  # [N] bool
        geodesic_dist_to_tumor=geo_dist,    # [N] float32
        is_tumor_vertex=is_tumor if has_tumor else np.zeros(len(verts_mm), dtype=bool),
        pca_centroid=centroid,      # [3]
        pca_axes=pca_axes,          # [3, 3]
    )

    # 2. Semantic Laplacian L_Φ (CSR)
    sparse.save_npz(out / "L_phi.npz", L_phi)

    # 3. Tumor transport data
    if tumor_bary_ids is not None:
        np.savez_compressed(
            out / "tumor_transport.npz",
            tumor_vertices=tumor_verts_mm,
            tumor_faces=tumor_faces,
            barycentric_ids=tumor_bary_ids,       # [n_tumor, k]
            barycentric_weights=tumor_bary_weights,  # [n_tumor, k]
            tumor_centroid=tumor_verts_mm.mean(axis=0),
        )

    # 4. Summary JSON
    summary = {
        "seg_path": str(seg_path),
        "n_vertices": int(len(verts_mm)),
        "n_faces": int(len(faces)),
        "n_components": int(len(np.unique(comp_ids))),
        "component_sizes": {int(c): int((comp_ids == c).sum()) for c in np.unique(comp_ids)},
        "n_divisions": n_divisions,
        "d_les_mm": d_les,
        "has_tumor": bool(has_tumor),
        "n_lesion_band_vertices": int(lesion_band_mask.sum()),
        "L_phi_nnz": int(L_phi.nnz),
        "pca_centroid_mm": centroid.tolist(),
        "spacing_mm": spacing.tolist(),
    }
    if tumor_bary_ids is not None:
        summary["n_tumor_boundary_points"] = int(len(tumor_bary_ids))
        summary["tumor_centroid_mm"] = tumor_verts_mm.mean(axis=0).tolist()

    with open(out / "phi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Φ construction complete.")
    log.info(f"  Components: {summary['n_components']}, "
             f"Lesion band: {summary['n_lesion_band_vertices']} verts, "
             f"L_Φ nnz: {summary['L_phi_nnz']}")

    return summary


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def find_kits23_cases(data_root: str) -> list[tuple[str, str]]:
    """Find KiTS23 cases with segmentation files."""
    root = Path(data_root)
    cases = []

    # KiTS23 directory structure: case_XXXXX/segmentation.nii.gz
    for case_dir in sorted(root.glob("case_*")):
        seg = case_dir / "segmentation.nii.gz"
        if seg.exists():
            cases.append((case_dir.name, str(seg)))

    log.info(f"Found {len(cases)} KiTS23 cases with segmentations")
    return cases


def batch_build_phi(data_root: str, output_root: str,
                    n_divisions: int = 2, d_les: float = 15.0,
                    max_cases: int = 0, mesh_step: int = 1) -> None:
    """Batch-process KiTS23 cases to build Φ for each."""
    cases = find_kits23_cases(data_root)
    if max_cases > 0:
        cases = cases[:max_cases]

    out_root = Path(output_root)
    results = []

    for i, (case_id, seg_path) in enumerate(cases):
        log.info(f"\n{'='*60}")
        log.info(f"Processing {case_id} ({i+1}/{len(cases)})")
        log.info(f"{'='*60}")

        case_out = out_root / case_id
        if (case_out / "phi_summary.json").exists():
            log.info(f"  Skipping (already exists)")
            with open(case_out / "phi_summary.json") as f:
                results.append(json.load(f))
            continue

        try:
            summary = build_phi(
                seg_path=seg_path,
                output_dir=str(case_out),
                n_divisions=n_divisions,
                d_les=d_les,
                mesh_step=mesh_step,
            )
            results.append(summary)
        except Exception as e:
            log.error(f"  FAILED: {e}")
            results.append({"case_id": case_id, "error": str(e)})

    # Save batch summary
    with open(out_root / "batch_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if "error" not in r)
    log.info(f"\nBatch complete: {n_ok}/{len(results)} succeeded")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build medical semantic partition Φ")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Single case
    p_single = sub.add_parser("single", help="Process a single case")
    p_single.add_argument("seg_path", help="Path to segmentation NIfTI")
    p_single.add_argument("--output-dir", required=True, help="Output directory")
    p_single.add_argument("--n-divisions", type=int, default=2, help="PCA axis divisions (2→8 sextants)")
    p_single.add_argument("--d-les", type=float, default=15.0, help="Lesion band radius (mm)")
    p_single.add_argument("--mesh-step", type=int, default=1, help="Marching cubes step size")

    # Batch KiTS23
    p_batch = sub.add_parser("batch", help="Batch process KiTS23 dataset")
    p_batch.add_argument("data_root", help="KiTS23 dataset root")
    p_batch.add_argument("--output-root", required=True, help="Output root directory")
    p_batch.add_argument("--n-divisions", type=int, default=2)
    p_batch.add_argument("--d-les", type=float, default=15.0)
    p_batch.add_argument("--max-cases", type=int, default=0, help="Max cases (0=all)")
    p_batch.add_argument("--mesh-step", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "single":
        build_phi(args.seg_path, args.output_dir,
                  n_divisions=args.n_divisions, d_les=args.d_les,
                  mesh_step=args.mesh_step)
    elif args.mode == "batch":
        batch_build_phi(args.data_root, args.output_root,
                        n_divisions=args.n_divisions, d_les=args.d_les,
                        max_cases=args.max_cases, mesh_step=args.mesh_step)


if __name__ == "__main__":
    main()
