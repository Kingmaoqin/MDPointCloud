# NIPSCODEPC

Cleaned core implementation for the paper's main algorithm:

**Structured safe updating for online anatomical state maintenance under partial intraoperative observations.**

This folder intentionally excludes experiment launchers, ablation sweeps, plotting scripts, result aggregation, and review-response code. It contains only the algorithm components needed to run the online updater.

## Files

- `nipscodepc/safe_update.py`
  Main algorithm entry point. `StructuredSafeUpdater.step(...)` consumes one partial observation frame and updates the anatomical state.

- `nipscodepc/observability_model.py`
  Learned trust / observability model, frame-level lattice blend head, and feature builders.

- `nipscodepc/sscu_engine_gpu.py`
  GPU structured solver: residual support update, semantic-Laplacian system, PCG solve, lesion transport utilities, and phi-state loading.

- `nipscodepc/lattice.py`
  Coarse volumetric lattice construction, projection, interpolation, and elastic lattice solve.

- `nipscodepc/anisotropic_residuals.py`
  Observation residual baking with optional tangential correction.

- `nipscodepc/build_phi.py`
  Preoperative phi/twin construction from segmentation. Included because the online updater expects the phi directory produced by this step.

## Minimal Use

```python
import numpy as np
from nipscodepc import StructuredSafeUpdater, load_hybrid_model

model, cfg = load_hybrid_model(
    "/path/to/hybrid_run_dir_or_checkpoint",
    device="cuda:0",
)
updater = StructuredSafeUpdater.from_phi(
    "/path/to/case_xxxx/phi",
    model=model,
    device="cuda:0",
    config=cfg,
)

frame = np.load("/path/to/frame_000.npz")
state, info = updater.step(
    frame["obs_points"],
    frame["obs_normals"],
    frame["obs_confidences"],
)

vertices_updated = state.vertices
tumor_centroid = info.get("tumor_centroid")
support_hat = info["support_hat"]
```

## Notes

- Default `SafeUpdateConfig` is the balanced operating point: trust-controlled solver, lattice stabilization, frame gating, lattice decay, and preoperative pullback.
- No benchmark metrics or held-out split logic are included here.
- Source provenance:
  - Final model/solver code: `md/test_hybrid_lattice_v1`
  - Phi builder: `scripts/nips_sscu/step1_build_phi.py`
