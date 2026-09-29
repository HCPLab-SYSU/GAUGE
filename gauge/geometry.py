"""Geometry helpers used by the GAUGE correction module."""

from __future__ import annotations

import numpy as np


def compute_pred_parallax_per_query(
    pred_m: np.ndarray,
    pred_visible: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Return a ground-truth-free per-query parallax proxy, in radians.

    The value reported for a query is the maximum angular separation between the
    viewing rays of any two frames where the query is predicted to be visible. It
    is a property of the prediction alone, so it can be computed without ground
    truth and is used to route low-parallax groups to the numerically stable
    anchor-centered form.

    Args:
        pred_m: [T, N, 3] predicted trajectories.
        pred_visible: [T, N] bool tracker-predicted visibility.
        centers: [T, 3] per-frame camera centers.

    Returns:
        [N] array of parallax angles in radians; queries with fewer than two
        visible frames get 0.
    """
    T, N = pred_m.shape[:2]
    parallax = np.full(N, np.nan, dtype=np.float64)
    for q in range(N):
        frames = np.nonzero(pred_visible[:, q])[0]
        if frames.size < 2:
            continue
        rays = pred_m[frames, q] - centers[frames]
        norms = np.linalg.norm(rays, axis=1)
        valid = norms > 1e-9
        if not valid.any():
            continue
        rays = rays[valid] / norms[valid, None]
        if rays.shape[0] < 2:
            continue
        cos = np.clip(rays @ rays.T, -1.0, 1.0)
        angles = np.arccos(cos)
        np.fill_diagonal(angles, 0.0)
        parallax[q] = float(np.max(angles))
    return np.nan_to_num(parallax, nan=0.0)
