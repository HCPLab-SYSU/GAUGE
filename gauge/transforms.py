"""Low-degree-of-freedom geometric transforms used by the correction module.

Every estimator here returns None instead of raising when the anchors do not
determine the transform, so that the caller can fall back to a coarser form.
"""

from __future__ import annotations

import numpy as np


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Estimate 7-DoF Sim3 from src to dst via Umeyama's method.

    Args:
        src, dst: [M, 3] corresponding points, M >= 3.

    Returns:
        (scale, rotation, translation) or None if degenerate.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3 or src.shape != dst.shape:
        return None

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma2_src = (src_c ** 2).sum() / src.shape[0]
    if sigma2_src < 1e-12:
        return None

    H = dst_c.T @ src_c / src.shape[0]
    U, D, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        S = np.diag([1.0, 1.0, -1.0])
        R = U @ S @ Vt

    D = np.diag(D)
    scale = (np.trace(D) / sigma2_src).item()
    if not np.isfinite(scale) or scale <= 1e-8:
        return None

    t = mu_dst - scale * R @ mu_src
    return float(scale), R.astype(np.float64), t.astype(np.float64)


def estimate_scale_about_camera_center(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    camera_center: np.ndarray,
    min_ratio: float = 1e-3,
) -> tuple[np.ndarray, float] | None:
    """Estimate a single radial scale about the camera center.

    Returns (camera_center, alpha) or None if degenerate.
    """
    X_pred = np.asarray(X_pred, dtype=np.float64)
    X_true = np.asarray(X_true, dtype=np.float64)
    if X_pred.shape[0] < 1 or X_pred.shape != X_true.shape or X_pred.shape[1] != 3:
        return None
    camera_center = np.asarray(camera_center, dtype=np.float64).reshape(3)

    d_pred = X_pred - camera_center
    d_true = X_true - camera_center
    n_pred = np.linalg.norm(d_pred, axis=1)
    n_true = np.linalg.norm(d_true, axis=1)
    valid = (n_pred > min_ratio) & np.isfinite(n_pred) & np.isfinite(n_true) & (n_true > 0)
    if not valid.any():
        return None
    alphas = n_true[valid] / n_pred[valid]
    if not np.all(np.isfinite(alphas)):
        return None
    alpha = float(np.median(alphas))
    if alpha <= 1e-8:
        return None
    return camera_center, alpha


def apply_scale_about_camera_center(
    transform: tuple[np.ndarray, float],
    pts: np.ndarray,
) -> np.ndarray:
    """Apply the 1-DOF radial scale estimated by estimate_scale_about_camera_center."""
    C, alpha = transform
    pts = np.asarray(pts, dtype=np.float64)
    return C + alpha * (pts - C)


def estimate_scale_about_camera_center_per_frame(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    group_idx: np.ndarray,
    camera_centers: np.ndarray,
    anchor_queries: np.ndarray,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
    min_anchors: int = 2,
    min_ratio: float = 1e-3,
) -> np.ndarray:
    """Estimate a per-frame radial scale about the camera center for a group.

    Returns alphas[T] with NaN where insufficient anchors are visible.
    """
    T = pred_m.shape[0]
    alphas = np.full(T, np.nan, dtype=np.float64)
    anchor_queries = np.asarray(anchor_queries, dtype=np.int64)
    group_idx = np.asarray(group_idx, dtype=np.int64)

    for t in range(T):
        vis = pred_visible[t, anchor_queries] & gt_valid[t, anchor_queries]
        if vis.sum() < min_anchors:
            continue
        q_vis = anchor_queries[vis]
        X_pred = pred_m[t, q_vis]
        X_true = gt_m[t, q_vis]
        C = camera_centers[t]

        d_pred = X_pred - C
        d_true = X_true - C
        n_pred = np.linalg.norm(d_pred, axis=1)
        n_true = np.linalg.norm(d_true, axis=1)
        valid = (
            (n_pred > min_ratio)
            & np.isfinite(n_pred)
            & np.isfinite(n_true)
            & (n_true > 0)
        )
        if valid.sum() < min_anchors:
            continue
        ratios = n_true[valid] / n_pred[valid]
        if not np.all(np.isfinite(ratios)):
            continue
        alpha = float(np.median(ratios))
        if alpha > 1e-8:
            alphas[t] = alpha
    return alphas


def smooth_1d_scales(scales: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Fill NaN and apply a small Gaussian smoothing to a 1-D scale trajectory."""
    scales = np.asarray(scales, dtype=np.float64)
    if not np.isfinite(scales).any():
        return scales

    x = np.arange(len(scales))
    valid = np.isfinite(scales)
    if valid.all():
        filled = scales.copy()
    else:
        filled = np.interp(
            x,
            x[valid],
            scales[valid],
            left=float(scales[valid][0]),
            right=float(scales[valid][-1]),
        )

    if sigma <= 0:
        out = scales.copy()
        out[valid] = filled[valid]
        return out

    radius = max(1, int(4 * sigma + 0.5))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(filled, (radius, radius), mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")

    out = scales.copy()
    out[valid] = smoothed[valid]
    return out


def apply_scale_about_camera_center_per_frame(
    transform: tuple[np.ndarray, np.ndarray],
    pts: np.ndarray,
) -> np.ndarray:
    """Apply per-frame radial scales about per-frame camera centers."""
    C, alpha = transform
    C = np.asarray(C, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    shape = pts.shape
    pts = pts.reshape(shape[0], -1, 3)
    out = C[:, None, :] + alpha[:, None, None] * (pts - C[:, None, :])
    return out.reshape(shape)


def estimate_scale_about_anchor(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    anchor_idx: int = 0,
    min_ratio: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Estimate a single scale about an anchor point.

    The anchor point is mapped exactly to its true position; all other points are
    scaled relative to it. Requires at least 2 points (anchor + one other).
    """
    X_pred = np.asarray(X_pred, dtype=np.float64)
    X_true = np.asarray(X_true, dtype=np.float64)
    if X_pred.shape[0] < 2 or X_pred.shape != X_true.shape or X_pred.shape[1] != 3:
        return None
    anchor_true = X_true[anchor_idx].copy()
    anchor_pred = X_pred[anchor_idx].copy()

    others = np.arange(X_pred.shape[0]) != anchor_idx
    dp = X_pred[others] - anchor_pred
    dt = X_true[others] - anchor_true
    np_off = np.linalg.norm(dp, axis=1)
    nt_off = np.linalg.norm(dt, axis=1)
    valid = (np_off > min_ratio) & np.isfinite(np_off) & np.isfinite(nt_off) & (nt_off > 0)
    if not valid.any():
        return None
    ratios = nt_off[valid] / np_off[valid]
    if not np.all(np.isfinite(ratios)):
        return None
    s = float(np.median(ratios))
    if s <= 1e-8:
        return None
    return anchor_true, anchor_pred, s


def apply_scale_about_anchor(
    transform: tuple[np.ndarray, np.ndarray, float],
    pts: np.ndarray,
) -> np.ndarray:
    """Apply the 1-DOF scale about an anchor estimated by estimate_scale_about_anchor."""
    anchor_true, anchor_pred, s = transform
    pts = np.asarray(pts, dtype=np.float64)
    return anchor_true + s * (pts - anchor_pred)


def estimate_group_translation(
    X_pred: np.ndarray,
    X_true: np.ndarray,
) -> np.ndarray | None:
    """Estimate a constant per-group translation (median residual)."""
    X_pred = np.asarray(X_pred, dtype=np.float64)
    X_true = np.asarray(X_true, dtype=np.float64)
    if X_pred.shape[0] < 1 or X_pred.shape != X_true.shape or X_pred.shape[1] != 3:
        return None
    delta = X_true - X_pred
    if not np.all(np.isfinite(delta)):
        return None
    return np.median(delta, axis=0)


def apply_group_translation(
    translation: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """Add a constant per-group translation to points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts + np.asarray(translation, dtype=np.float64)
