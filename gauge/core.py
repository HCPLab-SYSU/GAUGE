"""Entry point of the GAUGE correction module."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gauge.adapt import validate_groups, validate_inputs
from gauge.anchors import allocate_anchors_budget, sample_anchors
from gauge.config import GAUGEConfig
from gauge.geometry import compute_pred_parallax_per_query
from gauge.grouping import build_motion_groups
from gauge.transforms import (
    apply_group_translation,
    apply_scale_about_anchor,
    apply_scale_about_camera_center,
    apply_scale_about_camera_center_per_frame,
    estimate_group_translation,
    estimate_scale_about_anchor,
    estimate_scale_about_camera_center,
    estimate_scale_about_camera_center_per_frame,
    smooth_1d_scales,
    umeyama_sim3,
)

CorrectionForm = str


def _anchors_to_arrays(
    anchors: list[tuple[int, int]],
    pred_m: np.ndarray,
    gt_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the predicted and ground-truth positions of the anchors."""
    X_pred = np.array([pred_m[t, q] for q, t in anchors], dtype=np.float64)
    X_true = np.array([gt_m[t, q] for q, t in anchors], dtype=np.float64)
    return X_pred, X_true


def _apply_sim3_or_translation(
    corrected: np.ndarray,
    pred_m: np.ndarray,
    group_idx: np.ndarray,
    anchors: list[tuple[int, int]],
    X_pred: np.ndarray,
    X_true: np.ndarray,
) -> np.ndarray:
    """Generic fallback: Sim(3) from at least three anchors, else a translation.

    This form is only used when the structured form cannot be estimated. It has
    more degrees of freedom than the error structure supports, so it is a last
    resort rather than an alternative.
    """
    if len(anchors) >= 3:
        sim3 = umeyama_sim3(X_pred, X_true)
        if sim3 is not None:
            scale, rotation, translation = sim3
            flat = pred_m[:, group_idx].reshape(-1, 3)
            out = scale * (rotation @ flat.T).T + translation
            corrected[:, group_idx] = out.reshape(pred_m[:, group_idx].shape)
            return corrected
    if len(anchors) >= 1:
        translation = estimate_group_translation(X_pred, X_true)
        if translation is not None:
            corrected[:, group_idx] = apply_group_translation(
                translation, pred_m[:, group_idx]
            )
    return corrected


def apply_group_correction(
    corrected: np.ndarray,
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    group_idx: np.ndarray,
    anchors: list[tuple[int, int]],
    form: CorrectionForm,
    centers: np.ndarray,
    config: GAUGEConfig,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
) -> np.ndarray:
    """Estimate one correction form on the anchors and apply it to the group.

    Args:
        corrected: [T, N, 3] array modified in place for the group columns.
        pred_m: [T, N, 3] raw predicted trajectories.
        gt_m: [T, N, 3] ground-truth trajectories.
        group_idx: query indices of the group.
        anchors: sampled (query, frame) anchor pairs of the group.
        form: name of the correction form to apply.
        centers: [T, 3] per-frame camera centers.
        config: correction configuration.
        pred_visible: [T, N] bool predicted visibility.
        gt_valid: [T, N] bool ground-truth validity.

    Returns:
        The updated ``corrected`` array.
    """
    X_pred, X_true = _anchors_to_arrays(anchors, pred_m, gt_m)

    if form == "scale_about_camera_per_frame_translation":
        # Group-level radial scale about the camera center of the first anchor
        # frame, then the per-frame scale, then one group-level translation.
        C = centers[anchors[0][1]]
        transform = estimate_scale_about_camera_center(X_pred, X_true, C)
        if transform is None:
            return _apply_sim3_or_translation(
                corrected, pred_m, group_idx, anchors, X_pred, X_true
            )

        corrected[:, group_idx] = apply_scale_about_camera_center(
            transform, pred_m[:, group_idx]
        )
        global_alpha = transform[1]

        anchor_queries = np.array(sorted({q for q, _ in anchors}), dtype=np.int64)
        alphas = estimate_scale_about_camera_center_per_frame(
            pred_m,
            gt_m,
            group_idx,
            centers,
            anchor_queries,
            pred_visible,
            gt_valid,
            min_anchors=config.per_frame_scale_min_anchors,
        )
        alphas_filled = smooth_1d_scales(
            alphas, sigma=config.per_frame_scale_smooth_sigma
        )
        alphas_filled = np.where(np.isfinite(alphas_filled), alphas_filled, global_alpha)
        corrected[:, group_idx] = apply_scale_about_camera_center_per_frame(
            (centers, alphas_filled), corrected[:, group_idx]
        )

        scaled_anchor_pts = np.array(
            [corrected[t, q] for q, t in anchors], dtype=np.float64
        )
        translation = estimate_group_translation(scaled_anchor_pts, X_true)
        if translation is not None:
            corrected[:, group_idx] = apply_group_translation(
                translation, corrected[:, group_idx]
            )

    elif form == "scale_about_camera_translation":
        C = centers[anchors[0][1]]
        transform = estimate_scale_about_camera_center(X_pred, X_true, C)
        if transform is not None:
            corrected[:, group_idx] = apply_scale_about_camera_center(
                transform, pred_m[:, group_idx]
            )
            scaled_anchor_pts = np.array(
                [corrected[t, q] for q, t in anchors], dtype=np.float64
            )
            translation = estimate_group_translation(scaled_anchor_pts, X_true)
            if translation is not None:
                corrected[:, group_idx] = apply_group_translation(
                    translation, corrected[:, group_idx]
                )
        else:
            corrected = _apply_sim3_or_translation(
                corrected, pred_m, group_idx, anchors, X_pred, X_true
            )

    elif form == "scale_about_anchor_translation":
        transform = estimate_scale_about_anchor(X_pred, X_true, anchor_idx=0)
        if transform is not None:
            corrected[:, group_idx] = apply_scale_about_anchor(
                transform, pred_m[:, group_idx]
            )
            scaled_anchor_pts = np.array(
                [corrected[t, q] for q, t in anchors], dtype=np.float64
            )
            translation = estimate_group_translation(scaled_anchor_pts, X_true)
            if translation is not None:
                corrected[:, group_idx] = apply_group_translation(
                    translation, corrected[:, group_idx]
                )
        else:
            if len(anchors) >= 1:
                translation = estimate_group_translation(X_pred, X_true)
                if translation is not None:
                    corrected[:, group_idx] = apply_group_translation(
                        translation, pred_m[:, group_idx]
                    )

    elif form == "scale_about_anchor":
        transform = estimate_scale_about_anchor(X_pred, X_true, anchor_idx=0)
        if transform is not None:
            corrected[:, group_idx] = apply_scale_about_anchor(
                transform, pred_m[:, group_idx]
            )

    elif form == "full_sim3":
        corrected = _apply_sim3_or_translation(
            corrected, pred_m, group_idx, anchors, X_pred, X_true
        )

    elif form == "none":
        pass

    else:
        raise ValueError(f"Unsupported correction form: {form}")

    return corrected


def _correct_sequence(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
    centers: np.ndarray,
    config: GAUGEConfig,
    seed: int,
    groups: Sequence[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run grouping, anchor allocation and correction on one sequence."""
    validate_inputs(
        pred_m,
        gt_m,
        pred_visible,
        gt_valid,
        centers,
        target_anchor_frac=config.target_anchor_frac,
    )

    if groups is None:
        groups = build_motion_groups(pred_m, pred_visible, config)
    else:
        validate_groups(groups, pred_m.shape[1])

    n_queries = pred_m.shape[1]

    k_per_group = allocate_anchors_budget(
        groups,
        n_queries,
        config.target_anchor_frac,
        pred_m=pred_m,
        pred_visible=pred_visible,
        gt_valid=gt_valid,
        strategy=config.anchor_allocation_strategy,
        max_anchors_per_group=config.max_anchors_per_group,
    )

    parallax = compute_pred_parallax_per_query(pred_m, pred_visible, centers)
    parallax_threshold = config.parallax_threshold_rad()

    rng = np.random.default_rng(seed)
    corrected = pred_m.copy()

    n_groups_total = 0
    n_groups_solved = 0
    n_anchors_used = 0
    group_anchors: list[list[tuple[int, int]] | None] = []

    for gi, group in enumerate(groups):
        group_type = group.get("group_type")
        group_idx = np.asarray(group["indices"])

        if group_type == "independent_dynamic":
            group_anchors.append(None)
            continue

        n_groups_total += 1
        if group_idx.size == 0:
            group_anchors.append(None)
            continue

        k = k_per_group[gi]
        if k <= 0:
            group_anchors.append(None)
            continue

        # World-fixed groups use the anchor-centered form. A low-parallax group
        # switches to that same form even when it is not world-fixed, because
        # radial scaling about the camera center is ill-posed there.
        effective_form = config.default_form
        if group_type == "world_fixed":
            effective_form = config.world_fixed_form
        if parallax_threshold > 0.0 and config.low_parallax_form != "none":
            group_parallax = float(np.nanmedian(parallax[group_idx]))
            if group_parallax < parallax_threshold:
                effective_form = config.low_parallax_form

        if effective_form == "scale_about_anchor" and group_idx.size < 2:
            group_anchors.append(None)
            continue

        anchors = sample_anchors(
            pred_m,
            gt_m,
            group_idx,
            pred_visible,
            gt_valid,
            k,
            rng,
            prefer_common_frame=True,
        )
        group_anchors.append(anchors)
        if anchors is None:
            continue

        before = corrected[:, group_idx].copy()
        corrected = apply_group_correction(
            corrected,
            pred_m,
            gt_m,
            group_idx,
            anchors,
            effective_form,
            centers,
            config,
            pred_visible,
            gt_valid,
        )
        n_anchors_used += len(anchors)
        if np.linalg.norm(corrected[:, group_idx] - before).max() > 1e-6:
            n_groups_solved += 1

    diagnostics = {
        "n_groups_total": n_groups_total,
        "n_groups_solved": n_groups_solved,
        "n_anchors_used": n_anchors_used,
        "groups": groups,
        "k_per_group": k_per_group,
        "group_anchors": group_anchors,
    }
    return corrected, diagnostics


def apply_gauge(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
    centers: np.ndarray,
    config: GAUGEConfig | None = None,
    seed: int = 0,
    groups: Sequence[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply the GAUGE correction to one sequence.

    Args:
        pred_m: [T, N, 3] globally aligned predicted trajectories.
        gt_m: [T, N, 3] metric ground truth. It is read only at the entries where
            ``gt_valid`` is True, so it may be sparse; see
            :func:`gauge.adapt.make_anchor_arrays`.
        pred_visible: [T, N] bool tracker-predicted visibility.
        gt_valid: [T, N] bool mask of the entries where a metric observation
            exists. These entries form the candidate pool the anchors are drawn
            from.
        centers: [T, 3] camera centers in the same coordinate frame as ``pred_m``.
        config: correction configuration; defaults to :class:`GAUGEConfig`.
        seed: seed of the anchor sampling.
        groups: optional precomputed groups, for instance a segmentation or a
            motion clustering of your own. Each entry must be a dict with
            ``"indices"`` and ``"group_type"``; see
            :func:`gauge.adapt.validate_groups`. When None, the built-in
            unsupervised grouping of :func:`gauge.grouping.build_motion_groups`
            is used.

    Returns:
        corrected: [T, N, 3] corrected trajectories. Uncorrected points keep
        their input prediction.
    """
    if config is None:
        config = GAUGEConfig()
    corrected, _ = _correct_sequence(
        pred_m, gt_m, pred_visible, gt_valid, centers, config, seed, groups
    )
    return corrected


def apply_gauge_with_diagnostics(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
    centers: np.ndarray,
    config: GAUGEConfig | None = None,
    seed: int = 0,
    groups: Sequence[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Same as :func:`apply_gauge`, but also return grouping diagnostics.

    The returned dict holds ``n_groups_total``, ``n_groups_solved``,
    ``n_anchors_used``, ``groups``, ``k_per_group`` and ``group_anchors``.
    """
    if config is None:
        config = GAUGEConfig()
    return _correct_sequence(
        pred_m, gt_m, pred_visible, gt_valid, centers, config, seed, groups
    )
