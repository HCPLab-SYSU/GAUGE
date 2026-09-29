"""Helpers for plugging your own tracker into GAUGE.

GAUGE works on trajectories rather than on images, so adapting it to a new
predictor is a matter of producing five arrays and nothing else:

===========================  ==================  ==================================
argument                     shape               meaning
===========================  ==================  ==================================
``pred_m``                   ``[T, N, 3]``       predicted trajectories, already
                                                 aligned to the world frame
``pred_visible``             ``[T, N]`` bool     tracker-predicted visibility
``centers``                  ``[T, 3]``          per-frame camera center, in the
                                                 same frame as ``pred_m``
``gt_m``                     ``[T, N, 3]``       metric ground truth, read only at
                                                 the entries marked by ``gt_valid``
``gt_valid``                 ``[T, N]`` bool     where a metric observation exists
===========================  ==================  ==================================

Three functions cover the parts that are easy to get wrong:

- :func:`camera_centers_from_world_to_camera` converts extrinsics into centers.
- :func:`make_anchor_arrays` turns a handful of sparse metric observations into
  the ``gt_m`` / ``gt_valid`` pair, so you do not have to build a dense ground
  truth array with meaningless entries.
- :func:`validate_inputs` reports the mistakes that would otherwise fail
  silently.

See ``docs/adapt_your_tracker.md`` for a worked guide.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "GAUGEInputWarning",
    "camera_centers_from_world_to_camera",
    "make_anchor_arrays",
    "validate_inputs",
    "validate_groups",
]


class GAUGEInputWarning(UserWarning):
    """Warning raised when the inputs are usable but look inconsistent."""


def camera_centers_from_world_to_camera(w2c: np.ndarray) -> np.ndarray:
    """Return per-frame camera centers from world-to-camera matrices.

    Most trackers and datasets expose the extrinsic matrix that maps world points
    into the camera frame, ``X_cam = R X_world + t``, rather than the camera
    center that GAUGE needs. The center is ``C = -R^T t``.

    Args:
        w2c: ``[T, 3, 4]`` or ``[T, 4, 4]`` world-to-camera matrices.

    Returns:
        centers: ``[T, 3]`` camera centers in the world frame.

    Example:
        >>> centers = camera_centers_from_world_to_camera(extrinsics)
        >>> corrected = apply_gauge(pred_m, gt_m, pred_visible, gt_valid, centers)
    """
    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1] not in (3, 4):
        raise ValueError(f"w2c must be [T, 3, 4] or [T, 4, 4], got {w2c.shape}")
    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    centers = -np.einsum("tji,tj->ti", rotation, translation)
    return centers


def make_anchor_arrays(
    n_frames: int,
    n_queries: int,
    anchors: Iterable[Sequence[float]],
    *,
    fill_value: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``gt_m`` / ``gt_valid`` pair from sparse metric observations.

    GAUGE needs a device-independent accessor for metric anchors rather than a
    dense ground-truth volume: it reads ``gt_m`` only at the entries where
    ``gt_valid`` is True, so everything else can stay unset. This function
    allocates that pair for you.

    Args:
        n_frames: number of frames ``T``.
        n_queries: number of query points ``N``.
        anchors: iterable of ``(query_index, frame_index, x, y, z)``, or of
            ``(query_index, frame_index, xyz_array)``. These are the entries a
            user actually measured, for instance from sparse LiDAR returns, SLAM
            keyframes or a RGB-D sensor.
        fill_value: value written into ``gt_m`` outside the anchors. The default
            NaN makes it obvious that no observation exists there.

    Returns:
        ``(gt_m, gt_valid)`` with shapes ``[T, N, 3]`` and ``[T, N]``.

    Example:
        >>> gt_m, gt_valid = make_anchor_arrays(T, N, [(q, t, *xyz) for q, t, xyz in anchors])
        >>> corrected = apply_gauge(pred_m, gt_m, pred_visible, gt_valid, centers)
    """
    if n_frames <= 0 or n_queries <= 0:
        raise ValueError(f"n_frames and n_queries must be positive, got {n_frames}, {n_queries}")

    gt_m = np.full((n_frames, n_queries, 3), fill_value, dtype=np.float64)
    gt_valid = np.zeros((n_frames, n_queries), dtype=bool)

    for entry in anchors:
        if len(entry) == 3:
            query, frame, xyz = entry
            xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        elif len(entry) == 5:
            query, frame, x, y, z = entry
            xyz = np.array([x, y, z], dtype=np.float64)
        else:
            raise ValueError(
                "each anchor must be (query, frame, xyz) or (query, frame, x, y, z), "
                f"got a sequence of length {len(entry)}"
            )

        query = int(query)
        frame = int(frame)
        if not 0 <= query < n_queries:
            raise ValueError(f"anchor query index {query} is outside [0, {n_queries})")
        if not 0 <= frame < n_frames:
            raise ValueError(f"anchor frame index {frame} is outside [0, {n_frames})")

        gt_m[frame, query] = xyz
        gt_valid[frame, query] = True

    return gt_m, gt_valid


def validate_groups(groups: Sequence[dict[str, Any]], n_queries: int) -> None:
    """Check that ``groups`` follows the contract expected by :func:`apply_gauge`.

    The contract is a sequence of dicts, each with

    - ``"indices"``: 1-D integer array of query indices,
    - ``"group_type"``: one of ``"world_fixed"``, ``"co_moving"`` or
      ``"independent_dynamic"``.

    ``independent_dynamic`` groups are skipped by the correction, so it is valid
    to list them or to omit their points entirely.

    Raises:
        ValueError: if a group is malformed, or if the indices are out of range
            or repeated across groups.
    """
    from gauge.grouping import GROUP_TYPES

    seen = np.zeros(n_queries, dtype=bool)
    for position, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"group {position} must be a dict, got {type(group).__name__}")
        if "indices" not in group:
            raise ValueError(f"group {position} has no 'indices' entry")
        group_type = group.get("group_type", "co_moving")
        if group_type not in GROUP_TYPES:
            raise ValueError(
                f"group {position} has group_type {group_type!r}, expected one of {GROUP_TYPES}"
            )

        indices = np.asarray(group["indices"])
        if indices.ndim != 1:
            raise ValueError(f"group {position} 'indices' must be 1-D, got shape {indices.shape}")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"group {position} 'indices' must be integers, got {indices.dtype}")
        if indices.size and (indices.min() < 0 or indices.max() >= n_queries):
            raise ValueError(
                f"group {position} has indices outside [0, {n_queries})"
            )
        repeated = seen[indices] if indices.size else np.zeros(0, dtype=bool)
        if repeated.any():
            duplicates = indices[repeated][:5].tolist()
            raise ValueError(
                f"group {position} repeats query indices already assigned to an earlier "
                f"group (for example {duplicates}); every query may belong to at most one group"
            )
        seen[indices] = True


def validate_inputs(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
    centers: np.ndarray,
    *,
    target_anchor_frac: float | None = None,
    warn: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Check the arrays passed to :func:`apply_gauge` and report inconsistencies.

    Shape and type errors always raise :class:`ValueError`. Conditions that are
    legal but almost always a mistake, such as trajectories expressed in the
    camera frame, are reported as :class:`GAUGEInputWarning`.

    Args:
        pred_m: ``[T, N, 3]`` predicted trajectories.
        gt_m: ``[T, N, 3]`` metric ground truth, read only where ``gt_valid``.
        pred_visible: ``[T, N]`` bool predicted visibility.
        gt_valid: ``[T, N]`` bool mask of metric observations.
        centers: ``[T, 3]`` per-frame camera centers.
        target_anchor_frac: the configured anchor budget, used to detect a budget
            that will be capped by the number of available observations.
        warn: emit warnings through the :mod:`warnings` module.
        strict: raise instead of warning on the first inconsistency.

    Returns:
        A report dict with ``n_frames``, ``n_queries``, ``n_visible``,
        ``n_gt_valid``, ``n_anchor_candidates``, ``budget`` and
        ``budget_capped``.
    """
    # When called through apply_gauge the interesting frame is the caller of
    # apply_gauge, five frames up from here: warn, _report, validate_inputs,
    # _correct_sequence, apply_gauge, caller. Python falls back to the outermost
    # frame if the stack is shallower, so a direct call still reports sensibly.
    _STACKLEVEL = 5

    def _fail(message: str) -> None:
        raise ValueError(message)

    def _report(message: str) -> None:
        if strict:
            raise ValueError(message)
        if warn:
            warnings.warn(message, GAUGEInputWarning, stacklevel=_STACKLEVEL)

    pred_m = np.asarray(pred_m)
    if pred_m.ndim != 3 or pred_m.shape[2] != 3:
        _fail(f"pred_m must be [T, N, 3], got shape {pred_m.shape}")

    T, N = pred_m.shape[:2]

    expected = {
        "gt_m": ((T, N, 3), np.asarray(gt_m).shape),
        "pred_visible": ((T, N), np.asarray(pred_visible).shape),
        "gt_valid": ((T, N), np.asarray(gt_valid).shape),
        "centers": ((T, 3), np.asarray(centers).shape),
    }
    for name, (want, got) in expected.items():
        if want != got:
            _fail(f"{name} must have shape {want} to match pred_m, got {got}")

    pred_visible = np.asarray(pred_visible).astype(bool, copy=False)
    gt_valid = np.asarray(gt_valid).astype(bool, copy=False)
    centers = np.asarray(centers, dtype=np.float64)

    if not np.isfinite(centers).all():
        _fail("centers contain NaN or inf")

    n_visible = int(pred_visible.sum())
    n_gt_valid = int(gt_valid.sum())
    n_candidates = int((pred_visible & gt_valid).sum())

    nan_where_visible = int(np.isnan(pred_m[pred_visible]).any(axis=-1).sum()) if n_visible else 0
    if nan_where_visible:
        _report(
            f"pred_m contains NaN at {nan_where_visible} (frame, point) pairs that "
            "pred_visible marks as visible; those trajectories cannot be corrected. "
            "Clear the visibility flag there, or fill the positions."
        )

    if n_gt_valid == 0:
        _fail(
            "gt_valid is empty, so no metric observation is available and nothing can "
            "be corrected. Mark the (frame, point) pairs where you have a metric "
            "measurement; see gauge.adapt.make_anchor_arrays."
        )

    if np.allclose(centers, 0.0):
        _report(
            "all camera centers are zero. GAUGE scales each group radially about the "
            "camera center, which is not defined in the camera frame. If pred_m is in "
            "the camera frame, convert it to the world frame and pass the matching "
            "centers; see gauge.adapt.camera_centers_from_world_to_camera."
        )
    elif np.allclose(centers, centers[0]):
        _report(
            "the camera center is identical in every frame, so the sequence has no "
            "camera translation. Radial scaling about a fixed center degenerates to a "
            "single global scale, which is the case GAUGE is designed to go beyond."
        )

    budget = int(np.floor(target_anchor_frac * N)) if target_anchor_frac else None
    budget_capped = bool(budget is not None and budget > n_candidates)
    if budget_capped:
        _report(
            f"the anchor budget is floor({target_anchor_frac} * {N}) = {budget} but only "
            f"{n_candidates} (frame, point) observations are marked in gt_valid, so the "
            f"budget is capped at {n_candidates}. Mark more observations or lower "
            "target_anchor_frac to match what you have."
        )

    return {
        "n_frames": T,
        "n_queries": N,
        "n_visible": n_visible,
        "n_gt_valid": n_gt_valid,
        "n_anchor_candidates": n_candidates,
        "budget": budget,
        "budget_capped": budget_capped,
    }
