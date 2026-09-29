"""Anchor budget allocation and anchor sampling."""

from __future__ import annotations

from typing import Any

import numpy as np


def allocate_anchors_budget(
    groups: list[dict[str, Any]],
    n_queries: int,
    target_frac: float,
    pred_m: np.ndarray | None = None,
    pred_visible: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
    strategy: str = "motion_weighted",
    max_anchors_per_group: int | None = None,
) -> dict[int, int]:
    """Allocate an integer number of anchors to every group.

    The budget of a sequence is ``floor(target_frac * n_queries)``. Groups of type
    ``independent_dynamic`` receive nothing. The budget is then split with the
    largest-remainder method, so the total is exactly the budget and no group
    receives more anchors than it has valid queries.

    Two scores are available:

    - ``motion_weighted``: ``n_g * (1 + m_g / max_g' m_g')``, where ``n_g`` is the
      number of valid distinct queries of the group and ``m_g`` its mean 3D
      displacement. This is the allocation used in the paper.
    - ``proportional_size``: the number of valid distinct queries alone. It is
      kept as a control for the ablation on anchor allocation.

    Args:
        groups: motion groups, each a dict with ``indices`` and ``group_type``.
        n_queries: total number of query points of the sequence.
        target_frac: anchor budget as a fraction of ``n_queries``.
        pred_m: [T, N, 3] predicted trajectories, needed by ``motion_weighted``.
        pred_visible: [T, N] bool predicted visibility, needed by
            ``motion_weighted``.
        gt_valid: [T, N] bool ground-truth validity, needed by
            ``motion_weighted``.
        strategy: ``motion_weighted`` or ``proportional_size``.
        max_anchors_per_group: optional cap applied to every group.

    Returns:
        Mapping from group index to the number of anchors it receives.
    """
    k_per_group = {i: 0 for i in range(len(groups))}

    budget = int(np.floor(target_frac * n_queries))
    if budget <= 0:
        return k_per_group

    candidates = [
        (i, g)
        for i, g in enumerate(groups)
        if g.get("group_type") != "independent_dynamic"
    ]
    if not candidates:
        return k_per_group

    valid_sizes: list[int] = []
    for _, g in candidates:
        idx = np.asarray(g["indices"])
        if pred_visible is not None and gt_valid is not None:
            valid = (pred_visible[:, idx] & gt_valid[:, idx]).any(axis=0)
            valid_sizes.append(int(valid.sum()))
        else:
            valid_sizes.append(int(idx.size))

    total_valid = sum(valid_sizes)
    if total_valid == 0:
        return k_per_group
    budget = min(budget, total_valid)

    if strategy == "proportional_size":
        scores = [float(size) for size in valid_sizes]
    elif strategy == "motion_weighted":
        if pred_m is None or pred_visible is None or gt_valid is None:
            raise ValueError(
                "motion_weighted requires pred_m, pred_visible and gt_valid"
            )
        motions: list[float] = []
        for _, g in candidates:
            idx = np.asarray(g["indices"])
            vis = pred_visible[:, idx] & gt_valid[:, idx]
            coords = pred_m[:, idx, :]
            valid_pairs = vis[:-1] & vis[1:]
            if valid_pairs.any():
                disp = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
                motions.append(float(np.mean(disp[valid_pairs])))
            else:
                motions.append(0.0)
        max_motion = max(motions) if motions else 0.0
        if max_motion <= 0:
            max_motion = 1.0
        scores = [
            size * (1.0 + motion / max_motion)
            for size, motion in zip(valid_sizes, motions)
        ]
    else:
        raise ValueError(f"Unknown allocation strategy: {strategy}")

    total_score = sum(scores)
    if total_score == 0:
        return k_per_group

    fractions = [budget * s / total_score for s in scores]
    floors = [
        min(int(np.floor(f)), size) for f, size in zip(fractions, valid_sizes)
    ]
    extra = budget - sum(floors)

    while extra > 0:
        best_j = -1
        best_remainder = -1.0
        best_idx = -1
        for j, (i, _) in enumerate(candidates):
            if floors[j] >= valid_sizes[j]:
                continue
            remainder = fractions[j] - floors[j]
            if remainder > best_remainder or (
                remainder == best_remainder and i < best_idx
            ):
                best_remainder = remainder
                best_j = j
                best_idx = i
        if best_j < 0:
            break
        floors[best_j] += 1
        extra -= 1

    for j, (i, _) in enumerate(candidates):
        k = floors[j]
        if max_anchors_per_group is not None:
            k = min(k, max_anchors_per_group)
        k_per_group[i] = min(k, valid_sizes[j])
    return k_per_group


def sample_anchors(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    group_idx: np.ndarray,
    pred_visible: np.ndarray,
    gt_valid: np.ndarray,
    k: int,
    rng: np.random.Generator,
    prefer_common_frame: bool = True,
) -> list[tuple[int, int]] | None:
    """Sample ``k`` anchors, that is (query, frame) pairs, from one group.

    Candidates are the queries of the group that are predicted visible and
    ground-truth valid on at least one frame. When a single frame carries at least
    ``k`` candidates, all anchors are drawn from that frame, because common-frame
    anchors share the camera center and the object pose and therefore give a
    consistent radial scale. Otherwise ``k`` distinct queries are drawn and each
    one contributes one random valid frame.

    Args:
        pred_m: [T, N, 3] predicted trajectories.
        gt_m: [T, N, 3] ground-truth trajectories.
        group_idx: query indices of the group.
        pred_visible: [T, N] bool predicted visibility.
        gt_valid: [T, N] bool ground-truth validity.
        k: number of anchors to draw.
        rng: numpy random generator, so that sampling is reproducible.
        prefer_common_frame: draw all anchors from one frame when possible.

    Returns:
        A list of ``(query, frame)`` pairs, or None when the group cannot supply
        ``k`` anchors.
    """
    if k <= 0:
        return None

    query_candidates = [
        int(q)
        for q in np.asarray(group_idx).ravel()
        if (pred_visible[:, q] & gt_valid[:, q]).any()
    ]
    if len(query_candidates) < k:
        return None
    query_candidates_arr = np.array(query_candidates, dtype=np.int64)

    if prefer_common_frame:
        T = pred_visible.shape[0]
        frames_with_enough = [
            t
            for t in range(T)
            if int(
                (
                    pred_visible[t, query_candidates_arr]
                    & gt_valid[t, query_candidates_arr]
                ).sum()
            )
            >= k
        ]
        if frames_with_enough:
            frame_t = int(rng.choice(frames_with_enough))
            valid_in_frame = query_candidates_arr[
                pred_visible[frame_t, query_candidates_arr]
                & gt_valid[frame_t, query_candidates_arr]
            ]
            chosen = rng.choice(valid_in_frame, size=k, replace=False)
            return [(int(q), frame_t) for q in chosen]

    chosen = rng.choice(query_candidates_arr, size=k, replace=False)
    anchors: list[tuple[int, int]] = []
    for q in chosen:
        frames = np.nonzero(pred_visible[:, q] & gt_valid[:, q])[0]
        anchors.append((int(q), int(rng.choice(frames))))
    return anchors
