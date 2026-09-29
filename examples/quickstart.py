"""Minimal end-to-end example of the GAUGE correction module.

It loads the synthetic sample shipped with the repository, which needs no
dataset, no model weights, no GPU and no network access, and then shows the
three things a new user usually wants to see:

1. the arrays the module expects and how they are read from a file,
2. what the built-in grouping recovers on a scene whose true groups are known,
3. how the endpoint error of the moving points changes with the anchor budget.

Run from the root of the package:

    python examples/quickstart.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gauge import (  # noqa: E402
    GAUGEConfig,
    apply_gauge,
    apply_gauge_with_diagnostics,
)

SAMPLE = Path(__file__).resolve().parent / "sample_sequence.npz"


def median_epe(pred: np.ndarray, gt: np.ndarray, query_mask: np.ndarray) -> float:
    """Median endpoint error over the points selected by a [N] bool mask."""
    error = np.linalg.norm(pred.astype(np.float64) - gt.astype(np.float64), axis=-1)
    return float(np.median(error[:, query_mask]))


def plural(count: int, noun: str) -> str:
    """Return "1 anchor" or "3 anchors"."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


def main() -> None:
    if not SAMPLE.exists():
        raise SystemExit(
            f"{SAMPLE} is missing. Generate it first with:\n"
            "    python examples/make_sample_sequence.py"
        )

    data = np.load(SAMPLE, allow_pickle=False)
    meta = json.loads(str(data["meta"]))

    # The five arrays the module takes. See gauge.adapt for helpers that build
    # them from your own tracker or dataset.
    pred_m = data["pred_m"]           # [T, N, 3] predicted trajectories
    gt_m = data["gt_m"]               # [T, N, 3] metric ground truth
    pred_visible = data["pred_visible"]  # [T, N]    tracker-predicted visibility
    gt_valid = data["gt_valid"]       # [T, N]    where a metric observation exists
    centers = data["centers"]         # [T, 3]    per-frame camera center
    group_id = data["group_id"]       # [N]       ground-truth grouping, for reporting only

    T, N = pred_visible.shape
    moving = group_id >= 0
    print(f"sequence: {T} frames, {N} query points, {int(moving.sum())} on moving objects")

    # ------------------------------------------------------------------
    # 1. What the built-in grouping recovers
    # ------------------------------------------------------------------
    config = GAUGEConfig(target_anchor_frac=0.05)
    corrected, diag = apply_gauge_with_diagnostics(
        pred_m, gt_m, pred_visible, gt_valid, centers, config=config, seed=0
    )

    counts: dict[str, int] = {}
    for group in diag["groups"]:
        counts[group["group_type"]] = counts.get(group["group_type"], 0) + 1
    print(f"grouping: {counts} (the scene has 1 background and 2 moving objects)")
    print(f"anchors: {diag['n_anchors_used']} of a budget of "
          f"floor({config.target_anchor_frac} * {N}) = {int(np.floor(config.target_anchor_frac * N))}")

    # How well the recovered co-moving groups line up with the two objects. The
    # grouping is expected to be finer than the semantic instances and that is
    # favourable, because the radial scale bias is more consistent inside a
    # smaller group. What matters is that every moving point is covered and that
    # no recovered group mixes the two objects.
    co_moving = [g for g in diag["groups"] if g["group_type"] == "co_moving"]
    mixed = 0
    for group in co_moving:
        labels = set(group_id[np.asarray(group["indices"])].tolist()) - {-1}
        if len(labels) > 1:
            mixed += 1

    for label in sorted(set(group_id[moving].tolist())):
        members = np.nonzero(group_id == label)[0]
        covered = np.zeros(members.size, dtype=bool)
        touching = 0
        for group in co_moving:
            overlap = np.isin(members, np.asarray(group["indices"]))
            if overlap.any():
                touching += 1
                covered |= overlap
        print(f"  object {label}: {int(covered.sum())}/{members.size} points covered "
              f"by {touching} co-moving groups")
    print(f"  recovered co-moving groups mixing the two objects: {mixed}")

    background = np.nonzero(group_id < 0)[0]
    fixed_parts = [np.asarray(g["indices"]) for g in diag["groups"]
                   if g["group_type"] == "world_fixed"]
    fixed = np.concatenate(fixed_parts) if fixed_parts else np.zeros(0, dtype=int)
    print(f"  background: {int(np.isin(background, fixed).sum())}/{background.size} points "
          f"covered, as {plural(counts.get('world_fixed', 0), 'world-fixed group')}")

    # ------------------------------------------------------------------
    # 2. Effect of the anchor budget
    # ------------------------------------------------------------------
    baseline = median_epe(pred_m, gt_m, moving)
    print(f"\nmoving-point EPE ({meta['units']}):")
    print(f"  {'no correction':<20}{baseline:.4f}")
    for fraction in (0.01, 0.03, 0.05):
        result, budget_diag = apply_gauge_with_diagnostics(
            pred_m, gt_m, pred_visible, gt_valid, centers,
            config=GAUGEConfig(target_anchor_frac=fraction), seed=0,
        )
        epe = median_epe(result, gt_m, moving)
        label = f"GAUGE at {fraction:.0%} anchors"
        print(f"  {label:<20}{epe:.4f} "
              f"({100.0 * (baseline - epe) / baseline:+.1f}%)"
              f"   [{plural(budget_diag['n_anchors_used'], 'anchor')}]")
    print("  A 120-point sequence is far smaller than the benchmarks this method was")
    print("  designed for, so 1% is a single anchor and leaves most groups unfitted.")

    # ------------------------------------------------------------------
    # 3. Supplying your own groups
    # ------------------------------------------------------------------
    # If you already have a segmentation or a motion clustering, pass it in
    # instead of letting GAUGE build the groups. The contract is one dict per
    # group with "indices" and "group_type"; see gauge.adapt.validate_groups.
    own_groups = [
        {"group_type": "world_fixed", "indices": np.nonzero(group_id < 0)[0]},
        {"group_type": "co_moving", "indices": np.nonzero(group_id == 0)[0]},
        {"group_type": "co_moving", "indices": np.nonzero(group_id == 1)[0]},
    ]
    with_own = apply_gauge(
        pred_m, gt_m, pred_visible, gt_valid, centers,
        config=config, seed=0, groups=own_groups,
    )
    print(f"\nwith groups supplied by the caller     {median_epe(with_own, gt_m, moving):.4f}")

    # ------------------------------------------------------------------
    # 4. Using a handful of sparse metric observations as anchors
    # ------------------------------------------------------------------
    # GAUGE reads gt_m only where gt_valid is True, so a user with a few LiDAR
    # returns or SLAM keyframes can hand over exactly those, without owning a
    # dense metric ground truth.
    from gauge import make_anchor_arrays

    rng = np.random.default_rng(0)
    measured = np.argwhere(pred_visible)  # (frame, point) pairs a sensor could observe
    picked = measured[rng.choice(len(measured), size=int(0.05 * N), replace=False)]
    sparse_gt, sparse_valid = make_anchor_arrays(
        T, N, [(int(q), int(t), *gt_m[t, q]) for t, q in picked]
    )
    sparse = apply_gauge(
        pred_m, sparse_gt, pred_visible, sparse_valid, centers, config=config, seed=0
    )
    print(f"with {int(sparse_valid.sum())} sparse measured anchors   "
          f"{median_epe(sparse, gt_m, moving):.4f}")


if __name__ == "__main__":
    main()
