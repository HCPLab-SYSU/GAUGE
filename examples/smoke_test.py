"""End-to-end smoke test of the GAUGE correction module.

The test runs on a synthetic sequence, so it needs no dataset, no model weights,
no GPU and no network access. It checks three things:

1. the anchor budget is respected exactly;
2. the grouping separates the two moving objects from the static background;
3. the correction recovers a known per-group radial bias and lowers the endpoint
   error of the dynamic points, deterministically for a fixed seed.

Run it from the root of the package:

    python examples/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gauge import (  # noqa: E402
    GAUGEConfig,
    apply_gauge,
    apply_gauge_with_diagnostics,
)
from gauge.anchors import allocate_anchors_budget  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------

T, N_BG, N_A, N_B = 16, 90, 60, 60
N = N_BG + N_A + N_B
RADIAL_BIAS_A = 0.75
RADIAL_BIAS_B = 1.20


def build_scene(seed: int = 0):
    """Return (pred, gt, visible, centers, dynamic_mask).

    The static background is predicted correctly. The two moving objects are
    rigid translations whose predictions carry a known constant radial bias about
    the camera center, which is exactly the error family GAUGE removes.
    """
    rng = np.random.default_rng(seed)

    gt = np.zeros((T, N, 3), dtype=np.float64)
    pred = np.zeros((T, N, 3), dtype=np.float64)
    centers = np.linspace([0.0, 0.0, 0.0], [1.5, 0.2, 0.0], T)

    dynamic = np.zeros(N, dtype=bool)
    layout = [
        (0, N_BG, np.array([0.0, 0.0, 4.0]), np.zeros(3), 1.0),
        (N_BG, N_BG + N_A, np.array([2.2, 0.0, 3.0]), np.array([0.35, 0.0, 0.0]), RADIAL_BIAS_A),
        (N_BG + N_A, N, np.array([-2.2, 0.0, 3.5]), np.array([0.0, -0.30, 0.0]), RADIAL_BIAS_B),
    ]

    for lo, hi, origin, velocity, bias in layout:
        n = hi - lo
        cloud = origin + rng.normal(size=(n, 3)) * 0.15
        for t in range(T):
            gt[t, lo:hi] = cloud + velocity * t
            d = gt[t, lo:hi] - centers[t]
            pred[t, lo:hi] = centers[t] + bias * d
        if bias != 1.0:
            dynamic[lo:hi] = True

    visible = np.ones((T, N), dtype=bool)
    rng_occ = np.random.default_rng(seed + 100)
    visible[rng_occ.random((T, N)) < 0.05] = False
    return pred, gt, visible, centers, dynamic


def median_epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    err = np.linalg.norm(pred - gt, axis=-1)
    return float(np.median(err[:, mask]))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_anchor_budget() -> None:
    """The allocation returns exactly floor(rho * N) anchors and never more than
    a group has valid queries."""
    config = GAUGEConfig(target_anchor_frac=0.05)
    pred, gt, visible, _centers, dynamic = build_scene()

    groups = [
        {"group_type": "co_moving", "indices": np.nonzero(dynamic)[0][:40]},
        {"group_type": "co_moving", "indices": np.nonzero(dynamic)[0][40:80]},
        {"group_type": "world_fixed", "indices": np.nonzero(~dynamic)[0]},
        {"group_type": "independent_dynamic", "indices": np.array([0, 1, 2])},
    ]

    k_per_group = allocate_anchors_budget(
        groups,
        N,
        config.target_anchor_frac,
        pred_m=pred,
        pred_visible=visible,
        gt_valid=visible,
        strategy=config.anchor_allocation_strategy,
    )

    expected = int(np.floor(config.target_anchor_frac * N))
    total = sum(k_per_group.values())
    assert total == expected, f"budget {total} != floor(rho*N) = {expected}"
    assert k_per_group[3] == 0, "independent_dynamic groups must receive no anchor"
    for i, group in enumerate(groups):
        assert k_per_group[i] <= group["indices"].size, f"group {i} over-allocated"
    print(f"[ok] anchor budget: {total} anchors = floor(0.05 * {N})")


def check_grouping() -> None:
    """The two moving objects are separated and the background is world-fixed."""
    config = GAUGEConfig()
    pred, gt, visible, centers, dynamic = build_scene()
    _corrected, diag = apply_gauge_with_diagnostics(
        pred, gt, visible, visible, centers, config=config, seed=0
    )

    counts: dict[str, int] = {}
    for group in diag["groups"]:
        counts[group["group_type"]] = counts.get(group["group_type"], 0) + 1

    n_co_moving = counts.get("co_moving", 0)
    n_world_fixed = counts.get("world_fixed", 0)
    assert n_co_moving >= 2, f"expected at least 2 co-moving groups, got {n_co_moving}"
    assert n_world_fixed >= 1, f"expected a world-fixed group, got {n_world_fixed}"

    # The two object groups must not be the same group.
    object_groups = [
        g for g in diag["groups"] if g["group_type"] == "co_moving"
    ]
    sizes = sorted(int(np.asarray(g["indices"]).size) for g in object_groups)
    assert sizes[-2:] and sizes[-1] >= N_A * 0.5, f"object groups too small: {sizes}"
    print(f"[ok] grouping: {n_co_moving} co-moving and {n_world_fixed} world-fixed groups")


def check_end_to_end() -> None:
    """The correction removes a known radial bias and lowers the endpoint error."""
    pred, gt, visible, centers, dynamic = build_scene()
    base = median_epe(pred, gt, dynamic)

    results = []
    for frac in (0.01, 0.03, 0.05):
        config = GAUGEConfig(target_anchor_frac=frac)
        corrected = apply_gauge(
            pred, gt, visible, visible, centers, config=config, seed=0
        )
        epe = median_epe(corrected, gt, dynamic)
        gain = 100.0 * (base - epe) / base
        assert epe < base, f"budget {frac}: EPE {epe:.4f} did not improve on {base:.4f}"
        assert gain > 20.0, f"budget {frac}: gain {gain:.1f}% is unexpectedly small"
        results.append((frac, epe, gain))

    for frac, epe, gain in results:
        print(f"[ok] budget {int(frac * 100):>2d}%: dynamic EPE {base:.4f} -> {epe:.4f} ({gain:+.1f}%)")


def check_determinism() -> None:
    """The same seed reproduces the same corrected trajectories."""
    pred, gt, visible, centers, _dynamic = build_scene()
    config = GAUGEConfig()
    first = apply_gauge(pred, gt, visible, visible, centers, config=config, seed=7)
    second = apply_gauge(pred, gt, visible, visible, centers, config=config, seed=7)
    assert np.array_equal(first, second), "the pipeline is not deterministic in the seed"
    print("[ok] determinism: identical output for a fixed seed")


def main() -> None:
    print("GAUGE smoke test")
    print(f"synthetic sequence: {T} frames, {N} query points")
    check_anchor_budget()
    check_grouping()
    check_end_to_end()
    check_determinism()
    print("all checks passed")


if __name__ == "__main__":
    main()
