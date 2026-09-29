"""Generate ``examples/sample_sequence.npz``.

The sample is synthetic, so the repository carries no dataset content and the
file stays small. It is kept as a script rather than as an opaque binary so that
the provenance of the numbers in the quick start is auditable.

The scene has one static background and two rigidly moving objects. The
prediction of each object carries a constant radial bias about the camera center,
which is the error family GAUGE removes, plus a small tangential jitter that
GAUGE is expected to leave alone.

Run from the root of the package:

    python examples/make_sample_sequence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_FRAMES = 12
N_BACKGROUND = 60
N_OBJECT_A = 32
N_OBJECT_B = 28
N_QUERIES = N_BACKGROUND + N_OBJECT_A + N_OBJECT_B

OUT_PATH = Path(__file__).resolve().parent / "sample_sequence.npz"


def build_sequence(seed: int = 0):
    """Return the arrays stored in the sample file."""
    rng = np.random.default_rng(seed)

    # The camera translates, which is what makes the radial direction differ
    # between frames and gives the per-frame scale something to correct.
    centers = np.linspace([0.0, 0.0, 0.0], [1.30, 0.10, 0.0], NUM_FRAMES)

    # (first index, last index, scene origin, velocity, radial bias, group id)
    layout = [
        (0, N_BACKGROUND, np.array([0.0, 0.0, 4.2]), np.zeros(3), 1.00, -1),
        (N_BACKGROUND, N_BACKGROUND + N_OBJECT_A, np.array([2.1, 0.0, 3.2]),
         np.array([0.30, 0.02, 0.0]), 0.78, 0),
        (N_BACKGROUND + N_OBJECT_A, N_QUERIES, np.array([-2.0, 0.3, 3.6]),
         np.array([0.0, -0.24, 0.0]), 1.15, 1),
    ]

    gt = np.zeros((NUM_FRAMES, N_QUERIES, 3), dtype=np.float64)
    pred = np.zeros((NUM_FRAMES, N_QUERIES, 3), dtype=np.float64)
    group_id = np.full(N_QUERIES, -1, dtype=np.int64)

    for lo, hi, origin, velocity, bias, label in layout:
        count = hi - lo
        cloud = origin + rng.normal(size=(count, 3)) * 0.18
        group_id[lo:hi] = label

        for t in range(NUM_FRAMES):
            position = cloud + velocity * t
            gt[t, lo:hi] = position

            radial = position - centers[t]
            tangential = rng.normal(size=(count, 3)) * 0.008
            pred[t, lo:hi] = centers[t] + bias * radial + tangential

    # A handful of occlusions, so that the visibility mask and the co-visibility
    # tests in the grouping are exercised rather than trivially satisfied.
    visible = np.ones((NUM_FRAMES, N_QUERIES), dtype=bool)
    occluded = rng.random((NUM_FRAMES, N_QUERIES)) < 0.03
    visible[occluded] = False
    visible[0, N_BACKGROUND:] = True  # query points must be visible at frame 0

    meta = {
        "description": "synthetic sequence with one static background and two moving objects",
        "num_frames": NUM_FRAMES,
        "num_queries": N_QUERIES,
        "group_id": "-1 = static background, 0 and 1 = moving objects",
        "radial_bias": {"background": 1.0, "object_0": 0.78, "object_1": 1.15},
        "units": "arbitrary metric units",
        "generated_by": "examples/make_sample_sequence.py",
    }

    return pred, gt, visible, centers, group_id, meta


def main() -> None:
    pred, gt, visible, centers, group_id, meta = build_sequence()

    # The prediction in the file is what a tracker would return, already aligned
    # to the world frame. Every visible query is a valid anchor candidate here;
    # a real user would mark only the entries they measured.
    np.savez_compressed(
        OUT_PATH,
        pred_m=pred.astype(np.float32),
        gt_m=gt.astype(np.float32),
        pred_visible=visible,
        gt_valid=visible.copy(),
        centers=centers.astype(np.float32),
        group_id=group_id,
        meta=np.array(json.dumps(meta, indent=2)),
    )

    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  pred_m {pred.shape}, objects {sorted(set(group_id.tolist()))}")


if __name__ == "__main__":
    main()
