"""GAUGE: group-wise view-inconsistency rectification for feed-forward 4D tracking.

GAUGE is a training-free and model-agnostic post-hoc correction module. It takes the
3D trajectories a feed-forward tracker already outputs and returns corrected
trajectories. It modifies no weight of the tracker, needs no GPU, no
backpropagation and no optimizer, and enters no optimization loop.

The module runs in three stages: unsupervised motion grouping
(:mod:`gauge.grouping`), sparse metric anchor allocation (:mod:`gauge.anchors`)
and a low-degree-of-freedom geometric correction per group (:mod:`gauge.core`).

Example:
    >>> import numpy as np
    >>> from gauge import GAUGEConfig, apply_gauge
    >>> corrected = apply_gauge(
    ...     pred_m,        # [T, N, 3] predicted trajectories, world frame
    ...     gt_m,          # [T, N, 3] metric ground truth, read where gt_valid
    ...     pred_visible,  # [T, N]    bool predicted visibility
    ...     gt_valid,      # [T, N]    bool, the measured anchor entries
    ...     centers,       # [T, 3]    per-frame camera centers
    ...     config=GAUGEConfig(target_anchor_frac=0.05),
    ...     seed=0,
    ... )

If you already have groups of your own, pass them through ``groups``; see
:func:`gauge.adapt.validate_groups` for the contract. Helpers for converting a
tracker or dataset into the arrays above are in :mod:`gauge.adapt`.
"""

from __future__ import annotations

from gauge.adapt import (
    GAUGEInputWarning,
    camera_centers_from_world_to_camera,
    make_anchor_arrays,
    validate_groups,
    validate_inputs,
)
from gauge.config import GAUGEConfig
from gauge.core import (
    apply_gauge,
    apply_gauge_with_diagnostics,
    apply_group_correction,
)
from gauge.geometry import compute_pred_parallax_per_query
from gauge.grouping import (
    GROUP_TYPES,
    build_motion_groups,
    cluster_motion_groups,
)

__version__ = "1.0.0"

__all__ = [
    "GAUGEConfig",
    "apply_gauge",
    "apply_gauge_with_diagnostics",
    "apply_group_correction",
    "build_motion_groups",
    "cluster_motion_groups",
    "GROUP_TYPES",
    "compute_pred_parallax_per_query",
    "make_anchor_arrays",
    "camera_centers_from_world_to_camera",
    "validate_inputs",
    "validate_groups",
    "GAUGEInputWarning",
]
