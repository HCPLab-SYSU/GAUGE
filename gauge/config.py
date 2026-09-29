"""Configuration of the GAUGE correction module.

The defaults of :class:`GAUGEConfig` are the configuration evaluated in the
paper. The mapping between these fields and the hyperparameters listed in the
paper is given in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class GAUGEConfig:
    """Hyper-parameters of the GAUGE correction module.

    Grouping. A query point is world-fixed when the standard deviation of its
    predicted position over the frames where it is visible stays below
    ``world_fixed_std_threshold``. Every other point receives a descriptor made of
    the unit velocities over consecutive co-visible frames. Two points are linked
    when their descriptors agree on at least ``min_co_visible_frames`` shared
    frames with a mean cosine similarity of at least ``direction_cos_threshold``,
    and the test is evaluated inside a spatial kNN neighbourhood of
    ``direction_knn`` neighbours. Connected components form the co-moving groups,
    which are then split once more by spatial connectivity and merged by centroid
    distance and direction consistency so that fragments cut by occlusion or
    brief tracking failure are repaired.

    Correction. Co-moving groups are rescaled about the camera center, once for
    the whole sequence and once per frame, and then shifted by one group-level
    translation. World-fixed groups and low-parallax groups are rescaled about
    their first in-group anchor instead, because radial scaling about the camera
    center is ill-posed when the depth spread is large or the parallax is small.
    """

    # --- Grouping -----------------------------------------------------------
    descriptor_dirs: int = 3
    direction_cos_threshold: float = 0.90
    spatial_knn: int = 10
    motion_threshold: float = 0.01
    min_group_size: int = 3
    min_fitting_size: int = 5
    min_visible_frames: int = 5
    min_co_visible_frames: int = 3
    direction_knn: int | None = 50
    world_fixed_std_threshold: float = 0.05

    # World-fixed points cover the whole background, so their spatial spread is
    # much larger than that of a dynamic object. They are split with a wider
    # neighbourhood than the co-moving groups of ``spatial_knn``.
    world_fixed_spatial_knn: int = 15

    # --- Motion-consistent fragment merge -----------------------------------
    merge_direction_cos_threshold: float = 0.85
    merge_distance_factor: float = 3.0
    merge_min_group_size: int = 5
    merge_max_group_size: int | None = None

    # A fragment below ``min_fitting_size`` is merged into its nearest neighbour
    # when the centroid distance is below this factor times the local
    # nearest-neighbour scale of the fragment.
    fragment_merge_distance_factor: float = 5.0

    # --- Correction forms ---------------------------------------------------
    default_form: Literal[
        "scale_about_camera_per_frame_translation",
        "scale_about_camera_translation",
        "scale_about_anchor_translation",
        "scale_about_anchor",
        "full_sim3",
        "none",
    ] = "scale_about_camera_per_frame_translation"
    world_fixed_form: Literal[
        "scale_about_camera_translation",
        "scale_about_camera_per_frame_translation",
        "scale_about_anchor_translation",
        "full_sim3",
        "none",
    ] = "scale_about_anchor_translation"
    low_parallax_form: Literal[
        "scale_about_camera_translation",
        "scale_about_camera_per_frame_translation",
        "scale_about_anchor_translation",
        "full_sim3",
        "none",
    ] = "scale_about_anchor_translation"
    # Groups whose median per-query parallax is below this angle are corrected
    # in the anchor-centered form. A value of 0 disables the branch.
    parallax_threshold_deg: float = 2.0

    # --- Anchor budget ------------------------------------------------------
    target_anchor_frac: float = 0.05
    anchor_allocation_strategy: Literal["proportional_size", "motion_weighted"] = (
        "motion_weighted"
    )
    max_anchors_per_group: int | None = None

    # --- Per-frame radial scale ---------------------------------------------
    per_frame_scale_min_anchors: int = 1
    per_frame_scale_smooth_sigma: float = 2.0

    def parallax_threshold_rad(self) -> float:
        """Return the low-parallax threshold in radians."""
        return float(np.deg2rad(self.parallax_threshold_deg))
