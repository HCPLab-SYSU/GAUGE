# Hyperparameters

Every hyperparameter lives in `gauge.GAUGEConfig`. The defaults are the
configuration evaluated in the paper, so a user who does not pass a config gets
the published behaviour.

```python
from gauge import GAUGEConfig, apply_gauge

config = GAUGEConfig(target_anchor_frac=0.05)
corrected = apply_gauge(..., config=config)
```

## Grouping

| Field | Default | Effect |
| --- | --- | --- |
| `world_fixed_std_threshold` | `0.05` | position standard deviation below which a point counts as world-fixed. Raise it to absorb slowly drifting background, lower it if genuine motion is being classified as background. |
| `min_visible_frames` | `5` | visible frames required before the world-fixed test is applied. |
| `descriptor_dirs` | `3` | maximum number of unit velocities kept per direction descriptor. More directions make the descriptor stricter on non-rigid motion. |
| `motion_threshold` | `0.01` | minimum velocity norm for a frame pair to enter a descriptor. In the units of the trajectories. |
| `min_co_visible_frames` | `3` | minimum number of shared frames for a direction edge, and minimum number of usable velocities for a point to receive a descriptor at all. Lower it for very short sequences, raise it for noisy ones. |
| `direction_cos_threshold` | `0.90` | mean cosine similarity required to link two points. This is the main knob on how coarse the grouping is: lower values merge more points, higher values fragment objects. |
| `direction_knn` | `50` | neighbours considered when building direction edges. `None` compares all pairs, which is quadratic and only sensible for a few hundred points. |
| `spatial_knn` | `10` | neighbourhood size of the spatial split of co-moving groups. |
| `world_fixed_spatial_knn` | `15` | neighbourhood size of the spatial split of the world-fixed class. Wider, because the background spans the whole scene rather than one object. |
| `min_group_size` | `3` | minimum size for a component to become a group rather than `independent_dynamic`. |

## Merging and fragment cleanup

| Field | Default | Effect |
| --- | --- | --- |
| `merge_direction_cos_threshold` | `0.85` | direction threshold of the motion-consistent merge. |
| `merge_distance_factor` | `3.0` | centroid distance threshold of the merge, in units of the local nearest-neighbour scale. |
| `merge_min_group_size` | `5` | minimum size for a group to enter the merge. |
| `merge_max_group_size` | `None` | optional upper size beyond which two groups are no longer merged. Unused at the default. |
| `fragment_merge_distance_factor` | `5.0` | distance threshold used when a fragment too small to be fitted is absorbed by its nearest neighbour, and when world-fixed subgroups are consolidated. |
| `min_fitting_size` | `5` | minimum size for a group to be fitted rather than absorbed. |

## Correction

| Field | Default | Effect |
| --- | --- | --- |
| `default_form` | `scale_about_camera_per_frame_translation` | form applied to co-moving groups. |
| `world_fixed_form` | `scale_about_anchor_translation` | form applied to world-fixed groups, because radial scaling about the camera center is ill-posed for a class with a large depth spread. |
| `low_parallax_form` | `scale_about_anchor_translation` | form applied to groups below the parallax threshold. |
| `parallax_threshold_deg` | `2.0` | median per-query parallax below which the anchor-centered form is used. `0` disables the branch, so every non-world-fixed group uses `default_form`. |
| `per_frame_scale_min_anchors` | `1` | minimum anchors for a per-frame scale. Raise it to stabilise the per-frame estimate on noisy anchors, at the cost of leaving more frames interpolated. |
| `per_frame_scale_smooth_sigma` | `2.0` | width of the Gaussian used to smooth the per-frame scale. Set it to `0` to disable smoothing. |

The available correction forms are:

| Form | Degrees of freedom | Notes |
| --- | --- | --- |
| `scale_about_camera_per_frame_translation` | `T + 3` | the default for co-moving groups |
| `scale_about_camera_translation` | `4` | a single radial scale for the group |
| `scale_about_anchor_translation` | `4` | the default for world-fixed and low-parallax groups |
| `scale_about_anchor` | `1` | the anchor-centered scale alone |
| `full_sim3` | `7` | the generic fallback; acts along directions the observations already constrain |
| `none` | `0` | leave the group uncorrected |

## Anchor budget

| Field | Default | Effect |
| --- | --- | --- |
| `target_anchor_frac` | `0.05` | budget as a fraction of the query points, `K = floor(rho * N)`. The high-budget range saturates rather than degrades, so 5% is a trade-off on per-anchor efficiency rather than on accuracy. |
| `anchor_allocation_strategy` | `motion_weighted` | `motion_weighted` scores a group by `n_g * (1 + m_g / max m_g')`; `proportional_size` scores by `n_g` alone and is the control of the allocation ablation. |
| `max_anchors_per_group` | `None` | optional per-group cap on top of the cap at the group size. |

## Interaction with your own data

- The budget is capped at the number of entries marked in `gt_valid`, so a user
  with few metric observations gets all of them used and a warning. Lower
  `target_anchor_frac` to match what you have if the warning is noise to you.
- `motion_threshold` and `world_fixed_std_threshold` are in the units of the
  trajectories. They do not need to be metric; they only have to be consistent
  with the scale of the input.
- `direction_knn`, `spatial_knn` and the group-size thresholds interact with `N`.
  On a sequence of a few thousand query points the defaults are appropriate; on a
  tiny example, as in `examples/`, the grouping will fragment objects, which is
  harmless for the correction but makes the group count look large.
