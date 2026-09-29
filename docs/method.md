# Method

GAUGE is a post-hoc geometric module. It reads the trajectories a tracker already
produced and writes corrected ones. This document describes the three stages, the
estimator each one uses, and where it lives in the code.

Notation: `T` frames, `N` query points, `P_i^t` the predicted position of point `i`
at frame `t`, `Q_i^t` its metric ground truth, `C_t` the camera center of frame
`t`, and `V` the predicted visibility mask.

## 1. Motion grouping (`gauge/grouping.py`)

The correction acts on motion groups rather than on single points, because the
radial bias is shared inside a group. Grouping therefore only has to make the
points of a group undergo approximately the same rigid motion; it is deliberately
not a semantic segmentation.

### 1.1 World-fixed points

A query is world-fixed when the Euclidean norm of the per-axis standard
deviations of its predicted position over the frames where it is visible stays
below `world_fixed_std_threshold` (0.05), provided it has at least
`min_visible_frames` (5) visible frames. These
points join a single world-fixed class that covers the background and barely
moving objects. `_world_fixed_mask` implements this.

### 1.2 Direction descriptor

For every other point, collect the velocities over consecutive co-visible frame
pairs whose norm exceeds `motion_threshold` (0.01),

```
v_i^t = P_i^(t+1) - P_i^t ,   for t with V_i^t = V_i^(t+1) = 1 and ||v_i^t|| > tau_motion.
```

keep the unit directions, and resample to at most `descriptor_dirs` (3)
directions, carried on the frame set `F^i`. A point with fewer than
`min_co_visible_frames` (3) usable velocities gets no descriptor and is left
uncorrected. See `_velocity_descriptor`.

Direction alone is unreliable for points with small motion, which is why the
grouping is not based on it alone.

### 1.3 Direction graph and spatial split

Build a kNN graph over the points carrying a descriptor, with
`direction_knn` (50) neighbours. Two neighbours `i` and `j` are connected when

```
|F_ij| >= tau_cov                                  (tau_cov = 3 shared frames)
mean_{t in F_ij} v_hat_i^t . v_hat_j^t >= tau_dir  (tau_dir = 0.90)
```

where `F_ij = F^i ∩ F^j`. Connected components of size at least `min_group_size`
(3) become co-moving groups; smaller ones are marked `independent_dynamic`. Each
co-moving group is then split once more by spatial kNN connectivity
(`spatial_knn` = 10) over the per-point representative positions, which are the
median positions over the visible frames. `_build_direction_graph`,
`_direction_clusters` and `_spatial_split_groups` implement this.

### 1.4 Merging and cleanup

Co-moving groups of size at least `merge_min_group_size` (5) are merged when both
their centroid distance is below `merge_distance_factor` (3.0) times the local
nearest-neighbour scale and their centroid direction descriptors agree above
`merge_direction_cos_threshold` (0.85). This repairs fragments cut by occlusion or
a brief tracking failure. The world-fixed class is split by spatial kNN with a
wider neighbourhood, `world_fixed_spatial_knn` (15), because the background spans
the whole scene, and small fragments are absorbed by the nearest subgroup within
`fragment_merge_distance_factor` (5.0) times the local scale. Any dynamic
fragment below `min_fitting_size` (5) is absorbed by the nearest group within the
same distance factor, and a fragment without a target becomes
`independent_dynamic`. See `_merge_motion_consistent_fragments`,
`_split_world_fixed_spatial` and `_merge_small_fragments`.

### 1.5 The result

Every query ends up in exactly one of three classes:

| Class | Treatment |
| --- | --- |
| `world_fixed` | corrected in the anchor-centered form (Section 3.2) |
| `co_moving` | corrected in the camera-centered form (Section 3.1) |
| `independent_dynamic` | not corrected, keeps its prediction |

Finer groups are favourable rather than a defect: the scale bias is more
consistent inside a smaller group. The grouping does not need to match semantic
instances, and on the paper's benchmark it usually produces more groups than
there are objects.

## 2. Anchor allocation and sampling (`gauge/anchors.py`)

The budget of a sequence is `K = floor(rho * N)` with `rho = target_anchor_frac`
(default 0.05). Each group is scored by

```
s_g = n_g * (1 + m_g / max_g' m_g')
```

where `n_g` is its number of valid distinct queries and `m_g` its mean 3D
displacement over co-visible frame pairs, so that the score rewards both size and
motion. The budget is split with the largest-remainder method, `k_g = floor(K s_g / sum s_g')`
rounded up by decreasing fractional part, capped at `n_g`. Groups of type
`independent_dynamic` receive nothing. `allocate_anchors_budget` implements this;
`proportional_size` scores by `n_g` alone and is kept as the control of the
allocation ablation.

Sampling then draws anchors, that is `(query, frame)` pairs, from the candidates
where predicted visibility and metric observation both hold. When a single frame
carries at least `k_g` candidates, all anchors of the group are drawn from that
frame, because common-frame anchors share the camera center and the object pose
and therefore give a consistent radial scale. Otherwise `k_g` distinct queries are
drawn and each contributes one random valid frame. See `sample_anchors`.

## 3. Group-wise geometric correction (`gauge/transforms.py`, `gauge/core.py`)

### 3.1 Co-moving groups: scaling about the camera center

Two scales of the same form are estimated. For a center `C` and a set `S` of
anchor point-frame pairs,

```
alpha(C, S) = median_{(i,t) in S}  ||Q_i^t - C|| / ||P_i^t - C||
```

The **group-level scale** takes `C = C_{t0}`, the camera center of the frame of
the first anchor, and `S = A_g`:

```
alpha_global = alpha(C_t0, A_g)        P~_i^t = C_t0 + alpha_global (P_i^t - C_t0)
```

The **per-frame scale** takes `C = C_t` and restricts `S` to the anchors that are
predicted visible and carry a metric observation at frame `t`:

```
alpha_t = alpha(C_t, {i : (i,t) in A_g})      P-_i^t = C_t + alpha_t (P~_i^t - C_t)
```

Two details matter. First, `alpha_t` is estimated from the raw prediction `P`, not
from `P~`, so the two steps are estimated independently and applied in cascade.
Second, frames with fewer than `per_frame_scale_min_anchors` (1) valid anchors are
linearly interpolated, smoothed with a Gaussian kernel of width
`per_frame_scale_smooth_sigma` (2.0), and any remaining gap falls back to
`alpha_global`. The per-frame step is what removes the time-varying drift; the
group-level step alone cannot cover it.

Finally one constant translation absorbs the rigid bias the two scaling steps
leave behind:

```
delta = median_{(i,t) in A_g} (Q_i^t - P-_i^t)         P^_i^t = P-_i^t + delta
```

The correction therefore has `T + 3` degrees of freedom per group: one radial
scale per frame plus three translational ones. `estimate_scale_about_camera_center`,
`estimate_scale_about_camera_center_per_frame`, `smooth_1d_scales`,
`estimate_group_translation` and `apply_group_correction` implement this.

### 3.2 Special groups: scaling about an anchor

World-fixed groups, and groups whose median per-query parallax falls below
`parallax_threshold_deg` (2.0), use the anchor-centered form instead. The reason
is numerical rather than structural: the depth spread of static points over the
whole background, and the near-collinearity of low-parallax points, make radial
scaling about the camera center ill-posed. The first in-group anchor `a` becomes
the local origin, and its ground-truth position becomes the center:

```
s      = median_{(i,t) in A_g \ {a}}  ||Q_i^t - Q_a^(0)|| / ||P_i^t - P_a^(0)||
P-_i^t = Q_a^(0) + s (P_i^t - P_a^(0))
```

followed by the same translation step as above. The anchor is matched exactly, at
the cost that the scale is no longer strictly along the view direction. See
`estimate_scale_about_anchor` and `apply_group_correction`.

Per-query parallax is the maximum angular separation between viewing rays over the
frames where the query is predicted visible, so it needs no ground truth; see
`compute_pred_parallax_per_query`.

### 3.3 Degenerate cases

The fallback chain has fewer degrees of freedom at the structured end and more at
the generic end:

```
camera-centered form  --fails-->  Sim(3), from >= 3 anchors  --fails-->  pure translation  --fails-->  uncorrected
anchor-centered form  --fails-->  pure translation            --fails-->  uncorrected
```

The Sim(3) step is fitted with Umeyama's method (`umeyama_sim3`) and is a last
resort, not an alternative: its rotation and anisotropic scale act along
directions the 2D observations already constrain. The paper measures that the
seven-degree-of-freedom fit is worse than no correction at all on four of the
eight trackers, under the same grouping and the same anchors.

## 4. Why the radial direction

Frame `t` projects a world point as `u_t(X) ~ K_t R_t (X - C_t)`. Moving the point
along its ray about the camera center, `X' = C_t + alpha (X - C_t)` with
`alpha > 0`, changes nothing:

```
u_t(X') ~ K_t R_t (X' - C_t) = alpha K_t R_t (X - C_t) ~ K_t R_t (X - C_t) = u_t(X)
```

The extra factor cancels when homogeneous coordinates are normalized, so no 2D
observation constrains the radial distance from a point to the camera center of
that frame. What the observations do constrain is the angular relation between
the point and the view direction. An anchor supplies a true radial distance
`||Q_i^t - C_t||`, and comparing it with the predicted `||P_i^t - C_t||` yields the
one scalar that this point needs at this frame.

Correction is therefore restricted to the camera ray: it leaves the projection
unchanged and preserves the tangential component of the prediction, which an
anisotropic scale would break. The consequence is that the correction can only be
partial. Trackers trained with metric supervision already constrain the view
direction weakly rather than leaving it free, so what GAUGE recovers is bounded
above by the part of the error that lives in this family, not by the number of
anchors.
