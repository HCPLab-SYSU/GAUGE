# Adapting GAUGE to your own tracker

GAUGE reads trajectories, not images, so adapting it is about producing five
arrays. This document covers the contract, the three shapes of input you are
likely to have, and the mistakes that fail quietly if you do not know about them.

## 1. The contract

| Argument | Shape | Meaning |
| --- | --- | --- |
| `pred_m` | `[T, N, 3]` | predicted trajectories, already aligned to the world frame |
| `pred_visible` | `[T, N]` bool | tracker-predicted visibility |
| `gt_m` | `[T, N, 3]` | metric ground truth, read **only** where `gt_valid` is True |
| `gt_valid` | `[T, N]` bool | the entries where a metric observation exists |
| `centers` | `[T, 3]` | per-frame camera center, in the same frame as `pred_m` |

Nothing else is read. In particular, per-point confidence and the query pixel
coordinates are not part of the interface: they belong to whatever query-selection
protocol you use, not to the correction.

The module is deterministic in `seed`. Two calls with the same inputs and the same
seed return bitwise identical trajectories.

## 2. Getting the frame right

`pred_m` and `centers` must be expressed in the same coordinate frame, and that
frame must be a world frame rather than the camera frame. GAUGE scales each group
radially about the camera center, which is not defined in the camera frame.

This is the single most common mistake, so `gauge.validate_inputs` checks for it.
If every camera center is zero, you get:

```
GAUGEInputWarning: all camera centers are zero. GAUGE scales each group radially
about the camera center, which is not defined in the camera frame. ...
```

### From world-to-camera matrices

Most datasets and trackers expose `X_cam = R X_world + t`. The center is
`C = -R^T t`:

```python
from gauge import camera_centers_from_world_to_camera

centers = camera_centers_from_world_to_camera(extrinsics_w2c)  # [T, 3, 4] or [T, 4, 4]
```

### From camera-to-world matrices

If you have `X_world = R X_cam + t`, then `C = t` directly.

### If your tracker only works in the camera frame

Convert the trajectories to the world frame first, using your own extrinsics. Do
not pass camera-frame coordinates and hope: a fixed camera center degenerates the
per-frame radial scale into a single global scale, which is exactly the case GAUGE
is designed to go beyond.

### If the camera does not move

A static camera is legal, but you get a warning, because the per-frame radial
scale then has nothing to correct. In that case `scale_about_camera_translation`
or a pure global scale is the honest baseline.

## 3. Visibility

If your tracker reports visibility, pass it. If it does not, pass all `True`; the
correction is driven by the anchors and by the grouping, and visibility only
controls which frame pairs enter the direction descriptors and which anchors are
considered observable.

`pred_visible` and `gt_valid` are different things. Visibility is what the tracker
believes, validity is what you actually measured. A point may be visible according
to the tracker and have no metric observation, and the other way round.

## 4. Anchors: the three cases

GAUGE needs an external source of true scale, so it needs at least a few metric
observations. Which case you are in decides how you fill `gt_m` and `gt_valid`.

### Case A: you have dense metric ground truth

For instance when evaluating on a synthetic benchmark. Pass it and mark validity:

```python
gt_valid = gt_valid_from_dataset        # [T, N] bool, the same mask you use for evaluation
corrected = apply_gauge(pred_m, gt_m, pred_visible, gt_valid, centers)
```

The anchors are then drawn at random from the valid entries, with the configured
budget.

### Case B: you have a few sparse metric observations

This is the realistic deployment case: sparse LiDAR returns, SLAM keyframes, or
RGB-D measurements. You do not need to build a dense ground-truth volume.
`make_anchor_arrays` allocates the pair for you and fills the unobserved entries
with `NaN`:

```python
from gauge import make_anchor_arrays

# observations: (query index, frame index, x, y, z) for each point you measured
gt_m, gt_valid = make_anchor_arrays(T, N, observations)
corrected = apply_gauge(pred_m, gt_m, pred_visible, gt_valid, centers)
```

Note the budget behaviour: the budget is `floor(target_anchor_frac * N)`, but it
is capped at the number of entries marked in `gt_valid`. If you measured fewer
than the budget asks for, all of your observations are used and you get an
informative warning:

```
GAUGEInputWarning: the anchor budget is floor(0.05 * 168) = 8 but only 5
(frame, point) observations are marked in gt_valid, so the budget is capped at 5.
```

Set `target_anchor_frac` to match what you have if you would rather not see it.

### Case C: you have no metric information at all

Then GAUGE cannot run, and this is by design rather than a limitation of the
implementation. Monocular observations do not constrain the radial distance to the
camera, so something outside the video has to supply a scale. Passing an empty
`gt_valid` raises:

```
ValueError: gt_valid is empty, so no metric observation is available and nothing
can be corrected. Mark the (frame, point) pairs where you have a metric
measurement; see gauge.adapt.make_anchor_arrays.
```

## 5. What gets corrected

Every query is placed in one of three classes, and only the first two are
touched:

| Class | Corrected | Notes |
| --- | --- | --- |
| `co_moving` | yes | per-frame radial scale about the camera center plus one translation |
| `world_fixed` | yes | anchor-centered scale plus one translation |
| `independent_dynamic` | no | keeps its prediction |

Points with too few usable velocities, and groups with no anchors, also keep their
prediction. `apply_gauge_with_diagnostics` reports how many groups were actually
solved:

```python
corrected, diag = apply_gauge_with_diagnostics(...)
print(diag["n_groups_solved"], "of", diag["n_groups_total"])
print(diag["n_anchors_used"], "anchors used")
print([g["group_type"] for g in diag["groups"]])
```

## 6. Supplying your own groups

If you already have a segmentation, a motion clustering or an instance
segmentation, inject it and skip the built-in grouping:

```python
corrected = apply_gauge(
    pred_m, gt_m, pred_visible, gt_valid, centers,
    groups=[
        {"group_type": "world_fixed", "indices": np.nonzero(background)[0]},
        {"group_type": "co_moving",   "indices": np.nonzero(object_mask)[0]},
    ],
)
```

The contract is one dict per group with `"indices"` (1-D integer array) and
`"group_type"` (one of `world_fixed`, `co_moving`, `independent_dynamic`), which
is the same format `gauge.build_motion_groups` returns and
`apply_gauge_with_diagnostics` reports. `gauge.validate_groups` checks it and
raises a readable error when it is broken:

```
ValueError: group 1 repeats query indices already assigned to an earlier group
(for example [3, 4]); every query may belong to at most one group
```

Two things worth knowing when you supply your own groups:

- The grouping in the paper is deliberately finer than semantic instances. If
  your segmentation is coarse, the within-group scale bias will be less
  consistent and the gain will be smaller, not larger.
- `independent_dynamic` groups are skipped, so it is valid to list the points you
  do not want corrected, or to omit them entirely.

## 7. Common failure modes

| Symptom | Likely cause |
| --- | --- |
| The output equals the input | No group was solved. Check `gt_valid` is not empty, that the budget is at least one anchor per group, and that the groups are not all `independent_dynamic`. |
| A warning about zero camera centers | `pred_m` is in the camera frame. Convert it to the world frame. |
| A warning that the budget is capped | You marked fewer observations than the budget asks for. Expected in a sparse-anchor setup. |
| Almost every group is `independent_dynamic` | The direction test is too strict for the data, or the sequence has too few co-visible frames. Try lowering `direction_cos_threshold`, or `min_co_visible_frames`. |
| Many small groups for what looks like one object | Expected. The grouping targets consistent motion rather than semantic units, and finer groups are favourable. |
| The result is worse than the baseline | Check the frame question first. Then check whether your groups are much coarser than the motion, which makes the per-group scale a poor model. |

## 8. Sequences, seeds and cost

The interface takes one sequence at a time. Loop over your sequences and keep the
same `seed` per sequence if you want reproducible anchor sampling.

The correction adds no training and no optimization. On a single CPU core the
paper reports a median of 0.49 s per sequence under its dynamic-query protocol
and 2.60 s under the Full protocol, with no GPU, no backpropagation and no
optimizer.
