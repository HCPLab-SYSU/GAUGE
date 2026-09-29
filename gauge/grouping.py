"""Unsupervised motion grouping for the GAUGE correction module.

The grouping has four stages: direction-based clustering over the velocity
descriptors, a spatial kNN split inside every cluster, a merge of fragments whose
centroid motions are consistent, and a cleanup that reassigns fragments too small
to be fitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gauge.config import GAUGEConfig

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# The three group types produced by the grouping and understood by the
# correction. ``independent_dynamic`` points take no part in the correction and
# keep their uncorrected prediction.
GROUP_TYPES = ("world_fixed", "co_moving", "independent_dynamic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_positions(pred_coords: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Return a [N, 3] representative position for every point."""
    T, N = visibility.shape
    positions = np.full((N, 3), np.nan, dtype=np.float64)
    for i in range(N):
        vis = visibility[:, i]
        if vis.any():
            positions[i] = np.median(pred_coords[vis, i], axis=0)
        else:
            positions[i] = pred_coords[0, i]
    return positions


def _connected_components(adj: list[set[int]]) -> list[list[int]]:
    """Return connected components for a graph represented as adjacency lists."""
    N = len(adj)
    visited = [False] * N
    components: list[list[int]] = []
    for start in range(N):
        if visited[start]:
            continue
        comp: list[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(comp)
    return components


def _fragment_scale(indices: np.ndarray, positions: np.ndarray) -> float:
    """Median nearest-neighbor distance within a set of points."""
    if indices.size < 2:
        return 1.0
    pos = positions[indices]
    if not np.isfinite(pos).all():
        return 1.0
    if cKDTree is None:
        raise RuntimeError("_fragment_scale requires scipy")
    tree = cKDTree(pos)
    k = min(2, indices.size)
    dist, _ = tree.query(pos, k=k)
    if dist.ndim == 1:
        dist = dist[:, None]
    nn = dist[:, 1:]
    vals = nn[nn > 0]
    return float(np.median(vals)) if vals.size > 0 else 1.0


# ---------------------------------------------------------------------------
# World-fixed / static detection
# ---------------------------------------------------------------------------


def _world_fixed_mask(
    pred_coords: np.ndarray,
    visibility: np.ndarray,
    world_fixed_std_threshold: float,
    min_visible_frames: int,
) -> np.ndarray:
    """Return [N] bool mask of points considered static in world coordinates."""
    T, N = visibility.shape
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        vis = visibility[:, i]
        if vis.sum() < min_visible_frames:
            continue
        pts = pred_coords[vis, i]
        std = float(np.linalg.norm(np.std(pts, axis=0)))
        if np.isfinite(std) and std < world_fixed_std_threshold:
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Multi-frame direction descriptor
# ---------------------------------------------------------------------------


def _velocity_descriptor(
    pred_coords: np.ndarray,
    visibility: np.ndarray,
    descriptor_dirs: int,
    motion_threshold: float,
    min_co_visible_frames: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Build a consecutive-velocity descriptor for each point."""
    T, N = visibility.shape
    is_dynamic = np.zeros(N, dtype=bool)
    descriptors: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []

    for i in range(N):
        valid_t = np.nonzero(visibility[:-1, i] & visibility[1:, i])[0]
        if valid_t.size < min_co_visible_frames:
            continue

        velocities = pred_coords[valid_t + 1, i] - pred_coords[valid_t, i]
        norms = np.linalg.norm(velocities, axis=1)
        usable = norms > motion_threshold
        if usable.sum() < min_co_visible_frames:
            continue

        vels = velocities[usable]
        norms = norms[usable]
        vels_norm = vels / norms[:, None]
        t_idx = valid_t[usable]

        if descriptor_dirs is not None and descriptor_dirs > 0 and vels_norm.shape[0] > descriptor_dirs:
            idx = np.linspace(0, vels_norm.shape[0] - 1, descriptor_dirs, dtype=int)
            vels_norm = vels_norm[idx]
            t_idx = t_idx[idx]

        descriptors.append(vels_norm.astype(np.float64))
        frame_indices.append(t_idx.astype(np.int64))
        is_dynamic[i] = True

    return is_dynamic, descriptors, frame_indices


def _direction_similarity(
    desc_i: np.ndarray,
    frames_i: np.ndarray,
    desc_j: np.ndarray,
    frames_j: np.ndarray,
    min_co_visible_frames: int,
) -> float | None:
    """Average cosine similarity over co-visible velocity frames."""
    common, i_idx, j_idx = np.intersect1d(frames_i, frames_j, return_indices=True)
    if common.size < min_co_visible_frames:
        return None
    dots = np.einsum("ij,ij->i", desc_i[i_idx], desc_j[j_idx])
    return float(np.clip(dots.mean(), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Direction graph + clustering
# ---------------------------------------------------------------------------


def _build_direction_graph(
    is_dynamic: np.ndarray,
    descriptors: list[np.ndarray],
    frame_indices: list[np.ndarray],
    direction_cos_threshold: float,
    min_co_visible_frames: int,
    positions: np.ndarray,
    direction_knn: int | None,
) -> list[set[int]]:
    """Build an adjacency list over dynamic points based on direction similarity."""
    N = is_dynamic.shape[0]
    dynamic_map = np.nonzero(is_dynamic)[0]
    M = dynamic_map.size

    adj: list[set[int]] = [set() for _ in range(N)]
    if M == 0:
        return adj

    local_adj: list[set[int]] = [set() for _ in range(M)]

    def _add_edge(a: int, b: int) -> None:
        if a == b:
            return
        sim = _direction_similarity(
            descriptors[a],
            frame_indices[a],
            descriptors[b],
            frame_indices[b],
            min_co_visible_frames,
        )
        if sim is not None and sim >= direction_cos_threshold:
            local_adj[a].add(b)
            local_adj[b].add(a)

    if direction_knn is None or M <= direction_knn + 1:
        for a in range(M):
            for b in range(a + 1, M):
                _add_edge(a, b)
    else:
        if cKDTree is None:
            raise RuntimeError("direction_knn requires scipy")
        dyn_pos = positions[dynamic_map]
        tree = cKDTree(dyn_pos)
        k = min(direction_knn + 1, M)
        _, neigh = tree.query(dyn_pos, k=k)
        if neigh.ndim == 1:
            neigh = neigh[:, None]
        for a in range(M):
            for b_local in neigh[a, 1:]:
                b = int(b_local)
                if a < b:
                    _add_edge(a, b)

    for a in range(M):
        i = dynamic_map[a]
        for b in local_adj[a]:
            adj[i].add(dynamic_map[b])
    return adj


def _direction_clusters(
    adj: list[set[int]],
    is_dynamic: np.ndarray,
) -> list[np.ndarray]:
    """Return connected components over dynamic nodes."""
    N = is_dynamic.shape[0]
    dynamic_map = np.nonzero(is_dynamic)[0]
    M = dynamic_map.size
    if M == 0:
        return []

    local_adj: list[set[int]] = [set() for _ in range(M)]
    local_index = {int(global_idx): local_idx for local_idx, global_idx in enumerate(dynamic_map)}
    for local_idx, global_idx in enumerate(dynamic_map):
        for nb in adj[int(global_idx)]:
            local_nb = local_index.get(int(nb))
            if local_nb is not None:
                local_adj[local_idx].add(local_nb)

    components = _connected_components(local_adj)
    return [dynamic_map[np.array(comp, dtype=int)] for comp in components]


# ---------------------------------------------------------------------------
# Spatial split
# ---------------------------------------------------------------------------


def _spatial_split_groups(
    groups: list[dict[str, Any]],
    positions: np.ndarray,
    spatial_knn: int,
    min_group_size: int,
) -> list[dict[str, Any]]:
    """Split co-moving groups by spatial kNN connectivity."""
    if spatial_knn <= 0:
        return groups

    new_groups: list[dict[str, Any]] = []
    for g in groups:
        if g.get("group_type") != "co_moving":
            new_groups.append(g)
            continue

        indices = np.asarray(g["indices"])
        M = indices.size
        if M <= min_group_size:
            new_groups.append(g)
            continue

        pos = positions[indices]
        if cKDTree is None:
            raise RuntimeError("the spatial split requires scipy")
        tree = cKDTree(pos)
        k = min(spatial_knn + 1, M)
        _, neigh = tree.query(pos, k=k)
        if neigh.ndim == 1:
            neigh = neigh[:, None]

        adj = [set() for _ in range(M)]
        for a in range(M):
            for b_local in neigh[a, 1:]:
                b = int(b_local)
                adj[a].add(b)
                adj[b].add(a)

        components = _connected_components(adj)
        for comp in components:
            if len(comp) >= min_group_size:
                new_groups.append({"group_type": "co_moving", "indices": indices[comp].copy()})
            else:
                new_groups.append({"group_type": "independent_dynamic", "indices": indices[comp].copy()})
    return new_groups


# ---------------------------------------------------------------------------
# Motion-consistent fragment merge
# ---------------------------------------------------------------------------


def _group_direction_descriptor(
    pred_coords: np.ndarray,
    visibility: np.ndarray,
    indices: np.ndarray,
    descriptor_dirs: int | None,
    motion_threshold: float,
    min_visible_frames: int = 2,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a centroid-based normalized-velocity descriptor for a group."""
    T = pred_coords.shape[0]
    velocities: list[np.ndarray] = []
    frames: list[int] = []

    for t in range(T - 1):
        vis = visibility[t, indices] & visibility[t + 1, indices]
        if vis.sum() < min_visible_frames:
            continue
        c1 = np.median(pred_coords[t, indices[vis]], axis=0)
        c2 = np.median(pred_coords[t + 1, indices[vis]], axis=0)
        v = c2 - c1
        norm = float(np.linalg.norm(v))
        if norm > motion_threshold:
            velocities.append((v / norm).astype(np.float64))
            frames.append(t)

    if not velocities:
        return None, None

    vels = np.array(velocities, dtype=np.float64)
    frame_arr = np.array(frames, dtype=np.int64)
    if descriptor_dirs is not None and descriptor_dirs > 0 and vels.shape[0] > descriptor_dirs:
        idx = np.linspace(0, vels.shape[0] - 1, descriptor_dirs, dtype=int)
        vels = vels[idx]
        frame_arr = frame_arr[idx]
    return vels, frame_arr


def _merge_motion_consistent_fragments(
    groups: list[dict[str, Any]],
    pred_coords: np.ndarray,
    visibility: np.ndarray,
    positions: np.ndarray,
    descriptor_dirs: int | None,
    motion_threshold: float,
    min_co_visible_frames: int,
    merge_direction_cos_threshold: float,
    merge_distance_factor: float,
    merge_min_group_size: int,
    merge_max_group_size: int | None,
) -> list[dict[str, Any]]:
    """Merge over-segmented fragments whose centroid motions are consistent."""
    if merge_direction_cos_threshold <= 0 or merge_distance_factor <= 0:
        return groups

    eligible: list[dict[str, Any]] = []
    for gi, g in enumerate(groups):
        if g.get("group_type") != "co_moving":
            continue
        idx = np.asarray(g["indices"])
        if idx.size < merge_min_group_size:
            continue
        desc, frames = _group_direction_descriptor(
            pred_coords, visibility, idx, descriptor_dirs, motion_threshold
        )
        if desc is None or desc.shape[0] < min_co_visible_frames:
            continue
        eligible.append(
            {
                "gi": gi,
                "indices": idx,
                "desc": desc,
                "frames": frames,
                "centroid": np.median(positions[idx], axis=0),
                "scale": _fragment_scale(idx, positions),
                "size": idx.size,
            }
        )

    if len(eligible) < 2:
        return groups

    parent = list(range(len(eligible)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    max_size = merge_max_group_size if merge_max_group_size is not None else int(1e9)
    centroids = np.array([e["centroid"] for e in eligible], dtype=np.float64)
    tree = cKDTree(centroids)

    for i, ei in enumerate(eligible):
        threshold = merge_distance_factor * max(ei["scale"], 1e-6)
        neighbors = tree.query_ball_point(ei["centroid"], r=threshold)
        for j in neighbors:
            if j <= i:
                continue
            ej = eligible[j]
            if ei["size"] > max_size and ej["size"] > max_size:
                continue
            sim = _direction_similarity(
                ei["desc"],
                ei["frames"],
                ej["desc"],
                ej["frames"],
                min_co_visible_frames,
            )
            if sim is not None and sim >= merge_direction_cos_threshold:
                union(i, j)

    components: dict[int, list[int]] = {}
    for i, e in enumerate(eligible):
        root = find(i)
        components.setdefault(root, []).append(e["gi"])

    merged_gis = set()
    new_groups: list[dict[str, Any]] = []
    for comp_gis in components.values():
        merged_gis.update(comp_gis)
        if len(comp_gis) == 1:
            new_groups.append(groups[comp_gis[0]].copy())
        else:
            merged_indices = np.concatenate([groups[gi]["indices"] for gi in comp_gis])
            new_groups.append({"group_type": "co_moving", "indices": merged_indices})

    for gi, g in enumerate(groups):
        if gi not in merged_gis:
            new_groups.append(g.copy())

    return new_groups


# ---------------------------------------------------------------------------
# World-fixed spatial split
# ---------------------------------------------------------------------------


def _split_world_fixed_spatial(
    world_fixed_indices: np.ndarray,
    positions: np.ndarray,
    spatial_knn: int = 15,
    min_group_size: int = 3,
    min_fitting_size: int = 5,
    fragment_merge_distance_factor: float = 5.0,
) -> list[np.ndarray]:
    """Split a world_fixed group into spatially connected subgroups."""
    indices = np.asarray(world_fixed_indices, dtype=int)
    M = indices.size
    if M < min_group_size:
        return [indices]

    pos = positions[indices]
    if not np.isfinite(pos).all():
        return [indices]

    if cKDTree is None:
        raise RuntimeError("world_fixed spatial split requires scipy")

    tree = cKDTree(pos)
    k = min(spatial_knn + 1, M)
    _, neigh = tree.query(pos, k=k)
    if neigh.ndim == 1:
        neigh = neigh[:, None]

    adj = [set() for _ in range(M)]
    for a in range(M):
        for b_local in neigh[a, 1:]:
            b = int(b_local)
            adj[a].add(b)
            adj[b].add(a)

    components = _connected_components(adj)
    visited = set()
    for comp in components:
        for node in comp:
            visited.add(node)
    for a in range(M):
        if a not in visited:
            components.append([a])

    subgroups = [indices[np.array(comp, dtype=int)] for comp in components]
    if len(subgroups) <= 1:
        return subgroups

    centroids = []
    sizes = []
    for sg in subgroups:
        centroids.append(np.median(positions[sg], axis=0))
        sizes.append(sg.size)
    centroids = np.array(centroids, dtype=np.float64)
    sizes = np.array(sizes, dtype=int)

    merged = [False] * len(subgroups)
    result: list[np.ndarray] = []

    def _fragment_scale_local(idx: np.ndarray) -> float:
        return _fragment_scale(idx, positions)

    while True:
        candidates = [i for i in range(len(subgroups)) if not merged[i] and sizes[i] < min_fitting_size]
        if not candidates:
            break
        i = min(candidates, key=lambda x: sizes[x])
        idx_i = subgroups[i]
        merged[i] = True

        scale = max(_fragment_scale_local(idx_i), 1e-6)
        threshold = fragment_merge_distance_factor * scale

        best_j = None
        best_dist = float("inf")
        for j in range(len(subgroups)):
            if j == i or merged[j]:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j is not None and best_dist < threshold:
            subgroups[best_j] = np.concatenate([subgroups[best_j], idx_i])
            centroids[best_j] = np.median(positions[subgroups[best_j]], axis=0)
            sizes[best_j] = subgroups[best_j].size
        else:
            result.append(idx_i.copy())

    for i in range(len(subgroups)):
        if not merged[i]:
            result.append(subgroups[i])

    return result


# ---------------------------------------------------------------------------
# Fragment merge
# ---------------------------------------------------------------------------


def _merge_small_fragments(
    groups: list[dict[str, Any]],
    positions: np.ndarray,
    min_fitting_size: int,
    fragment_merge_distance_factor: float = 5.0,
) -> list[dict[str, Any]]:
    """Merge tiny dynamic groups into the nearest group with enough points."""
    if min_fitting_size <= 1:
        return groups

    fixed_groups = [g for g in groups if g.get("group_type") == "world_fixed"]
    dynamic_groups = [g for g in groups if g.get("group_type") != "world_fixed"]

    centroids = []
    sizes = []
    for g in dynamic_groups:
        idx = np.asarray(g["indices"])
        sizes.append(idx.size)
        centroids.append(np.median(positions[idx], axis=0))
    centroids = np.array(centroids, dtype=np.float64)
    sizes = np.array(sizes, dtype=int)

    if len(dynamic_groups) == 0:
        return fixed_groups

    merged = [False] * len(dynamic_groups)
    result: list[dict[str, Any]] = []

    def _fragment_scale_local(idx: np.ndarray) -> float:
        return _fragment_scale(idx, positions)

    while True:
        candidates = [i for i in range(len(dynamic_groups)) if not merged[i] and sizes[i] < min_fitting_size]
        if not candidates:
            break
        i = min(candidates, key=lambda x: sizes[x])
        idx_i = np.asarray(dynamic_groups[i]["indices"])
        merged[i] = True

        scale = max(_fragment_scale_local(idx_i), 1e-6)
        threshold = fragment_merge_distance_factor * scale

        best_j = None
        best_dist = float("inf")
        for j in range(len(dynamic_groups)):
            if j == i or merged[j]:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j is not None and best_dist < threshold:
            dynamic_groups[best_j]["indices"] = np.concatenate(
                [np.asarray(dynamic_groups[best_j]["indices"]), idx_i]
            )
            centroids[best_j] = np.median(positions[np.asarray(dynamic_groups[best_j]["indices"])], axis=0)
            sizes[best_j] = len(dynamic_groups[best_j]["indices"])
            if sizes[best_j] >= min_fitting_size:
                dynamic_groups[best_j]["group_type"] = "co_moving"
        else:
            result.append({"group_type": "independent_dynamic", "indices": idx_i.copy()})

    for i in range(len(dynamic_groups)):
        if not merged[i]:
            result.append(dynamic_groups[i])

    return fixed_groups + result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def cluster_motion_groups(
    pred_coords: np.ndarray,
    visibility: np.ndarray,
    descriptor_dirs: int = 3,
    direction_cos_threshold: float = 0.90,
    spatial_knn: int = 10,
    motion_threshold: float = 0.01,
    world_fixed_std_threshold: float = 0.05,
    world_fixed_spatial_knn: int = 15,
    min_group_size: int = 3,
    min_fitting_size: int = 5,
    min_visible_frames: int = 5,
    min_co_visible_frames: int = 3,
    direction_knn: int | None = 50,
    merge_direction_cos_threshold: float = 0.85,
    merge_distance_factor: float = 3.0,
    merge_min_group_size: int = 5,
    merge_max_group_size: int | None = None,
    fragment_merge_distance_factor: float = 5.0,
) -> list[dict[str, Any]]:
    """Partition predicted trajectories into motion groups.

    Every returned group is a dict with an ``indices`` array and a
    ``group_type``, one of ``world_fixed``, ``co_moving`` and
    ``independent_dynamic``. Only the first two take part in the correction;
    ``independent_dynamic`` points keep their uncorrected prediction.

    Args:
        pred_coords: [T, N, 3] predicted trajectories.
        visibility: [T, N] bool predicted visibility.
        descriptor_dirs: maximum number of unit velocities kept per descriptor.
        direction_cos_threshold: mean cosine similarity required for an edge.
        spatial_knn: neighbourhood size of the spatial split of co-moving groups.
        motion_threshold: minimum velocity norm for a frame pair to enter a
            descriptor.
        world_fixed_std_threshold: position standard deviation below which a
            point counts as world-fixed.
        world_fixed_spatial_knn: neighbourhood size of the spatial split of the
            world-fixed group, which is wider because the background is spread
            over the whole scene.
        min_group_size: minimum size for a component to become a real group.
        min_fitting_size: minimum size for a group to be fitted instead of being
            merged into a neighbour.
        min_visible_frames: minimum number of visible frames for a world-fixed
            decision.
        min_co_visible_frames: minimum number of shared frames for a direction
            edge.
        direction_knn: neighbours considered for direction edges; None compares
            all pairs.
        merge_direction_cos_threshold: direction threshold of the fragment merge.
        merge_distance_factor: centroid distance threshold of the fragment merge,
            in units of the local nearest-neighbour scale.
        merge_min_group_size: minimum size for a group to enter the merge.
        merge_max_group_size: optional upper size beyond which two groups are no
            longer merged.
        fragment_merge_distance_factor: distance threshold used when a fragment
            too small to be fitted is absorbed by its nearest neighbour.

    Returns:
        The list of motion groups.
    """
    pred_coords = np.asarray(pred_coords, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    if pred_coords.ndim != 3 or pred_coords.shape[2] != 3:
        raise ValueError(f"pred_coords must be (T, N, 3), got {pred_coords.shape}")
    if visibility.shape != pred_coords.shape[:2]:
        raise ValueError(
            f"visibility shape {visibility.shape} does not match pred_coords shape {pred_coords.shape[:2]}"
        )

    N = pred_coords.shape[1]
    positions = _median_positions(pred_coords, visibility)

    world_fixed = np.nonzero(
        _world_fixed_mask(
            pred_coords, visibility, world_fixed_std_threshold, min_visible_frames
        )
    )[0]

    is_dynamic, descriptors, frame_indices = _velocity_descriptor(
        pred_coords, visibility, descriptor_dirs, motion_threshold, min_co_visible_frames
    )
    is_dynamic = is_dynamic & ~np.isin(np.arange(N), world_fixed)

    adj = _build_direction_graph(
        is_dynamic,
        descriptors,
        frame_indices,
        direction_cos_threshold,
        min_co_visible_frames,
        positions,
        direction_knn=direction_knn,
    )
    components = _direction_clusters(adj, is_dynamic)

    groups: list[dict[str, Any]] = []
    if world_fixed.size > 0:
        groups.append({"group_type": "world_fixed", "indices": world_fixed.copy()})

    for comp in components:
        if comp.size >= min_group_size:
            groups.append({"group_type": "co_moving", "indices": comp.copy()})
        else:
            groups.append({"group_type": "independent_dynamic", "indices": comp.copy()})

    groups = _spatial_split_groups(groups, positions, spatial_knn, min_group_size)

    groups = _merge_motion_consistent_fragments(
        groups,
        pred_coords,
        visibility,
        positions,
        descriptor_dirs=descriptor_dirs,
        motion_threshold=motion_threshold,
        min_co_visible_frames=min_co_visible_frames,
        merge_direction_cos_threshold=merge_direction_cos_threshold,
        merge_distance_factor=merge_distance_factor,
        merge_min_group_size=merge_min_group_size,
        merge_max_group_size=merge_max_group_size,
    )

    groups_no_wf = [g for g in groups if g.get("group_type") != "world_fixed"]
    wf_groups = [g for g in groups if g.get("group_type") == "world_fixed"]
    if wf_groups:
        wf_subgroups = _split_world_fixed_spatial(
            wf_groups[0]["indices"],
            positions,
            spatial_knn=world_fixed_spatial_knn,
            min_group_size=min_group_size,
            min_fitting_size=min_fitting_size,
            fragment_merge_distance_factor=fragment_merge_distance_factor,
        )
        groups = [
            {"group_type": "world_fixed", "indices": idx} for idx in wf_subgroups
        ] + groups_no_wf

    groups = _merge_small_fragments(
        groups,
        positions,
        min_fitting_size,
        fragment_merge_distance_factor=fragment_merge_distance_factor,
    )

    return groups


def build_motion_groups(
    pred_m: np.ndarray,
    pred_visible: np.ndarray,
    config: GAUGEConfig,
) -> list[dict[str, Any]]:
    """Build the motion groups of a sequence from a :class:`GAUGEConfig`."""
    return cluster_motion_groups(
        pred_m,
        pred_visible,
        descriptor_dirs=config.descriptor_dirs,
        direction_cos_threshold=config.direction_cos_threshold,
        spatial_knn=config.spatial_knn,
        motion_threshold=config.motion_threshold,
        world_fixed_std_threshold=config.world_fixed_std_threshold,
        world_fixed_spatial_knn=config.world_fixed_spatial_knn,
        min_group_size=config.min_group_size,
        min_fitting_size=config.min_fitting_size,
        min_visible_frames=config.min_visible_frames,
        min_co_visible_frames=config.min_co_visible_frames,
        direction_knn=config.direction_knn,
        merge_direction_cos_threshold=config.merge_direction_cos_threshold,
        merge_distance_factor=config.merge_distance_factor,
        merge_min_group_size=config.merge_min_group_size,
        merge_max_group_size=config.merge_max_group_size,
        fragment_merge_distance_factor=config.fragment_merge_distance_factor,
    )
