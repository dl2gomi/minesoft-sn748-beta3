from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from geometry.mesh.schemas import MeshData


def _sample_points_normals(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    pts, face_idx = trimesh.sample.sample_surface(mesh, count=count)
    normals = mesh.face_normals[face_idx]
    return pts, normals


def _parallel_shell_score(
    inner_pts: np.ndarray,
    inner_normals: np.ndarray,
    outer_pts: np.ndarray,
    outer_normals: np.ndarray,
    dist_threshold: float,
    *,
    knn_k: int = 56,
    opposing_dot: float = -0.22,
    opposing_dot_relaxed: float = -0.12,
    distance_relax: float = 2.25,
) -> float:
    """
    Fraction of inner samples that see a *close, opposing* outer surface among k-NN outer points.

    Euclidean 1-NN is wrong here: the closest point on the outer *mesh* is often the exterior
    or an unrelated sheet, so inner·outer ≈ 0. We search k nearest *sampled* outer points and
    require a candidate that is both within distance and (approximately) facing the inner sample.
    """
    if inner_pts.shape[0] == 0 or outer_pts.shape[0] == 0:
        return 0.0

    no = outer_pts.shape[0]
    k = int(min(max(2, knn_k), no))
    tree = cKDTree(outer_pts)
    try:
        dd, ii = tree.query(inner_pts, k=k, workers=-1)
    except TypeError:
        dd, ii = tree.query(inner_pts, k=k)

    dd = np.asarray(dd, dtype=np.float64)
    ii = np.asarray(ii, dtype=np.int64)
    if dd.ndim == 1:
        dd = dd.reshape(-1, 1)
        ii = ii.reshape(-1, 1)

    outer_n = outer_normals[ii]
    inner_n = inner_normals[:, None, :]
    dots = np.sum(inner_n * outer_n, axis=2)

    strict = (dd <= dist_threshold) & (dots <= opposing_dot)
    relaxed = (dd <= dist_threshold * distance_relax) & (dots <= opposing_dot_relaxed)
    hit = np.any(strict | relaxed, axis=1)
    return float(np.mean(hit))


def remove_parallel_internal_shells(mesh_data: MeshData, **kwargs) -> MeshData:
    """Parallel-wall-only pass. Prefer `run_trellis_shell_cleanup` (single split) in Trellis."""
    from geometry.mesh.shell_cleanup import run_trellis_shell_cleanup

    return run_trellis_shell_cleanup(mesh_data, parallel=kwargs, internal={"enabled": False})

