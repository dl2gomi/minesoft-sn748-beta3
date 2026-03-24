from __future__ import annotations

import numpy as np
import torch
import trimesh
from logger_config import logger

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
) -> float:
    """
    Score how much inner points look like a close parallel shell to outer points.
    High score => close distance + opposite normals.
    """
    if inner_pts.shape[0] == 0 or outer_pts.shape[0] == 0:
        return 0.0

    # Pairwise distance on sampled points only (bounded sizes), then nearest outer.
    # Shapes: (Ni, 1, 3) and (1, No, 3) -> (Ni, No)
    diff = inner_pts[:, None, :] - outer_pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(d2, axis=1)
    nearest_dist = np.sqrt(d2[np.arange(len(nearest_idx)), nearest_idx])
    nearest_outer_normals = outer_normals[nearest_idx]

    dots = np.einsum("ij,ij->i", inner_normals, nearest_outer_normals)
    close = nearest_dist <= dist_threshold
    opposed = dots <= -0.7
    return float(np.mean(close & opposed))


def remove_parallel_internal_shells(
    mesh_data: MeshData,
    *,
    sample_count_outer: int = 3000,
    sample_count_inner: int = 1200,
    dist_ratio: float = 0.01,
    score_threshold: float = 0.40,
    min_faces_to_check: int = 1000,
) -> MeshData:
    """
    Remove disconnected internal shells that are close and parallel to the outer shell.

    This is designed for the Trellis "double-wall" artifact where inner shell can be large,
    making size-only filtering ineffective.
    """
    tm = trimesh.Trimesh(
        vertices=mesh_data.vertices.detach().cpu().numpy(),
        faces=mesh_data.faces.detach().cpu().numpy(),
        process=False,
    )
    components = list(tm.split(only_watertight=False))
    if len(components) <= 1:
        return mesh_data

    components = sorted(components, key=lambda c: len(c.faces), reverse=True)
    outer = components[0]
    keep = [outer]
    removed = 0

    bbox_min, bbox_max = outer.bounds
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if bbox_diag <= 1e-9:
        return mesh_data
    dist_threshold = max(1e-4, bbox_diag * dist_ratio)

    outer_pts, outer_normals = _sample_points_normals(outer, count=sample_count_outer)

    for comp in components[1:]:
        if len(comp.faces) < min_faces_to_check:
            keep.append(comp)
            continue

        comp_min, comp_max = comp.bounds
        # Require component bbox to be fully inside outer bbox (with small tolerance).
        tol = bbox_diag * 0.005
        inside_bbox = np.all(comp_min >= (bbox_min - tol)) and np.all(comp_max <= (bbox_max + tol))
        if not inside_bbox:
            keep.append(comp)
            continue

        inner_pts, inner_normals = _sample_points_normals(comp, count=sample_count_inner)
        score = _parallel_shell_score(
            inner_pts=inner_pts,
            inner_normals=inner_normals,
            outer_pts=outer_pts,
            outer_normals=outer_normals,
            dist_threshold=dist_threshold,
        )

        if score >= score_threshold:
            removed += 1
            logger.warning(
                f"Removed parallel internal shell: faces={len(comp.faces)} score={score:.3f} "
                f"dist_threshold={dist_threshold:.6f}"
            )
        else:
            keep.append(comp)

    if removed == 0:
        return mesh_data

    merged = trimesh.util.concatenate(keep)
    logger.warning(f"Removed {removed} parallel internal shell components (kept {len(keep)}).")
    device = mesh_data.vertices.device
    return MeshData(
        vertices=torch.as_tensor(merged.vertices, dtype=mesh_data.vertices.dtype, device=device),
        faces=torch.as_tensor(merged.faces, dtype=mesh_data.faces.dtype, device=device),
        uvs=None,
        vertex_normals=None,
        bvh=None,
    )

