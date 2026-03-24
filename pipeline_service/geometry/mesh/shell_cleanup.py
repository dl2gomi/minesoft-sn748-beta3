"""
Single-pass Trellis shell cleanup: one mesh split, one ordered walk, one merge.

Parallel "double-wall" shells are internal offset duplicates (separate components).
Internal cleanup drops tiny fragments and components fully inside a larger *watertight*
keeper. Both are "internal" in plain language; the tests differ (geometry vs containment).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import trimesh
from logger_config import logger

from geometry.mesh.internal_shells import _sample_component_points
from geometry.mesh.parallel_shells import (
    _parallel_shell_score,
    _sample_points_normals,
)
from geometry.mesh.schemas import MeshData


def run_trellis_shell_cleanup(
    mesh_data: MeshData,
    *,
    parallel: dict | None = None,
    internal: dict | None = None,
) -> MeshData:
    """
    Apply parallel-wall removal and internal fragment/enclosure removal in one split.

    Order per secondary component (face count descending): parallel test first, then
    internal. This matches the previous pipeline (parallel → merge → internal) when
    removals are disconnected components.
    """
    parallel = dict(parallel or {})
    internal = dict(internal or {})
    parallel_enabled = bool(parallel.get("enabled", True))
    internal_enabled = bool(internal.get("enabled", True))
    if not parallel_enabled and not internal_enabled:
        return mesh_data

    t0 = time.perf_counter()
    device = mesh_data.vertices.device

    tm = trimesh.Trimesh(
        vertices=mesh_data.vertices.detach().cpu().numpy(),
        faces=mesh_data.faces.detach().cpu().numpy(),
        process=False,
    )
    t_split = time.perf_counter()
    components = list(tm.split(only_watertight=False))
    split_s = time.perf_counter() - t_split
    if len(components) <= 1:
        logger.debug(f"shell_cleanup: single component split_t={split_s:.2f}s")
        return mesh_data

    components = sorted(components, key=lambda c: len(c.faces), reverse=True)
    outer = components[0]
    largest_faces = max(1, int(len(outer.faces)))
    rel_face_threshold = max(
        int(internal.get("min_component_faces", 64)),
        int(largest_faces * float(internal.get("min_face_ratio_to_keep", 0.005))),
    )
    max_components_to_check = int(internal.get("max_components_to_check", 64))

    bbox_min, bbox_max = outer.bounds
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if bbox_diag <= 1e-9:
        return mesh_data

    sample_count_outer = int(parallel.get("sample_count_outer", 3000))
    sample_count_inner = int(parallel.get("sample_count_inner", 1200))
    dist_ratio = float(parallel.get("dist_ratio", 0.01))
    score_threshold = float(parallel.get("score_threshold", 0.35))
    min_faces_to_check = int(parallel.get("min_faces_to_check", 1000))
    max_parallel_scored = int(parallel.get("max_secondary_components_to_check", 512))
    knn_outer_neighbors = int(parallel.get("knn_outer_neighbors", 56))
    opposing_dot = float(parallel.get("opposing_dot", -0.22))
    opposing_dot_relaxed = float(parallel.get("opposing_dot_relaxed", -0.12))
    distance_relax = float(parallel.get("distance_relax", 2.25))

    dist_threshold = max(1e-4, bbox_diag * dist_ratio)
    tol = bbox_diag * 0.005

    outer_pts: np.ndarray | None = None
    outer_normals: np.ndarray | None = None
    if parallel_enabled:
        outer_pts, outer_normals = _sample_points_normals(outer, count=sample_count_outer)

    keep: list[trimesh.Trimesh] = [outer]
    removed_parallel = 0
    removed_internal = 0
    parallel_scored = 0
    capped_logged = False

    for global_idx, comp in enumerate(components[1:], start=1):
        dropped = False

        if parallel_enabled and outer_pts is not None and outer_normals is not None:
            if len(comp.faces) >= min_faces_to_check:
                comp_min, comp_max = comp.bounds
                inside_bbox = np.all(comp_min >= (bbox_min - tol)) and np.all(comp_max <= (bbox_max + tol))
                if inside_bbox and parallel_scored < max_parallel_scored:
                    inner_pts, inner_normals = _sample_points_normals(comp, count=sample_count_inner)
                    score = _parallel_shell_score(
                        inner_pts=inner_pts,
                        inner_normals=inner_normals,
                        outer_pts=outer_pts,
                        outer_normals=outer_normals,
                        dist_threshold=dist_threshold,
                        knn_k=knn_outer_neighbors,
                        opposing_dot=opposing_dot,
                        opposing_dot_relaxed=opposing_dot_relaxed,
                        distance_relax=distance_relax,
                    )
                    parallel_scored += 1
                    if score >= score_threshold:
                        removed_parallel += 1
                        dropped = True
                elif inside_bbox and parallel_scored >= max_parallel_scored and not capped_logged:
                    logger.warning(
                        f"shell_cleanup: max parallel scores ({max_parallel_scored}) reached; "
                        "remaining secondaries use internal rules only"
                    )
                    capped_logged = True

        if dropped:
            continue

        if not internal_enabled:
            keep.append(comp)
            continue

        if global_idx >= max_components_to_check:
            if len(comp.faces) < rel_face_threshold:
                removed_internal += 1
            else:
                keep.append(comp)
            continue

        if len(comp.faces) < rel_face_threshold:
            removed_internal += 1
            continue

        enclosed = False
        for out in keep:
            if len(out.faces) <= len(comp.faces):
                continue
            if not out.is_watertight:
                continue
            comp_min, comp_max = comp.bounds
            out_min, out_max = out.bounds
            if np.any(comp_min < out_min) or np.any(comp_max > out_max):
                continue
            pts = _sample_component_points(comp, max_points=24)
            if pts.shape[0] == 0:
                continue
            try:
                inside = out.contains(pts)
            except Exception:
                continue
            if bool(np.all(inside)):
                enclosed = True
                break

        if enclosed:
            removed_internal += 1
        else:
            keep.append(comp)

    if removed_parallel == 0 and removed_internal == 0:
        logger.debug(
            f"shell_cleanup: no removals split_t={split_s:.2f}s total={time.perf_counter() - t0:.2f}s"
        )
        return mesh_data

    merged = trimesh.util.concatenate(keep)
    logger.warning(
        f"shell_cleanup: removed parallel={removed_parallel} internal={removed_internal} "
        f"kept_components={len(keep)} ({time.perf_counter() - t0:.2f}s)"
    )
    return MeshData(
        vertices=torch.as_tensor(merged.vertices, dtype=mesh_data.vertices.dtype, device=device),
        faces=torch.as_tensor(merged.faces, dtype=mesh_data.faces.dtype, device=device),
        uvs=None,
        vertex_normals=None,
        bvh=None,
    )
