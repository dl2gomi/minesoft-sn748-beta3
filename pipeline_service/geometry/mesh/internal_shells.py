from __future__ import annotations

from typing import List

import numpy as np
import torch
import trimesh
from logger_config import logger

from geometry.mesh.schemas import MeshData


def _sample_component_points(component: trimesh.Trimesh, max_points: int = 32) -> np.ndarray:
    """Sample representative points from a component for containment checks."""
    verts = component.vertices
    if len(verts) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if len(verts) <= max_points - 1:
        pts = verts
    else:
        idx = np.linspace(0, len(verts) - 1, num=max_points - 1, dtype=np.int64)
        pts = verts[idx]
    return np.vstack([component.centroid.reshape(1, 3), pts])


def remove_internal_enclosed_shells(
    mesh_data: MeshData,
    *,
    min_component_faces: int = 64,
    max_components_to_check: int = 64,
    min_face_ratio_to_keep: float = 0.005,
) -> MeshData:
    """
    Remove disconnected components that are fully enclosed by larger watertight shells.
    This preserves the outer shell geometry and drops interior shells/artifacts.
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
    largest_faces = max(1, int(len(components[0].faces)))
    rel_face_threshold = max(min_component_faces, int(largest_faces * min_face_ratio_to_keep))

    # Process only the head for expensive containment checks; tail is still handled
    # by deterministic relative-size culling below.
    head = components[:max_components_to_check]
    tail = components[max_components_to_check:]

    keep: List[trimesh.Trimesh] = []
    removed = 0
    for i, comp in enumerate(head):
        if i == 0:
            keep.append(comp)
            continue
        if len(comp.faces) < rel_face_threshold:
            removed += 1
            continue

        enclosed = False
        for outer in keep:
            if len(outer.faces) <= len(comp.faces):
                continue
            if not outer.is_watertight:
                continue

            # Quick reject by AABB: inner bbox must be inside outer bbox.
            comp_min, comp_max = comp.bounds
            out_min, out_max = outer.bounds
            if np.any(comp_min < out_min) or np.any(comp_max > out_max):
                continue

            # Robust check: sampled points must be inside outer shell.
            pts = _sample_component_points(comp, max_points=24)
            if pts.shape[0] == 0:
                continue
            try:
                inside = outer.contains(pts)
            except Exception:
                # contains() can fail if optional acceleration deps are missing
                continue
            if bool(np.all(inside)):
                enclosed = True
                break

        if enclosed:
            removed += 1
        else:
            keep.append(comp)

    # Deterministic culling for remaining tiny fragments not covered by head checks.
    for comp in tail:
        if len(comp.faces) < rel_face_threshold:
            removed += 1
        else:
            keep.append(comp)

    if removed == 0:
        return mesh_data

    merged = trimesh.util.concatenate(keep)
    logger.warning(
        f"Removed {removed} internal/fragment components (kept {len(keep)}). "
        f"threshold_faces={rel_face_threshold} largest_faces={largest_faces}"
    )
    device = mesh_data.vertices.device
    return MeshData(
        vertices=torch.as_tensor(merged.vertices, dtype=mesh_data.vertices.dtype, device=device),
        faces=torch.as_tensor(merged.faces, dtype=mesh_data.faces.dtype, device=device),
        uvs=None,
        vertex_normals=None,
        bvh=None,
    )

