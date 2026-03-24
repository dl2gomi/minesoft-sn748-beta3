from __future__ import annotations

import numpy as np
import trimesh

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


def remove_internal_enclosed_shells(mesh_data: MeshData, **kwargs) -> MeshData:
    """Internal-only pass (fragments + watertight enclosure). Prefer `run_trellis_shell_cleanup` in Trellis."""
    from geometry.mesh.shell_cleanup import run_trellis_shell_cleanup

    return run_trellis_shell_cleanup(mesh_data, parallel={"enabled": False}, internal=kwargs)

