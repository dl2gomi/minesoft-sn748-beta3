from __future__ import annotations

import json
from pathlib import Path
import time

import torch
import cumesh
try:
    import warp as wp
except Exception:
    wp = None
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from geometry.mesh.schemas import MeshData
from logger_config import logger

if wp is not None:
    @wp.kernel
    def _cull_internal_faces_kernel(
        points: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        faces_flat: wp.array(dtype=wp.int32),
        grid: wp.uint64,
        mesh_id: wp.uint64,
        voxel_size: float,
        aabb_min: wp.vec3,
        aabb_max: wp.vec3,
        aabb_extent: wp.vec3,
        hit_ratio_threshold: float,
        neighbor_radius_scale: float,
        opposite_dot_max: float,
        hop_mode: int,
        start_index: int,
        out_mask: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        i = start_index + tid
        p = points[i]
        n = normals[i]

        eps = float(wp.max(1.0e-6, voxel_size * 0.01))
        radius = float(neighbor_radius_scale * voxel_size)
        g_count = int(0)
        g_inf_hit = int(0)

        i0 = faces_flat[i * 3 + 0]
        i1 = faces_flat[i * 3 + 1]
        i2 = faces_flat[i * 3 + 2]

        q = wp.hash_grid_query(grid, p, radius)
        j = int(0)
        while wp.hash_grid_query_next(q, j):
            if j == i:
                continue
            if wp.length(points[j] - p) > radius:
                continue
            dot = wp.dot(n, normals[j])
            if dot > opposite_dot_max:
                continue

            if hop_mode >= 1:
                j0 = faces_flat[j * 3 + 0]
                j1 = faces_flat[j * 3 + 1]
                j2 = faces_flat[j * 3 + 2]
                if i0 == j0 or i0 == j1 or i0 == j2 or i1 == j0 or i1 == j1 or i1 == j2 or i2 == j0 or i2 == j1 or i2 == j2:
                    continue

            # Infinity-hit test on candidate face normal (+normal only).
            nj = normals[j]
            cj = points[j]
            t_exit = float(1.0e9)
            for axis in range(3):
                d = nj[axis]
                if wp.abs(d) > 1.0e-7:
                    boundary = wp.where(d > 0.0, aabb_max[axis], aabb_min[axis])
                    t_to_boundary = (boundary - cj[axis]) / d
                    if t_to_boundary > 0.0:
                        t_exit = wp.min(t_exit, t_to_boundary)

            # L2 directional extent (sqrt of squared projected axis terms).
            proj_extent = 0.4 * wp.sqrt(
                (nj[0] * aabb_extent[0]) * (nj[0] * aabb_extent[0]) +
                (nj[1] * aabb_extent[1]) * (nj[1] * aabb_extent[1]) +
                (nj[2] * aabb_extent[2]) * (nj[2] * aabb_extent[2])
            )
            target_dist = wp.min(proj_extent, t_exit)
            if target_dist <= 0.0:
                continue

            g_count = g_count + 1
            ray_o = cj + nj * eps
            if not wp.mesh_query_ray_anyhit(mesh_id, ray_o, nj, target_dist, -1):
                g_inf_hit = g_inf_hit + 1

        if g_count > 0:
            ratio = float(g_inf_hit) / float(g_count)
            if ratio >= hit_ratio_threshold:
                out_mask[tid] = 1

def _compute_face_centroids_normals(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tri = vertices[faces]
    v01 = tri[:, 1] - tri[:, 0]
    v02 = tri[:, 2] - tri[:, 0]
    normals = torch.cross(v01, v02, dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-12)
    centroids = tri.mean(dim=1)
    return centroids, normals


def _compact_mesh(vertices: torch.Tensor, faces: torch.Tensor, keep_face_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kept_faces = faces[keep_face_mask]
    if kept_faces.numel() == 0:
        return vertices, faces
    unique_verts, inverse = torch.unique(kept_faces.reshape(-1), sorted=True, return_inverse=True)
    return vertices[unique_verts], inverse.reshape(-1, 3)


def _dump_onehop_debug_samples(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    centroids: torch.Tensor,
    normals: torch.Tensor,
    internal_mask: torch.Tensor,
    voxel_size: float,
    distance_max_scale: float,
    opposite_dot_max: float,
    hit_ratio_threshold: float,
    hop_mode: int,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    aabb_size: torch.Tensor,
) -> None:
    # Temporary hardcoded debug dump. Remove after investigation.
    out_dir = Path("/root/sn748/minesoft-sn748-beta3/debug/onehop")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "onehop_face_debug.json"

    device = vertices.device
    n_faces = int(faces.shape[0])
    radius = float(distance_max_scale) * float(voxel_size)
    eps = max(1.0e-6, float(voxel_size) * 0.01)
    face_ids = torch.arange(n_faces, device=device)

    internal_ids = torch.nonzero(internal_mask, as_tuple=False).squeeze(1)
    external_ids = torch.nonzero(~internal_mask, as_tuple=False).squeeze(1)
    sample_internal = internal_ids[:5]
    sample_external = external_ids[:5]
    sample_ids = torch.cat([sample_internal, sample_external], dim=0)
    if sample_ids.numel() == 0:
        return

    bvh = cumesh.cuBVH(vertices, faces)
    sample_records: list[dict] = []
    aabb_extent = aabb_size.float()

    for i_t in sample_ids:
        i = int(i_t.item())
        p = centroids[i]
        n = normals[i]

        dist = torch.linalg.norm(centroids - p[None, :], dim=1)
        dots = torch.sum(normals * n[None, :], dim=1)
        neighbor_mask = (face_ids != i) & (dist <= radius) & (dots <= float(opposite_dot_max))

        if hop_mode >= 1:
            fi = faces[i]
            share = (
                (faces[:, 0] == fi[0]) | (faces[:, 1] == fi[0]) | (faces[:, 2] == fi[0]) |
                (faces[:, 0] == fi[1]) | (faces[:, 1] == fi[1]) | (faces[:, 2] == fi[1]) |
                (faces[:, 0] == fi[2]) | (faces[:, 1] == fi[2]) | (faces[:, 2] == fi[2])
            )
            neighbor_mask = neighbor_mask & (~share)

        nbr_idx = torch.nonzero(neighbor_mask, as_tuple=False).squeeze(1)
        nbr_count_pre_ray = int(nbr_idx.numel())
        if nbr_count_pre_ray == 0:
            sample_records.append(
                {
                    "face_id": i,
                    "kernel_marked_internal": bool(internal_mask[i].item()),
                    "neighbor_radius": radius,
                    "neighbors_total_after_filters": 0,
                    "neighbors_ray_checked": 0,
                    "neighbors_infinity_hits": 0,
                    "infinity_ratio": 0.0,
                    "recomputed_internal": False,
                    "neighbors": [],
                }
            )
            continue

        nj = normals[nbr_idx]
        cj = centroids[nbr_idx]

        t_exit = torch.full((nbr_idx.shape[0],), 1.0e9, dtype=torch.float32, device=device)
        for axis in range(3):
            d = nj[:, axis]
            valid = d.abs() > 1.0e-7
            boundary = torch.where(d > 0.0, aabb_max[axis], aabb_min[axis])
            t_to_boundary = (boundary - cj[:, axis]) / d
            valid = valid & (t_to_boundary > 0.0)
            t_exit = torch.where(valid, torch.minimum(t_exit, t_to_boundary), t_exit)

        proj_extent = 0.4 * torch.sqrt(
            (nj[:, 0] * aabb_extent[0]) ** 2 +
            (nj[:, 1] * aabb_extent[1]) ** 2 +
            (nj[:, 2] * aabb_extent[2]) ** 2
        )
        target_dist = torch.minimum(proj_extent, t_exit)
        valid_ray = target_dist > 0.0
        ray_idx = nbr_idx[valid_ray]

        if ray_idx.numel() > 0:
            ray_o = centroids[ray_idx] + normals[ray_idx] * eps
            ray_d = normals[ray_idx]
            _, hit_face_id, hit_depth = bvh.ray_trace(ray_o, ray_d)
            ray_hit = (hit_face_id >= 0) & (hit_depth > 0.0) & (hit_depth <= target_dist[valid_ray])
            inf_hit = ~ray_hit
            g_count = int(ray_idx.numel())
            g_inf_hit = int(inf_hit.sum().item())
        else:
            ray_hit = torch.empty((0,), dtype=torch.bool, device=device)
            inf_hit = torch.empty((0,), dtype=torch.bool, device=device)
            g_count = 0
            g_inf_hit = 0

        ratio = (float(g_inf_hit) / float(g_count)) if g_count > 0 else 0.0
        recomputed_internal = bool(g_count > 0 and ratio >= float(hit_ratio_threshold))

        ray_lookup = {int(ray_idx[k].item()): k for k in range(ray_idx.shape[0])}
        neighbors = []
        for k in range(nbr_idx.shape[0]):
            j = int(nbr_idx[k].item())
            ray_k = ray_lookup.get(j, None)
            neighbors.append(
                {
                    "face_id": j,
                    "distance": float(dist[j].item()),
                    "normal_dot": float(dots[j].item()),
                    "target_dist": float(target_dist[k].item()),
                    "ray_checked": bool(ray_k is not None),
                    "ray_hit_any_face": bool(ray_hit[ray_k].item()) if ray_k is not None else False,
                    "ray_hits_infinity": bool(inf_hit[ray_k].item()) if ray_k is not None else False,
                }
            )

        sample_records.append(
            {
                "face_id": i,
                "kernel_marked_internal": bool(internal_mask[i].item()),
                "neighbor_radius": radius,
                "neighbors_total_after_filters": nbr_count_pre_ray,
                "neighbors_ray_checked": g_count,
                "neighbors_infinity_hits": g_inf_hit,
                "infinity_ratio": ratio,
                "recomputed_internal": recomputed_internal,
                "neighbors": neighbors,
            }
        )

    payload = {
        "meta": {
            "voxel_size": float(voxel_size),
            "distance_max_scale": float(distance_max_scale),
            "opposite_dot_max": float(opposite_dot_max),
            "hit_ratio_threshold": float(hit_ratio_threshold),
            "hop_mode": int(hop_mode),
            "sample_face_count": int(sample_ids.numel()),
            "total_faces": n_faces,
        },
        "samples": sample_records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.warning(f"onehop debug dump saved: {out_path}")


def remove_internal_faces_onehop_cuda(
    mesh_data: MeshData,
    *,
    enabled: bool = True,
    exact_voxel_size: float | None = None,
    distance_max_scale: float = 1.5,
    opposite_dot_max: float = -0.5,
    hit_ratio_threshold: float = 0.7,
    bin_scale: float = 1.5,
    hop_mode: int = 1,
    launch_chunk_size: int = 500000,
    progress_log_every_bins: int = 1,
    use_tqdm: bool = False,
    tqdm_mininterval: float = 1.0,
) -> MeshData:
    if not enabled:
        return mesh_data
    if mesh_data.faces.shape[0] < 16:
        return mesh_data

    t0 = time.perf_counter()
    vertices = mesh_data.vertices
    faces = mesh_data.faces.long()
    device = vertices.device
    if device.type != "cuda":
        logger.warning("internal_faces_onehop requires CUDA tensors; skipping.")
        return mesh_data
    if wp is None:
        logger.warning("internal_faces_onehop requires warp; skipping because warp is unavailable.")
        return mesh_data

    kernel = _cull_internal_faces_kernel

    centroids, normals = _compute_face_centroids_normals(vertices, faces)
    aabb_min = vertices.min(dim=0).values
    aabb_max = vertices.max(dim=0).values
    aabb_size = aabb_max - aabb_min
    voxel_size = float(exact_voxel_size) if exact_voxel_size is not None else 0.0
    if voxel_size <= 0.0:
        voxel_size = float(aabb_size.max().item()) / 1024.0
        logger.warning(
            "onehop_shells exact_voxel_size is missing; falling back to AABB-derived voxel size."
        )
    bin_size = max(1e-7, voxel_size * float(bin_scale))

    wp_device = "cuda"
    n_faces = int(faces.shape[0])
    points_w = wp.from_torch(centroids.float().contiguous(), dtype=wp.vec3)
    normals_w = wp.from_torch(normals.float().contiguous(), dtype=wp.vec3)
    faces_flat_w = wp.from_torch(faces.reshape(-1).int().contiguous(), dtype=wp.int32)
    verts_w = wp.from_torch(vertices.float().contiguous(), dtype=wp.vec3)
    mesh = wp.Mesh(points=verts_w, indices=faces_flat_w)

    dim_x = max(8, min(2048, int(torch.ceil(aabb_size[0] / bin_size).item()) + 2))
    dim_y = max(8, min(2048, int(torch.ceil(aabb_size[1] / bin_size).item()) + 2))
    dim_z = max(8, min(2048, int(torch.ceil(aabb_size[2] / bin_size).item()) + 2))
    grid = wp.HashGrid(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z, device=wp_device)
    grid.build(points_w, float(bin_size))

    internal = torch.zeros((n_faces,), dtype=torch.bool, device=device)
    n_chunks = max(1, (n_faces + int(launch_chunk_size) - 1) // int(launch_chunk_size))
    use_bar = bool(use_tqdm and tqdm is not None)
    chunk_iter = range(n_chunks)
    if use_bar:
        chunk_iter = tqdm(
            chunk_iter,
            total=n_chunks,
            desc="onehop_shells",
            unit="chunk",
            mininterval=float(tqdm_mininterval),
            leave=False,
        )
    elif use_tqdm and tqdm is None:
        logger.warning("onehop_shells use_tqdm=true but tqdm is not installed; using log progress only.")

    t_loop = time.perf_counter()
    for ci in chunk_iter:
        start = ci * int(launch_chunk_size)
        end = min(n_faces, start + int(launch_chunk_size))
        m = end - start
        if m <= 0:
            continue

        out_mask_t = torch.zeros((m,), dtype=torch.int32, device=device)
        out_mask_w = wp.from_torch(out_mask_t, dtype=wp.int32)

        wp.launch(
            kernel=kernel,
            dim=m,
            inputs=[
                points_w,
                normals_w,
                faces_flat_w,
                grid.id,
                mesh.id,
                float(voxel_size),
                wp.vec3(float(aabb_min[0].item()), float(aabb_min[1].item()), float(aabb_min[2].item())),
                wp.vec3(float(aabb_max[0].item()), float(aabb_max[1].item()), float(aabb_max[2].item())),
                wp.vec3(float(aabb_size[0].item()), float(aabb_size[1].item()), float(aabb_size[2].item())),
                float(hit_ratio_threshold),
                float(distance_max_scale),
                float(opposite_dot_max),
                int(hop_mode),
                int(start),
                out_mask_w,
            ],
            device=wp_device,
        )
        internal[start:end] = out_mask_t > 0

        if (not use_bar) and progress_log_every_bins > 0 and (ci % progress_log_every_bins == 0 or ci == n_chunks - 1):
            elapsed = time.perf_counter() - t_loop
            done = ci + 1
            eta = (elapsed / done) * (n_chunks - done) if done > 0 else 0.0
            logger.info(f"onehop_shells progress | chunk {done}/{n_chunks} elapsed={elapsed:.2f}s eta={eta:.2f}s")

    keep = ~internal
    _dump_onehop_debug_samples(
        vertices=vertices,
        faces=faces,
        centroids=centroids,
        normals=normals,
        internal_mask=internal,
        voxel_size=voxel_size,
        distance_max_scale=distance_max_scale,
        opposite_dot_max=opposite_dot_max,
        hit_ratio_threshold=hit_ratio_threshold,
        hop_mode=hop_mode,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        aabb_size=aabb_size,
    )

    if bool(internal.any()):
        new_vertices, new_faces = _compact_mesh(vertices, faces, keep)
        logger.warning(
            f"onehop_shells | faces {int(faces.shape[0])}->{int(new_faces.shape[0])} "
            f"verts {int(vertices.shape[0])}->{int(new_vertices.shape[0])} "
            f"removed_faces={int(internal.sum().item())} time={time.perf_counter() - t0:.2f}s"
        )
        return MeshData(vertices=new_vertices, faces=new_faces, uvs=None, vertex_normals=None, bvh=None)

    logger.info(
        f"onehop_shells | faces {int(faces.shape[0])}->{int(faces.shape[0])} "
        f"verts {int(vertices.shape[0])}->{int(vertices.shape[0])} removed_faces=0 "
        f"time={time.perf_counter() - t0:.2f}s"
    )
    return mesh_data

