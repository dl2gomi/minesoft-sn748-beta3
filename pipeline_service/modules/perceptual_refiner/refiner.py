from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import cumesh

from geometry.mesh.schemas import MeshDataWithAttributeGrid, AttributeGrid
from logger_config import logger
from .settings import PerceptualRefinerConfig


@dataclass
class PerceptualRefiner:
    settings: PerceptualRefinerConfig

    @staticmethod
    def _mend_mesh_for_refinement(mesh: MeshDataWithAttributeGrid) -> MeshDataWithAttributeGrid:
        """
        Clean isolated/degenerate components before differentiable optimization.
        Keeps attribute grid untouched.
        """
        try:
            cm = cumesh.CuMesh()
            cm.init(mesh.vertices, mesh.faces)
            cm.remove_duplicate_faces()
            cm.repair_non_manifold_edges()
            cm.remove_small_connected_components(1e-5)
            cm.fill_holes(max_hole_perimeter=3e-2)
            cm.unify_face_orientations()
            verts, faces = cm.read()
            cm.compute_vertex_normals()
            normals = cm.read_vertex_normals()
            return MeshDataWithAttributeGrid(
                vertices=verts,
                faces=faces,
                vertex_normals=normals,
                uvs=None,
                bvh=None,
                attrs=mesh.attrs,
            )
        except Exception as exc:
            logger.warning(f"Refinement pre-mend skipped due to error: {exc}")
            return mesh

    def _refine_alpha_attribute_volume(
        self,
        mesh: MeshDataWithAttributeGrid,
        target_alpha: torch.Tensor,
    ) -> MeshDataWithAttributeGrid:
        """
        Refine Trellis alpha attribute channel (index 5) with a lightweight
        distribution-matching optimization against soft target alpha statistics.
        """
        try:
            attrs = mesh.attrs
            values = attrs.values
            if values.ndim != 2 or values.shape[1] <= 5:
                return mesh

            alpha_src = values[:, 5].float().clamp(0.0, 1.0)
            device = alpha_src.device
            target = target_alpha.float().to(device).clamp(0.0, 1.0)
            target_mean = target.mean().detach()
            target_var = target.var(unbiased=False).detach()

            gain = torch.tensor(1.0, device=device, requires_grad=True)
            bias = torch.tensor(0.0, device=device, requires_grad=True)
            opt = torch.optim.Adam([gain, bias], lr=self.settings.alpha_refine_lr)

            for _ in range(self.settings.alpha_refine_steps):
                opt.zero_grad(set_to_none=True)
                mapped = torch.sigmoid(gain * (alpha_src - 0.5) + bias)
                loss = (
                    (mapped.mean() - target_mean).pow(2)
                    + 0.25 * (mapped.var(unbiased=False) - target_var).pow(2)
                    + 1e-3 * (gain - 1.0).pow(2)
                    + 1e-3 * bias.pow(2)
                )
                loss.backward()
                opt.step()

            alpha_new = torch.sigmoid(gain.detach() * (alpha_src - 0.5) + bias.detach()).clamp(0.0, 1.0)
            values_new = values.clone()
            values_new[:, 5] = alpha_new.to(values_new.dtype)
            attrs_new = AttributeGrid(
                values=values_new,
                coords=attrs.coords,
                aabb=attrs.aabb,
                voxel_size=attrs.voxel_size,
            )
            logger.info(
                "Perceptual alpha-attr refine | "
                f"mean {float(alpha_src.mean()):.4f}->{float(alpha_new.mean()):.4f}"
            )
            return MeshDataWithAttributeGrid(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                uvs=mesh.uvs,
                bvh=mesh.bvh,
                attrs=attrs_new,
            )
        except Exception as exc:
            logger.warning(f"Alpha attribute refinement skipped due to error: {exc}")
            return mesh

    def _is_available(self) -> bool:
        try:
            import pytorch3d  # noqa: F401
            return True
        except Exception:
            return False

    def refine(
        self,
        meshes: Iterable[MeshDataWithAttributeGrid],
        alpha_mask: torch.Tensor | None,
    ) -> tuple[MeshDataWithAttributeGrid, ...]:
        if not self.settings.enabled or alpha_mask is None:
            return tuple(meshes)
        if not self._is_available():
            logger.warning("Perceptual refiner enabled but pytorch3d not installed; skipping.")
            return tuple(meshes)

        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            MeshRasterizer,
            MeshRenderer,
            RasterizationSettings,
            SoftSilhouetteShader,
            BlendParams,
        )
        from pytorch3d.structures import Meshes
        from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss

        meshes_tuple = tuple(meshes)
        out: list[MeshDataWithAttributeGrid] = []
        target = alpha_mask
        if target.ndim == 3:
            target = target[..., 0]
        target = target.float().clamp(0.0, 1.0)
        total_start = time.perf_counter()
        logger.info(
            f"Perceptual refinement started | candidates={len(meshes_tuple)} | "
            f"iterations={self.settings.iterations} | image_size={self.settings.image_size}"
        )

        for candidate_idx, mesh in enumerate(meshes_tuple):
            candidate_start = time.perf_counter()
            try:
                mesh = self._mend_mesh_for_refinement(mesh)
                device = mesh.vertices.device
                target_resized = F.interpolate(
                    target.unsqueeze(0).unsqueeze(0),
                    size=(self.settings.image_size, self.settings.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0).to(device)

                verts_base = mesh.vertices.detach()
                faces = mesh.faces.long().detach()
                delta = torch.zeros_like(verts_base, requires_grad=True)

                cameras = FoVPerspectiveCameras(device=device)
                raster_settings = RasterizationSettings(
                    image_size=self.settings.image_size,
                    blur_radius=self.settings.raster_blur_radius,
                    faces_per_pixel=self.settings.raster_faces_per_pixel,
                )
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                    shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
                )
                optimizer = torch.optim.Adam([delta], lr=self.settings.lr)
                log_every = max(1, self.settings.iterations // 5)
                pbar = tqdm(
                    range(self.settings.iterations),
                    desc=f"PerceptualRefiner c{candidate_idx}",
                    leave=False,
                    dynamic_ncols=True,
                )

                for step in pbar:
                    optimizer.zero_grad(set_to_none=True)
                    verts = verts_base + torch.tanh(delta) * self.settings.max_vertex_shift
                    mesh3d = Meshes(verts=[verts], faces=[faces])
                    rgba = renderer(mesh3d, cameras=cameras)
                    pred_alpha = rgba[..., 3].squeeze(0)

                    loss_sil = F.mse_loss(pred_alpha, target_resized)
                    loss_lap = mesh_laplacian_smoothing(mesh3d, method="cot")
                    loss_nrm = mesh_normal_consistency(mesh3d)
                    loss_edge = mesh_edge_loss(mesh3d)
                    loss = (
                        self.settings.w_silhouette * loss_sil
                        + self.settings.w_laplacian * loss_lap
                        + self.settings.w_normal * loss_nrm
                        + self.settings.w_edge * loss_edge
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                    optimizer.step()
                    pbar.set_postfix(
                        loss=f"{float(loss.detach()):.5f}",
                        sil=f"{float(loss_sil.detach()):.5f}",
                    )
                    if (step + 1) % log_every == 0 or (step + 1) == self.settings.iterations:
                        logger.info(
                            f"Perceptual refinement c{candidate_idx} | "
                            f"step {step + 1}/{self.settings.iterations} | "
                            f"loss={float(loss.detach()):.6f} "
                            f"(sil={float(loss_sil.detach()):.6f}, "
                            f"lap={float(loss_lap.detach()):.6f}, "
                            f"nrm={float(loss_nrm.detach()):.6f}, "
                            f"edge={float(loss_edge.detach()):.6f})"
                        )
                pbar.close()

                refined_vertices = (verts_base + torch.tanh(delta) * self.settings.max_vertex_shift).detach()
                refined_mesh = MeshDataWithAttributeGrid(
                    vertices=refined_vertices,
                    faces=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    uvs=mesh.uvs,
                    bvh=mesh.bvh,
                    attrs=mesh.attrs,
                )
                if self.settings.refine_alpha_attr:
                    refined_mesh = self._refine_alpha_attribute_volume(refined_mesh, target_resized)
                out.append(refined_mesh)
                logger.info(
                    f"Perceptual refinement done for c{candidate_idx} | "
                    f"time={time.perf_counter() - candidate_start:.2f}s"
                )
            except Exception as exc:
                logger.warning(f"Perceptual refinement failed on candidate; using original mesh: {exc}")
                out.append(mesh)

        logger.info(f"Perceptual refinement total time: {time.perf_counter() - total_start:.2f}s")
        return tuple(out)
