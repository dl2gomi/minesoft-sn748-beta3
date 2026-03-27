import time
from typing import Dict, Optional, Tuple
from logger_config import logger
import numpy as np
import torch
import kaolin
from PIL import Image
import trimesh
import trimesh.visual
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from geometry.mesh.schemas import DEFAULT_AABB, MeshData, MeshDataWithAttributeGrid, AttributeGrid
from geometry.texturing.dithering import bayer_dither_pattern
from geometry.mesh.utils import sort_mesh, map_vertices_positions, count_boundary_loops
from geometry.mesh.subdivisions import subdivide_egdes
from geometry.mesh.smoothing import taubin_smooth
from geometry.texturing.utils import dilate_attributes, map_mesh_rasterization, rasterize_mesh_data, sample_grid_attributes
from geometry.texturing.enums import AlphaMode
from .params import GLBConverterParams
from .schemas import GLBConverterInput, GLBConverterOutput
import cumesh


DITHER_PATTERN_SIZE = 16
DITHER_PATTERN = bayer_dither_pattern(4096, 4096, DITHER_PATTERN_SIZE)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg or "cumesh" in msg


class GLBConverter:
    """Converter for extracting and texturing meshes to GLB format."""
    DILATION_KERNEL_SIZE = 5
    DEFAULT_ATTRIBUTES_LAYOUT = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    
    def __init__(self, params: GLBConverterParams):
        """Initialize converter with settings."""
        self.default_params = params
        self.logger = logger
        self.last_uv_unwrap_info: dict | None = None
    
    def convert(self, request: GLBConverterInput) -> GLBConverterOutput:
        """Convert the given mesh to a textured GLB format."""
        mesh = request.mesh
        logger.debug(f"Original mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        params = self.default_params.overrided(request.params)
        logger.debug(f"Using GLB conversion parameters: {params}")
        self.last_uv_unwrap_info = None

        # 1. Prepare original mesh data with BVH
        original_mesh_data = self._prepare_original_mesh(mesh)
        
        # 2. Remesh if required (cleanup otherwise)
        if params.remesh.enabled:
            mesh_data = self._remesh_mesh(original_mesh_data, params)
        else:
            mesh_data = self._cleanup_mesh(original_mesh_data, params)

        # 3. UV unwrap the mesh
        mesh_data = self._uv_unwrap_mesh(mesh_data, params)

        # 4. subdivide unwrapped mesh
        mesh_data = self._subdivide_mesh(mesh_data, original_mesh_data, params)
        
        # 5. Rasterize attributes onto the mesh UVs
        attributes_layout = getattr(mesh, "layout", self.DEFAULT_ATTRIBUTES_LAYOUT)
        attributes, attributes_layout = self._rasterize_attributes(mesh_data, original_mesh_data, attributes_layout, params)
        
        # 6. Post-process the rasterized attributes into textures
        base_color, orm_texture = self._texture_postprocess(attributes, attributes_layout, params)

        # 7. Create the textured mesh
        textured_mesh = self._create_textured_mesh(mesh_data, base_color, orm_texture, params)

        return GLBConverterOutput(glb_mesh=textured_mesh)


    def _prepare_original_mesh(self, mesh: MeshDataWithAttributeGrid, compute_vertex_normals: bool = False) -> MeshDataWithAttributeGrid:
        """
        Cleanup mesh topology and build BVH while preserving attribute grid.
        This method also fills holes outputing additional faces compared to input one.
        """
        logger.debug(f"Preparing original mesh data")
        start_time = time.time()
        device = mesh.vertices.device

        # Prepare attribute grid
        attrs = mesh.attrs.to(device)

        vertices = mesh.vertices.to(device)
        faces = mesh.faces.to(device)
        
        vertex_normals = None
        try:
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(vertices, faces)
            cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
            logger.debug(f"After filling holes: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            vertices, faces = cumesh_mesh.read()

            if compute_vertex_normals:
                cumesh_mesh.compute_vertex_normals()
                vertex_normals = cumesh_mesh.read_vertex_normals()
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(f"CuMesh OOM in _prepare_original_mesh, skipping fill_holes: {exc}")

        original_mesh_data = MeshDataWithAttributeGrid(vertices=vertices, faces=faces,vertex_normals=vertex_normals, attrs=attrs)
        
        # Build BVH for the current mesh to guide remeshing
        logger.debug(f"Building BVH for current mesh...")
        original_mesh_data.build_bvh()
        logger.debug(f"Done building BVH | Time: {time.time() - start_time:.2f}s")
        
        return original_mesh_data

    def _cleanup_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Cleanup and optimize the mesh using decimation and remeshing."""
        try:
            # Create cumesh from current mesh data
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

            if params.remesh.smoothing.enabled:
                logger.debug(
                    f"Applying Taubin smoothing: iterations={params.remesh.smoothing.iterations}, "
                    f"lambda={params.remesh.smoothing.smooth_lambda}"
                )
                smoothed_vertices = taubin_smooth(
                    original_mesh_data,
                    iterations=params.remesh.smoothing.iterations,
                    lambda_factor=params.remesh.smoothing.smooth_lambda,
                    mu_factor=-(params.remesh.smoothing.smooth_lambda + 0.01),
                )
                cumesh_mesh.init(smoothed_vertices, original_mesh_data.faces)

            # Step 1: Aggressive simplification (3x target)
            cumesh_mesh.simplify(params.remesh.decimation_target * 3, verbose=params.remesh.verbose)
            logger.debug(f"After initial simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            
            # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
            cumesh_mesh.remove_duplicate_faces()
            cumesh_mesh.repair_non_manifold_edges()
            cumesh_mesh.remove_small_connected_components(1e-5)
            cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
            logger.debug(f"After initial cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
                
            # Step 3: Final simplification to target count
            cumesh_mesh.simplify(params.remesh.decimation_target, verbose=params.remesh.verbose)
            logger.debug(f"After final simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            
            # Step 4: Final Cleanup loop
            cumesh_mesh.remove_duplicate_faces()
            cumesh_mesh.repair_non_manifold_edges()
            cumesh_mesh.remove_small_connected_components(1e-5)
            cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
            logger.debug(f"After final cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            hole_count = count_boundary_loops(*cumesh_mesh.read())
            logger.debug(f"Holes after cleanup: {hole_count}")

            # Step 5: Unify face orientations
            cumesh_mesh.unify_face_orientations()

            # Extract cleaned mesh data
            vertices, faces = cumesh_mesh.read()
            return MeshData(
                vertices=vertices,
                faces=faces
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(f"CuMesh OOM in _cleanup_mesh, returning mesh without cleanup: {exc}")
            return MeshData(vertices=original_mesh_data.vertices, faces=original_mesh_data.faces)

    def _remesh_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Remesh the given mesh to improve quality."""

        # Create cumesh from current mesh data
        logger.debug("Starting remeshing")
        start_time = time.time()

        try:
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

            if params.remesh.smoothing.enabled:
                logger.debug(
                    f"Applying Taubin smoothing: iterations={params.remesh.smoothing.iterations}, "
                    f"lambda={params.remesh.smoothing.smooth_lambda}"
                )
                smoothed_vertices = taubin_smooth(
                    original_mesh_data,
                    iterations=params.remesh.smoothing.iterations,
                    lambda_factor=params.remesh.smoothing.smooth_lambda,
                    mu_factor=-(params.remesh.smoothing.smooth_lambda + 0.01),
                )
                cumesh_mesh.init(smoothed_vertices, original_mesh_data.faces)
                logger.debug(f"Done smoothing | Time: {time.time() - start_time:.2f}s")

            voxel_size = original_mesh_data.attrs.voxel_size
            aabb = original_mesh_data.attrs.aabb
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

            resolution = grid_size.max().item()
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            
            # Perform Dual Contouring remeshing (rebuilds topology)
            vertices, faces = cumesh.remeshing.remesh_narrow_band_dc(
                *cumesh_mesh.read(),
                center=center,
                scale=(resolution + 3 * params.remesh.band) / resolution * scale,
                resolution=resolution,
                band=params.remesh.band,
                project_back=params.remesh.project,  # Snaps vertices back to original surface
                verbose=params.remesh.verbose,
                bvh=original_mesh_data.bvh,
                remove_inner_faces=params.remesh.remove_inner_faces,
            )
            cumesh_mesh.init(vertices, faces)
            logger.debug(f"After remeshing: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            
            # Simplify and clean the remeshed result
            cumesh_mesh.simplify(params.remesh.decimation_target, verbose=params.remesh.verbose)
            logger.debug(f"After simplifying: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            # Extract remeshed data
            vertices, faces = cumesh_mesh.read()
            hole_count = count_boundary_loops(vertices, faces)
            logger.debug(f"Holes after remesh: {hole_count}")

            logger.debug(f"Done remeshing | Time: {time.time() - start_time:.2f}s")
            return MeshData(
                vertices=vertices,
                faces=faces
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(f"CuMesh OOM in _remesh_mesh, falling back to cleanup only: {exc}")
            return self._cleanup_mesh(original_mesh_data, params)

    def _uv_unwrap_mesh(self, mesh_data: MeshData, params: GLBConverterParams) -> MeshData:
        """Perform UV unwrapping on the mesh."""
        # Create cumesh from current mesh data
        logger.debug("Starting UV unwrapping")
        start_time = time.time()
        
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)
        
        xatlas_compute_charts_kwargs = {
            "max_chart_area": 1.0,
            "max_boundary_length": 2.0,
            "max_cost": 10.0,
            "normal_seam_weight": 5.0,
            "normal_deviation_weight": 1.0,
            "fix_winding": True
        }
        
        try:
            compute_charts_kwargs = {
                "threshold_cone_half_angle_rad": np.radians(params.uv_unwrap.mesh_cluster_threshold_cone_half_angle),
                "refine_iterations": params.uv_unwrap.mesh_cluster_refine_iterations,
                "global_iterations": params.uv_unwrap.mesh_cluster_global_iterations,
                "smooth_strength": params.uv_unwrap.mesh_cluster_smooth_strength,
            }
            cumesh_mesh.compute_charts(**compute_charts_kwargs)
            num_charts, _, _, _, _, _ = cumesh_mesh.read_atlas_charts()

            xatlas_pack_charts_kwargs = {
                "block_align": params.uv_unwrap.xatlas_pack_block_align,
            }
            out_vertices, out_faces, out_uvs, out_vmaps = cumesh_mesh.uv_unwrap(
                compute_charts_kwargs=compute_charts_kwargs,
                xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
                xatlas_pack_charts_kwargs=xatlas_pack_charts_kwargs,
                return_vmaps=True,
                verbose=True,
            )
            device = mesh_data.vertices.device
            out_vertices = out_vertices.to(device)
            out_faces = out_faces.to(device)
            out_uvs = out_uvs.to(device)
            out_vmaps = out_vmaps.to(device)
            
            cumesh_mesh.compute_vertex_normals()
            out_normals = cumesh_mesh.read_vertex_normals()[out_vmaps]
            self.last_uv_unwrap_info = {"mode": "xatlas", "reason": None, "num_charts": int(num_charts)}
            
            logger.debug(f"Done UV unwrapping | Time: {time.time() - start_time:.2f}s")
            
            return MeshData(
                vertices=out_vertices,
                faces=out_faces,
                vertex_normals=out_normals,
                uvs=out_uvs
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_uv_unwrap_info = {"mode": "trivial", "reason": "oom", "num_charts": None}
            logger.warning(f"CuMesh OOM in _uv_unwrap_mesh, propagating for caller retry: {exc}")
            raise

    def _subdivide_mesh(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Subdivide mesh with uv data and optionally reproject vertices to original mesh surface."""
        subdivided_mesh = subdivide_egdes(mesh_data, iterations=params.subdivision.iterations)
        
        if params.subdivision.vertex_reproject > 0.0:
            subdivided_mesh = map_vertices_positions(
                subdivided_mesh,
                original_mesh_data,
                weight=params.subdivision.vertex_reproject,
                inplace=True,
            )

        subdivided_mesh = sort_mesh(subdivided_mesh, axes=(2,1,0))
        
        return subdivided_mesh

    def _rasterize_attributes(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, layout: Dict[str,slice], params: GLBConverterParams) -> Tuple[torch.Tensor, Dict[str,slice]]:
        """Rasterize the given attributes onto the mesh UVs."""
        logger.debug("Sampling attributes(Texture rasterization)")
        start_time = time.time()

        # Rasterize mesh surface
        rast_data = rasterize_mesh_data(mesh_data, params.texture.texture_size, use_vertex_normals=True)
        logger.debug(
            f"Texture baking: sampling {rast_data.positions.shape[0]} valid pixels out of "
            f"{params.texture.texture_size * params.texture.texture_size}"
        )
        logger.debug(f"Attribute volume has {original_mesh_data.attrs.values.shape[0]} voxels")

        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        rast_data = map_mesh_rasterization(rast_data, original_mesh_data, flip_vertex_normals=True)

        # Trilinear sampling from the attribute volume (Color, Material props)
        attributes = sample_grid_attributes(rast_data, original_mesh_data.attrs)

        # Fill seams by dilating valid pixels into nearby empty UV space
        attrs = dilate_attributes(attributes, self.DILATION_KERNEL_SIZE)

        logger.debug(f"Done attribute sampling | Time: {time.time() - start_time:.2f}s")
        
        return attrs, layout
    
    def _texture_postprocess(self, attributes: torch.Tensor, attr_layout: Dict, params: GLBConverterParams) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
        """Post-process the rasterized attributes into final textures."""
        logger.debug("Finalizing mesh textures")
        start_time = time.time()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = attributes[..., attr_layout['base_color']]
        metallic = attributes[..., attr_layout['metallic']]
        roughness = attributes[..., attr_layout['roughness']]
        alpha = attributes[..., attr_layout['alpha']]
        occlusion_channel  = torch.ones_like(metallic)

        # Adjust alpha with gamma
        alpha = alpha.pow(params.texture.alpha_gamma)
        
        # Handle alpha mode
        alpha_mode = params.texture.alpha_mode
        if alpha_mode == AlphaMode.BLEND:
            alpha_mode = AlphaMode.OPAQUE if np.all(alpha == 255) else alpha_mode

        # Apply alpha dithering if flag is set
        if alpha_mode == AlphaMode.DITHER:
            h, w = alpha.shape[:2]
            dither_pattern = torch.as_tensor(DITHER_PATTERN[:h, :w, None], device=alpha.device)
            alpha = (alpha > dither_pattern).float()
            logger.debug(f"Dithered alpha channel has {np.sum(alpha == 0)} transparent pixels out of {alpha.size} total pixels")
            # alpha_mode = AlphaMode.MASK : After dithering, treat as MASK

        rgba = torch.cat([base_color, alpha], dim=-1).clamp(0,1)
        orm = torch.cat([occlusion_channel, roughness, metallic], dim=-1).clamp(0,1)

        base_color_texture = to_pil_image(rgba.permute(2,0,1).cpu())
        orm_texture = to_pil_image(orm.permute(2,0,1).cpu())
        
        logger.debug(f"Done finalizing mesh textures | Time: {time.time() - start_time:.2f}s")
        return base_color_texture, orm_texture

    def _create_textured_mesh(self, mesh_data: MeshData, base_color: Image.Image, orm_texture: Image.Image, params: GLBConverterParams) -> trimesh.Trimesh:
        """Create a textured trimesh mesh from the mesh data and textures."""
        
        logger.debug("Creating textured mesh")
        start_time = time.time()

        alpha_mode = params.texture.alpha_mode
        alpha_mode = AlphaMode.MASK if alpha_mode is AlphaMode.DITHER else alpha_mode

        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_color,
            baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
            metallicRoughnessTexture=orm_texture,
            roughnessFactor=1.0,
            metallicFactor=1.0,
            alphaMode=alpha_mode.value,
            alphaCutoff=alpha_mode.cutoff,
            doubleSided=bool(not params.remesh.enabled)
        )
        
        # --- Coordinate System Conversion & Final Object ---
        vertices_np = mesh_data.vertices.cpu().numpy()
        faces_np = mesh_data.faces.cpu().numpy()
        uvs_np = mesh_data.uvs.cpu().numpy()
        normals_np = mesh_data.vertex_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1]  # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )
        
        logger.debug(f"Done creating textured mesh | Time: {time.time() - start_time:.2f}s")
                
        return textured_mesh
