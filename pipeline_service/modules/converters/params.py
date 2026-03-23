from typing import TypeAlias

from schemas.overridable import OverridableModel
from geometry.texturing.enums import AlphaMode

class SmoothingParams(OverridableModel):
    """Mesh smoothing options used before cleanup/remesh."""
    enabled: bool = False
    iterations: int = 5
    smooth_lambda: float = 0.5


class RemeshParams(OverridableModel):
    """Cleanup/remesh controls shared by both paths."""
    enabled: bool = True
    band: float = 1.0
    project: float = 0.0
    decimation_target: int = 1000000
    smoothing: SmoothingParams = SmoothingParams()


class UVUnwrapParams(OverridableModel):
    """UV charting/unwrapping controls."""
    mesh_cluster_refine_iterations: int = 0
    mesh_cluster_global_iterations: int = 1
    mesh_cluster_smooth_strength: float = 1.0
    mesh_cluster_threshold_cone_half_angle: float = 90.0


class SubdivisionParams(OverridableModel):
    """Post-unwrap subdivision controls."""
    iterations: int = 2
    vertex_reproject: float = 0.0


class TextureParams(OverridableModel):
    """Texture baking/postprocess controls."""
    texture_size: int = 1024
    alpha_mode: AlphaMode = AlphaMode.OPAQUE
    alpha_gamma: float = 2.2

class GLBConverterParams(OverridableModel):
    """GLB conversion parameters with stage-specific nested overrides."""
    remesh: RemeshParams = RemeshParams()
    uv_unwrap: UVUnwrapParams = UVUnwrapParams()
    subdivision: SubdivisionParams = SubdivisionParams()
    texture: TextureParams = TextureParams()


GLBConverterParamsOverrides: TypeAlias = GLBConverterParams.Overrides
