from pydantic import Field, AliasChoices
from typing import TypeAlias
from modules.mesh_generator.enums import TrellisMode, TrellisPipeType
from schemas.overridable import OverridableModel


class SamplerParams(OverridableModel):
    steps: int = Field(default=12, validation_alias=AliasChoices("steps", "num_inference_steps"))
    guidance_strength: float = Field(default=3.0, validation_alias=AliasChoices("guidance_strength", "cfg_strength"))


class ShellCleanupParallelParams(OverridableModel):
    enabled: bool = True
    sample_count_outer: int = 3000
    sample_count_inner: int = 1200
    dist_ratio: float = 0.01
    score_threshold: float = 0.35
    min_faces_to_check: int = 1000
    max_secondary_components_to_check: int = 512
    knn_outer_neighbors: int = 56
    opposing_dot: float = -0.22
    opposing_dot_relaxed: float = -0.12
    distance_relax: float = 2.25


class ShellCleanupInternalParams(OverridableModel):
    enabled: bool = True
    min_component_faces: int = 64
    max_components_to_check: int = 64
    min_face_ratio_to_keep: float = 0.005


class ShellCleanupParams(OverridableModel):
    parallel: ShellCleanupParallelParams = ShellCleanupParallelParams()
    internal: ShellCleanupInternalParams = ShellCleanupInternalParams()


class TrellisParams(OverridableModel):
    """TRELLIS.2 parameters with automatic fallback to settings."""
    sparse_structure: SamplerParams = SamplerParams(steps=12, guidance_strength=7.5)
    shape_slat: SamplerParams = SamplerParams(steps=12, guidance_strength=3.0)
    tex_slat: SamplerParams = SamplerParams(steps=12, guidance_strength=3.0)
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024_CASCADE  # '512', '1024', '1024_cascade', '1536_cascade'
    mode: TrellisMode = TrellisMode.STOCHASTIC  # Currently unused in TRELLIS.2
    max_num_tokens: int = 49152
    num_samples: int = 1
    shell_cleanup: ShellCleanupParams = ShellCleanupParams()
    
TrellisParamsOverrides: TypeAlias = TrellisParams.Overrides