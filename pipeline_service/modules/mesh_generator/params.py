from pydantic import Field, AliasChoices
from typing import TypeAlias
from modules.mesh_generator.enums import TrellisMode, TrellisPipeType
from schemas.overridable import OverridableModel


class SamplerParams(OverridableModel):
    steps: int = Field(default=12, validation_alias=AliasChoices("steps", "num_inference_steps"))
    guidance_strength: float = Field(default=3.0, validation_alias=AliasChoices("guidance_strength", "cfg_strength"))


class TrellisParams(OverridableModel):
    """TRELLIS.2 parameters with automatic fallback to settings."""
    sparse_structure: SamplerParams = SamplerParams(steps=12, guidance_strength=7.5)
    shape_slat: SamplerParams = SamplerParams(steps=12, guidance_strength=3.0)
    tex_slat: SamplerParams = SamplerParams(steps=12, guidance_strength=3.0)
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024_CASCADE  # '512', '1024', '1024_cascade', '1536_cascade'
    mode: TrellisMode = TrellisMode.STOCHASTIC  # Currently unused in TRELLIS.2
    max_num_tokens: int = 49152
    num_samples: int = 1
    
TrellisParamsOverrides: TypeAlias = TrellisParams.Overrides