from typing import TypeAlias
from schemas.overridable import OverridableModel


class ImageGenerationParams(OverridableModel):
    """Image generation parameters with automatic fallback to settings."""
    height: int
    width: int
    num_inference_steps: int
    true_cfg_scale: float = 1.0

ImageGenerationParamsOverrides: TypeAlias = ImageGenerationParams.Overrides
