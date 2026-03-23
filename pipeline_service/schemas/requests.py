from typing import Optional

from pydantic import BaseModel

from schemas.enums import PromptType
from modules.mesh_generator.schemas import TrellisParamsOverrides
from modules.converters.params import GLBConverterParamsOverrides

class GenerationRequest(BaseModel):
    # Prompt data
    prompt_type: PromptType = PromptType.IMAGE
    prompt_image: str 
    seed: int = -1
    render_grid_view: bool = False

    # Trellis parameters
    trellis_params: Optional[TrellisParamsOverrides] = None
    glbconv_params: Optional[GLBConverterParamsOverrides] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt_type": "text",
                "prompt_image": "file_name.jpg",
                "seed": 42,
                "render_grid_view": False,
                "trellis_params": {
                    "sparse_structure": {
                        "steps": 12,
                        "guidance_strength": 7.5
                    },
                    "shape_slat": {
                        "steps": 12,
                        "guidance_strength": 7.5
                    },
                    "tex_slat": {
                        "steps": 12,
                        "guidance_strength": 7.5
                    },
                    "pipeline_type": "1024_cascade",
                    "max_num_tokens": 49152,
                },
                "glbconv_params": {
                    "remesh": {
                        "enabled": True,
                        "band": 1.0,
                        "project": 0.0,
                        "decimation_target": 1000,
                        "smoothing": {
                            "enabled": False,
                            "iterations": 5,
                            "smooth_lambda": 0.5
                        }
                    },
                    "uv_unwrap": {
                        "mesh_cluster_refine_iterations": 0,
                        "mesh_cluster_global_iterations": 1,
                        "mesh_cluster_smooth_strength": 1.0,
                        "mesh_cluster_threshold_cone_half_angle": 90.0
                    },
                    "subdivision": {
                        "iterations": 1,
                        "vertex_reproject": 0.0
                    },
                    "texture": {
                        "texture_size": 512,
                        "alpha_mode": "OPAQUE",
                        "alpha_gamma": 2.0
                    }
                }
            }
        }
