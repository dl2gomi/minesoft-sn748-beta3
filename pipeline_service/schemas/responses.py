from typing import Optional

from pydantic import BaseModel


class GenerationResponse(BaseModel):
    generation_time: float 
    glb_file_base64: Optional[str | bytes] = None
    grid_view_file_base64: Optional[str | bytes] = None
    grid_views_from_num_samples_base64: Optional[str] = None
    image_edited_file_base64: Optional[str] = None
    image_without_background_file_base64: Optional[str] = None
    trellis_oom_retry: Optional[bool] = None
    qwen_oom_retry: Optional[bool] = None
    qwen_edit_skipped: Optional[bool] = None
    trellis_pipeline_used: Optional[str] = None
    uv_unwrap_mode: Optional[str] = None
    uv_unwrap_reason: Optional[str] = None
    cluster_count: Optional[int] = None
    duel_done: Optional[bool] = None
    duel_winner: Optional[int] = None
    duel_explanation: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "generation_time": 7.2,
                "glb_file_base64": "base64_encoded_glb_file",
                "grid_view_file_base64": "base64_encoded_grid_view_file",
                "grid_views_from_num_samples_base64": "base64_encoded_grid_view_file",
                "image_edited_file_base64": "base64_encoded_image_edited_file",
                "image_without_background_file_base64": "base64_encoded_image_without_background_file",
                "trellis_oom_retry": True,
                "qwen_oom_retry": True,
                "qwen_edit_skipped": False,
                "trellis_pipeline_used": "1024_cascade",
                "uv_unwrap_mode": "xatlas",
                "uv_unwrap_reason": None,
                "cluster_count": 312,
                "duel_done": True,
                "duel_winner": 0,
                "duel_explanation": "candidate 0 better matches prompt silhouette",
            }
        }

