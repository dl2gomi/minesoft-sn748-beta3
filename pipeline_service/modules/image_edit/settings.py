from pathlib import Path

from config.types import ModelConfig
from modules.image_edit.params import ImageGenerationParams


class QwenConfig(ModelConfig):
    """Qwen model configuration"""
    model_id: str = "Qwen/Qwen-Image-Edit-2511"
    base_model_path: str = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
    lora_path: str = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    fuse_lightning_lora: bool = True
    lora_angles_path: str = "" 
    lora_angles_filename: str = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
    params: ImageGenerationParams = ImageGenerationParams(height=1024, width=1024, num_inference_steps=8, true_cfg_scale=1.0)
    prompt_path_base: Path = Path("prompts") / "qwen_edit_prompt_v1.json"
    dtype: str = "bf16"
