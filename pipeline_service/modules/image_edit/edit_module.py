import time
from typing import Iterable

import torch
from PIL import Image

from .image_edit_pipeline import ImageEditPipeline
from logger_config import logger
from modules.image_edit.params import ImageGenerationParams
from modules.image_edit.prompting import TextPrompting
from modules.image_edit.schemas import (
    ImageEditInput,
    ImageEditOutput,
)
from schemas.image_convertions import any_images_to_pil_tuple, pil_images_to_images_tensor

INPUT_IMAGE_SIZE = 1024 * 1024


class ImageEditModule:
    """Runs image edits with a provided loaded image-edit pipeline."""

    def __init__(self, default_edit_params: ImageGenerationParams):
        self.default_edit_params = default_edit_params

    def _prepare_input_image(self, image: Image.Image, pixels: int = INPUT_IMAGE_SIZE) -> Image.Image:
        total = int(pixels)
        scale_by = (total / (image.width * image.height)) ** 0.5
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def edit_image(self, request: ImageEditInput) -> ImageEditOutput:
        """ 
        Edit the image using Qwen Edit.

        Args:
            request: Image edit request.

        Returns:
            Image edit result.
        """
        model: ImageEditPipeline = request.model
        assert model.is_ready(), "Edit pipeline is not loaded."
        
        try:
            start_time = time.time()

            prompting = request.prompting
            if isinstance(prompting, str):
                prompting = TextPrompting(positive=prompting)

            prompt_image = request.prompt_image
            prompt_images = [prompt_image] if not isinstance(prompt_image, Iterable) else prompt_image
            resized_images = [self._prepare_input_image(image) for image in any_images_to_pil_tuple(prompt_images)]

            prompting_args = prompting.model_dump()
            generation_args = self.default_edit_params.overrided(request.params).model_dump()

            if request.seed is not None:
                prompting_args["generator"] = torch.Generator(device=model.device).manual_seed(request.seed)

            result = model.pipeline(
                image=resized_images,
                **generation_args,
                **prompting_args,
            )
            
            generation_time = time.time() - start_time
            
            results = tuple(result.images)
            if not results:
                raise ValueError("Image edit pipeline returned no images")
            edited_images = pil_images_to_images_tensor(results)
            
            logger.success(
                f"Edited image generated in {generation_time:.2f}s, Size: {results[0].size}, Seed: {request.seed}"
            )
            
            return ImageEditOutput(edited_images=edited_images, pil_images=tuple(results))
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e
