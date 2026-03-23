from typing import Iterable, Optional, Tuple

from pydantic import BaseModel

from modules.image_edit.image_edit_pipeline import ImageEditPipeline
from modules.image_edit.params import ImageGenerationParamsOverrides
from modules.image_edit.prompting import TextPrompting
from schemas.internal import Internal
from schemas.types import ImageTensor, PILImage, ImagesTensor


class ImageEditInput(BaseModel):
    model: Internal[ImageEditPipeline]
    prompt_image: PILImage | Iterable[PILImage] | ImagesTensor
    seed: int
    prompting: TextPrompting | str
    params: Optional[ImageGenerationParamsOverrides] = None


class ImageEditOutput(BaseModel):
    edited_images: ImagesTensor
    pil_images: Optional[Tuple[PILImage, ...]] = None
