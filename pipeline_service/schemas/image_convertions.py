import io
import base64
from typing import Iterable, Tuple
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from schemas.types import ImageTensor, ImagesTensor, PILImage


def pil_to_image_tensor(image: PILImage) -> ImageTensor:
    return to_tensor(image).permute(1, 2, 0).contiguous()


def image_tensor_to_pil(image: ImageTensor) -> PILImage:
    return to_pil_image(image.detach().cpu().permute(2, 0, 1).contiguous())

def images_tensor_to_tuple(images: ImagesTensor) -> Tuple[ImageTensor, ...]:
    return tuple(image.contiguous() for image in images)


def pil_images_to_images_tensor(images: Iterable[PILImage]) -> ImagesTensor:
    return torch.stack([pil_to_image_tensor(image) for image in images], dim=0)


def image_tensors_to_images_tensor(images: Iterable[ImageTensor]) -> ImagesTensor:
    return torch.stack([image.contiguous() for image in images], dim=0)


def any_images_to_pil_tuple(images: Iterable[PILImage | ImageTensor]) -> Tuple[PILImage, ...]:
    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img)
        elif isinstance(img, torch.Tensor):
            pil_images.append(image_tensor_to_pil(img))
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    return tuple(pil_images)