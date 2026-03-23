from PIL import Image

import io
import base64
from datetime import datetime
from typing import Iterable, Optional, Tuple
import os
import random
import numpy as np
from schemas.types import ImageTensor, PILImage
import torch

from logger_config import logger
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, resize

from config.settings import settings

def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)

def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")

def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")

def save_files(
    glb_bytes: Optional[bytes],
    image_edited: Image.Image,
    image_without_background: Image.Image,
    grid_views_from_num_samples: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        glb_bytes: GLB file bytes to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
        grid_views_from_num_samples: The grid views from num samples to save.
    """
    # Save the GLB result if available
    if glb_bytes:
        format = 'glb'
        save_file_bytes(glb_bytes, folder=format, prefix="mesh", suffix=f".{format}")

    # Save the images using PIL Image.save()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)
    save_image(grid_views_from_num_samples, "png", "grid_views_from_num_samples", timestamp)

def image_grid(images: Iterable[ImageTensor], cell_size: Optional[Tuple]=(512,512)) -> PILImage:
    images = (img.permute(2, 0, 1) for img in images)
    if cell_size is not None:
        images = (resize(img, cell_size) for img in images)
    tensor_grid = make_grid(list(images))
    return to_pil_image(tensor_grid)

