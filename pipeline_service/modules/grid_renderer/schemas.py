from typing import Iterable, Tuple

from pydantic import BaseModel

from schemas.types import ImageTensor


class GridRendererInput(BaseModel):
    glb_bytes: Iterable[bytes]


class GridRendererOutput(BaseModel):
    grids: Tuple[ImageTensor, ...]
