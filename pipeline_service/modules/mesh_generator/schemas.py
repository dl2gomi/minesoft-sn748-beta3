from pydantic import BaseModel
from collections.abc import Iterable
from typing import Optional, Tuple
from modules.mesh_generator.mesh_generation_pipeline import MeshGenerationPipeline
from geometry.mesh.schemas import MeshDataWithAttributeGrid
from modules.mesh_generator.params import TrellisParamsOverrides
from schemas.internal import Internal
from schemas.types import ImageTensor, ImagesTensor, Bytes


class TrellisInput(BaseModel):
    """Request for TRELLIS.2 3D generation"""
    model: Internal[MeshGenerationPipeline]
    images: Iterable[ImageTensor] | ImagesTensor
    seed: int
    params: Optional[TrellisParamsOverrides] = None

class TrellisOutput(BaseModel):
    """Output from TRELLIS.2 3D generation"""
    meshes: Tuple[MeshDataWithAttributeGrid, ...]

class TrellisResult(BaseModel):
    """Result from TRELLIS.2 3D generation."""
    file_bytes: Optional[Bytes] = None
