from pydantic import BaseModel
from typing import Optional

from schemas.types import TriMesh
from .params import GLBConverterParamsOverrides
from geometry.mesh.schemas import MeshDataWithAttributeGrid


class GLBConverterInput(BaseModel):
    mesh: MeshDataWithAttributeGrid
    params: Optional[GLBConverterParamsOverrides] = None


class GLBConverterOutput(BaseModel):
    glb_mesh: TriMesh