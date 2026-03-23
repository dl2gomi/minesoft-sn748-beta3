from libs.trellis2.representations.mesh.base import MeshWithVoxel
import torch
import cumesh
from typing import Optional
from pydantic import BaseModel
from schemas.types import AnyTensor, IntegerTensor, FloatTensor
from schemas.internal import Internal

DEFAULT_AABB: FloatTensor = torch.as_tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)

class AttributeGrid(BaseModel):
    values: AnyTensor          # (N, K) attribute values for N voxels
    coords: IntegerTensor          # (N, 3) voxel coordinates on the grid
    aabb: FloatTensor          # (2, 3) axis-aligned bounding box (optional)
    voxel_size: FloatTensor    # (3,) size of voxel in each dimension

    @property
    def grid_size(self) -> FloatTensor:
        return ((self.aabb[1] - self.aabb[0]) / self.voxel_size).round().int()
    

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (self.values.shape[1], *self.grid_size.tolist()))

    @classmethod
    def from_mesh_with_voxels(cls, mesh: MeshWithVoxel, aabb: FloatTensor = DEFAULT_AABB) -> "AttributeGrid":
        device = mesh.attrs.device
        return cls(
            values=mesh.attrs,
            coords=mesh.coords,
            aabb = torch.as_tensor(aabb, dtype=torch.float32, device=device),
            voxel_size = torch.as_tensor(mesh.voxel_size, device=device).broadcast_to(3)
        )

    def to(self, device):
        return self.__class__(
            values=self.values.to(device),
            coords=self.coords.to(device),
            aabb=self.aabb.to(device),
            voxel_size=self.voxel_size.to(device),
        )
    

class MeshData(BaseModel):
    """Mesh geometry with vertices, faces, vertex normals and UVs."""
    vertices: FloatTensor                            # (V, 3) vertex positions
    faces: IntegerTensor                             # (F, 3) face indices
    vertex_normals: Optional[FloatTensor] = None     # (V, 3) vertex normals (optional)
    uvs: Optional[FloatTensor] = None                # (V, 2) UV coordinates (optional)
    bvh: Optional[Internal[cumesh.cuBVH]] = None     # BVH tree for ray tracing and projection

    def build_bvh(self):
        self.bvh = cumesh.cuBVH(self.vertices, self.faces)

    def to(self, device):
        return self.__class__(
            vertices=self.vertices.to(device),
            faces=self.faces.to(device),
            vertex_normals=self.vertex_normals.to(device) if self.vertex_normals is not None else None,
            uvs=self.uvs.to(device) if self.uvs is not None else None,
            bvh=self.bvh,  # BVH is not a tensor, so we keep it as is
        )


class MeshDataWithAttributeGrid(MeshData):
    attrs: AttributeGrid

    @classmethod
    def from_mesh_and_attrs(cls, mesh: MeshData, attrs: AttributeGrid) -> "MeshDataWithAttributeGrid":
        return cls(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            uvs=mesh.uvs,
            bvh=mesh.bvh,
            attrs=attrs
        )

    @classmethod
    def from_mesh_with_voxels(cls, mesh: MeshWithVoxel, aabb: FloatTensor = DEFAULT_AABB) -> "MeshDataWithAttributeGrid":
        attrs = AttributeGrid.from_mesh_with_voxels(mesh, aabb)
        return cls(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attrs=attrs
        )

    def to(self, device):
        return self.__class__(
            vertices=self.vertices.to(device),
            faces=self.faces.to(device),
            vertex_normals=self.vertex_normals.to(device) if self.vertex_normals is not None else None,
            uvs=self.uvs.to(device) if self.uvs is not None else None,
            bvh=self.bvh,  # BVH is not a tensor, so we keep it as is
            attrs=self.attrs.to(device)
        )

    



