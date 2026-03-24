from pydantic import BaseModel, Field


class PerceptualRefinerConfig(BaseModel):
    enabled: bool = True
    iterations: int = Field(default=120, ge=1, le=1000)
    image_size: int = Field(default=256, ge=64, le=1024)
    lr: float = Field(default=1e-3, gt=0.0)
    max_vertex_shift: float = Field(default=0.08, gt=0.0)
    raster_blur_radius: float = Field(default=1e-4, ge=0.0)
    raster_faces_per_pixel: int = Field(default=32, ge=1, le=128)
    w_silhouette: float = Field(default=1.0, ge=0.0)
    w_laplacian: float = Field(default=0.03, ge=0.0)
    w_normal: float = Field(default=0.01, ge=0.0)
    w_edge: float = Field(default=0.02, ge=0.0)
    refine_alpha_attr: bool = True
    alpha_refine_steps: int = Field(default=40, ge=1, le=400)
    alpha_refine_lr: float = Field(default=5e-2, gt=0.0)
