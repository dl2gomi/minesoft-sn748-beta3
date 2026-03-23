from __future__ import annotations

import time

from .params import TrellisParams
import torch
from geometry.mesh.schemas import MeshDataWithAttributeGrid

from logger_config import logger
from schemas.image_convertions import image_tensor_to_pil

from .schemas import TrellisInput, TrellisOutput
from modules.utils import set_random_seed


class MeshGeneratorModule:
    """Runs mesh generation using a provided loaded pipeline."""

    def __init__(self, default_params: TrellisParams):
        self.default_params = default_params

    def generate(self, request: TrellisInput) -> TrellisOutput:
        model = request.model
        assert model.is_ready(), f"{model.settings.model_id} pipeline not loaded."

        set_random_seed(request.seed)

        images = list(request.images)
        images_rgb = [image_tensor_to_pil(image).convert("RGB") for image in images]
        num_images = len(images_rgb)
        if num_images == 0:
            raise ValueError("At least one image tensor is required for mesh generation")

        params = self.default_params.overrided(request.params)

        logger.info(
            f"Generating Trellis {request.seed=} and image size {images_rgb[0].size} "
            f"(Using {num_images} images) | Pipeline: {params.pipeline_type.value} "
            f"| Max Tokens: {params.max_num_tokens} | "
            f"{'Mode: ' + params.mode.value if params.mode.value else ''} | "
            f"Num Samples: {params.num_samples}"
        )
        logger.debug(f"Trellis generation parameters: {params}")

        start = time.time()
        try:
            meshes = model.loaded_pipeline.run_multi_image(
                images=images_rgb,
                seed=request.seed,
                sparse_structure_sampler_params=params.sparse_structure.model_dump(),
                shape_slat_sampler_params=params.shape_slat.model_dump(),
                tex_slat_sampler_params=params.tex_slat.model_dump(),
                mode=params.mode,
                pipeline_type=params.pipeline_type,
                max_num_tokens=params.max_num_tokens,
                num_samples=params.num_samples,
            )

            # Simplify meshes before converting to output format
            for mesh in meshes:
                mesh.simplify()
                
            meshes_data = tuple(MeshDataWithAttributeGrid.from_mesh_with_voxels(mesh) for mesh in meshes)

            generation_time = time.time() - start
            logger.success(f"{model.settings.model_id} finished in {generation_time:.2f}s. {len(meshes)} meshes generated.")
            return TrellisOutput(meshes=meshes_data)
        finally:
            torch.cuda.empty_cache()
