from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image
from modules.mesh_generator.params import TrellisParams
import torch
import gc

from config.settings import SettingsConf
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from schemas.types import ImageTensor, ImagesTensor
from schemas.image_convertions import (
    image_tensors_to_images_tensor,
    images_tensor_to_tuple,
    pil_to_image_tensor,
    image_tensor_to_pil,
)
from modules.mesh_generator.schemas import TrellisInput, TrellisOutput
from modules.image_edit.schemas import ImageEditInput
from modules.image_edit.edit_module import ImageEditModule
from modules.image_edit.qwen_edit_pipeline import QwenEditPipeline
from modules.background_removal.ben2_pipeline import BEN2BackgroundRemovalPipeline
from modules.background_removal.birefnet_pipeline import BirefNetBackgroundRemovalPipeline
from modules.background_removal.background_removal_module import BackgroundRemovalModule
from modules.background_removal.schemas import BackgroundRemovalInput
from modules.background_removal.enums import RMBGModelType
from modules.grid_renderer.render import GridViewRenderer
from modules.grid_renderer.schemas import GridRendererInput
from modules.mesh_generator.mesh_generator_module import MeshGeneratorModule
from modules.mesh_generator.enums import TrellisPipeType
from modules.mesh_generator.trellis_pipeline import Trellis2MeshPipeline
from modules.converters.glb_converter import GLBConverter
from modules.perceptual_refiner.refiner import PerceptualRefiner
from modules.judge.duel_manager import DuelManager
from modules.judge.vllm_judge_pipeline import VllmJudgePipeline
from modules.judge.schemas import JudgeInput
from modules.converters.schemas import GLBConverterInput
from modules.utils import image_grid, secure_randint, set_random_seed, save_files
from schemas.bytes import base64_to_image, image_to_base64
    
class GenerationPipeline:
    """
    Generation pipeline 
    """

    def __init__(self, settings: SettingsConf, renderer: Optional[GridViewRenderer] = None) -> None:
        self.settings = settings
        self.renderer = renderer

        # Initialize modules
        self.qwen_pipeline = QwenEditPipeline(settings.qwen, settings.model_versions)
        self.qwen_edit = ImageEditModule(settings.qwen.params)
        self.rmbg_module = BackgroundRemovalModule(settings.background_removal.params)

        # Initialize background removal module
        model_type = self.settings.background_removal.model_type
        if model_type == RMBGModelType.BEN2:
            self.rmbg_pipeline = BEN2BackgroundRemovalPipeline(settings.background_removal, settings.model_versions)
        elif model_type == RMBGModelType.BIREFNET:
            self.rmbg_pipeline = BirefNetBackgroundRemovalPipeline(settings.background_removal, settings.model_versions)
        else:
            raise ValueError(f"Unsupported background removal model: {self.settings.background_removal.model_id}")

        # Initialize prompting libraries for both modes
        self.prompting_library = PromptingLibrary.from_file(settings.qwen.prompt_path_base)

        # Initialize Trellis module
        self.mesh_pipeline = Trellis2MeshPipeline(settings.trellis, settings.model_versions)
        self.mesh_generator = MeshGeneratorModule(settings.trellis.params)
        self.glb_converter = GLBConverter(settings.glb_converter)
        self.perceptual_refiner = PerceptualRefiner(settings.perceptual_refiner)
        self._last_trellis_oom_retry = False
        self._last_qwen_oom_retry = False
        self._last_qwen_edit_skipped = False

        # Initialize Judge module
        if settings.judge.enabled:
            self.judge_pipeline = VllmJudgePipeline(settings.judge)
            self.duel_manager = DuelManager(renderer=renderer)
        else:
            self.judge_pipeline = None
            self.duel_manager = None
        
    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_pipeline.startup()
        await self.rmbg_pipeline.startup()
        await self.mesh_pipeline.startup()
        if self.judge_pipeline is not None:
            await self.judge_pipeline.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_pipeline.shutdown()
        await self.rmbg_pipeline.shutdown()
        await self.mesh_pipeline.shutdown()
        if self.judge_pipeline is not None:
            await self.judge_pipeline.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _aggressive_cuda_cleanup(self) -> None:
        """
        Stronger CUDA cleanup used before OOM retries.
        """
        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            # Ensure queued kernels complete before cache operations.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Releases inter-process cached blocks when possible.
            torch.cuda.ipc_collect()
        except Exception:
            # Cleanup should never block the main flow.
            pass

    def _save_debug_image_tensor(self, image: torch.Tensor, path: Path) -> None:
        try:
            img = image.detach().float().cpu()
            if img.ndim == 2:
                img = img.clamp(0.0, 1.0)
            else:
                img = img.clamp(0.0, 1.0)
            pil = image_tensor_to_pil(img if img.ndim == 3 else torch.stack([img, img, img], dim=-1))
            pil.save(path, format="PNG")
        except Exception as exc:
            logger.warning(f"Failed to save debug image {path}: {exc}")

    def _save_debug_phase_outputs(
        self,
        *,
        seed: int,
        images_edited: ImagesTensor,
        images_without_background: ImagesTensor,
        alpha_hint_mask: Optional[torch.Tensor],
        alpha_hard_mask: Optional[torch.Tensor] = None,
        textures: Optional[dict[str, Image.Image]] = None,
        candidate_idx: Optional[int] = None,
    ) -> None:
        # Temporary debug dump path (no extra folders; keep in configured output root)
        debug_dir = Path(self.settings.output.output_dir)
        suffix = f"_c{candidate_idx}" if candidate_idx is not None else ""
        try:
            if len(images_edited) > 0:
                self._save_debug_image_tensor(images_edited[0], debug_dir / f"qwen_edited_seed{seed}{suffix}.png")
            if len(images_without_background) > 0:
                self._save_debug_image_tensor(images_without_background[0], debug_dir / f"foreground_seed{seed}{suffix}.png")
            if alpha_hint_mask is not None:
                alpha = alpha_hint_mask.detach().float().cpu().clamp(0.0, 1.0)
                if alpha.ndim == 3:
                    alpha = alpha[..., 0]
                alpha_rgb = torch.stack([alpha, alpha, alpha], dim=-1)
                self._save_debug_image_tensor(alpha_rgb, debug_dir / f"alpha_mask_seed{seed}{suffix}.png")
            if alpha_hard_mask is not None:
                alpha_hard = alpha_hard_mask.detach().float().cpu().clamp(0.0, 1.0)
                if alpha_hard.ndim == 3:
                    alpha_hard = alpha_hard[..., 0]
                alpha_hard_rgb = torch.stack([alpha_hard, alpha_hard, alpha_hard], dim=-1)
                self._save_debug_image_tensor(alpha_hard_rgb, debug_dir / f"alpha_mask_hard_seed{seed}{suffix}.png")
            if textures:
                for name, image in textures.items():
                    try:
                        image.save(debug_dir / f"phase_texture_{name}_seed{seed}{suffix}.png", format="PNG")
                    except Exception as exc:
                        logger.warning(f"Failed to save debug texture {name}: {exc}")
        except Exception as exc:
            logger.warning(f"Failed writing debug phase outputs: {exc}")

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(512,512),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(temp_image_bytes).decode("utf-8")

        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=42
        )

        result = await self.generate(request)
        
        if result.glb_file_base64 and self.renderer:
            grid_output = self.renderer.render_grids(GridRendererInput(glb_bytes=[result.glb_file_base64]))
            if grid_output is None:
                logger.warning("Grid view generation failed during warmup")

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return GLB as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            output_type: Desired output type (MESH) (default: MESH)
            
        Returns:
            GLB file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )

        response = await self.generate(request)
        
        return response.glb_file_base64 # bytes
        
    def _edit_images(self, image: Image.Image, seed: int) -> ImagesTensor:
        """
        Edit image based on current mode (multiview or base).
        
        Args:
            image: Input image to edit
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of edited image tensors in HWC format.
        """


        def _is_cuda_oom(exc: BaseException) -> bool:
            msg = str(exc).lower()
            return (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg
                or "cuda" in msg
                or "acceleratorerror" in msg
            )

        def _edit_with_oom_guard(prompting: object, prompt_image: Image.Image) -> tuple[ImagesTensor, bool]:
            """
            Try Qwen edit; on CUDA OOM clear cache and retry once.
            Returns (edited_images, used_retry).
            """
            try:
                out = self.qwen_edit.edit_image(
                    request=ImageEditInput(
                        model=self.qwen_pipeline,
                        prompt_image=prompt_image,
                        seed=seed,
                        prompting=prompting,
                    )
                ).edited_images
                return out, False
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                logger.warning(f"Qwen edit OOM; retrying once after cache clear: {exc}")
                self._last_qwen_oom_retry = True
                self._clean_gpu_memory()
                out = self.qwen_edit.edit_image(
                    request=ImageEditInput(
                        model=self.qwen_pipeline,
                        prompt_image=prompt_image,
                        seed=seed,
                        prompting=prompting,
                    )
                ).edited_images
                return out, True

        if self.settings.trellis.multiview:
            logger.info("Multiview mode: generating multiple views")
            views_prompt = self.prompting_library.promptings['views']

            edited_images: list[ImageTensor] = []
            for prompt_text in views_prompt.prompt:
                logger.debug(f"Editing view with prompt: {prompt_text}")
                try:
                    edited_tensor, _ = _edit_with_oom_guard(prompt_text, image)
                    edited_images.extend(images_tensor_to_tuple(edited_tensor))
                except Exception as exc:
                    if _is_cuda_oom(exc):
                        logger.warning(
                            f"Qwen edit OOM in multiview; falling back to original image only: {exc}"
                        )
                        self._last_qwen_edit_skipped = True
                        self._clean_gpu_memory()
                        return image_tensors_to_images_tensor([pil_to_image_tensor(image.copy())])
                    raise

            original_image = image.copy()
            if edited_images:
                image_size = tuple(edited_images[0].shape[:2])
                original_image = original_image.resize(image_size, Image.Resampling.LANCZOS)
            edited_images.append(pil_to_image_tensor(original_image))

            return image_tensors_to_images_tensor(edited_images)
        
        # Base mode: only clean background, single view (1 image)
        logger.info("Base mode: single view with background cleaning and rotation")
        base_prompt = self.prompting_library.promptings['base']
        logger.debug(f"Editing base view with prompt: {base_prompt}")
        try:
            edited_tensor, _ = _edit_with_oom_guard(base_prompt, image)
            return image_tensors_to_images_tensor(edited_tensor)
        except Exception as exc:
            if _is_cuda_oom(exc):
                logger.warning(
                    f"Qwen edit OOM in base mode after retry; falling back to original image: {exc}"
                )
                self._last_qwen_edit_skipped = True
                self._clean_gpu_memory()
                return image_tensors_to_images_tensor([pil_to_image_tensor(image.copy())])
            raise

    async def generate_mesh(
        self,
        request: GenerationRequest,
    ) -> Tuple[TrellisOutput, Tuple[ImageTensor, ...], Tuple[ImageTensor, ...], Optional[torch.Tensor]]:
        """
        Generate mesh from Trellis pipeline, along with processed images.

        Args:
            request: Generation request with prompt and settings

        Returns:
            Tuple of (mesh, images_edited, images_without_background, alpha_hint_mask)
        """
        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
        set_random_seed(request.seed)
        self._last_qwen_oom_retry = False
        self._last_qwen_edit_skipped = False

        # Decode input image
        image = base64_to_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        images_edited = self._edit_images(image, request.seed)
   
        # 2. Remove background
        background_removal_result = self.rmbg_module.remove_background(
            request=BackgroundRemovalInput(
                model=self.rmbg_pipeline,
                images=image_tensors_to_images_tensor(images_edited),
            )
        )
        images_without_background = background_removal_result.images
        alpha_hint_mask = background_removal_result.alpha_masks[0] if background_removal_result.alpha_masks else None
        alpha_hard_mask = (alpha_hint_mask > 0.5).float() if alpha_hint_mask is not None else None
        if alpha_hint_mask is not None:
            alpha_min = float(alpha_hint_mask.min().item())
            alpha_max = float(alpha_hint_mask.max().item())
            alpha_mean = float(alpha_hint_mask.mean().item())
            alpha_soft_ratio = float(((alpha_hint_mask > 0.02) & (alpha_hint_mask < 0.98)).float().mean().item())
            logger.info(
                "RMBG alpha stats | "
                f"min={alpha_min:.4f} max={alpha_max:.4f} mean={alpha_mean:.4f} "
                f"soft_ratio={alpha_soft_ratio:.4f}"
            )
        self._save_debug_phase_outputs(
            seed=request.seed,
            images_edited=images_edited,
            images_without_background=images_without_background,
            alpha_hint_mask=alpha_hint_mask,
            alpha_hard_mask=alpha_hard_mask,
        )

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
        self._last_trellis_oom_retry = False
        self._last_trellis_pipeline_used = (
            getattr(trellis_params, "pipeline_type", None)
            or getattr(self.settings.trellis.params, "pipeline_type", None)
        )

        # 3. Generate the 3D model with guarded OOM retry.
        def _is_cuda_oom(exc: BaseException) -> bool:
            msg = str(exc).lower()
            return (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg
                or "cuda" in msg
                or "acceleratorerror" in msg
            )

        try:
            mesh_output = self.mesh_generator.generate(
                request=TrellisInput(
                    model=self.mesh_pipeline,
                    images=images_without_background,
                    seed=request.seed,
                    params=trellis_params
                ),
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            logger.warning(f"Trellis OOM during mesh generation; retrying with lighter settings: {exc}")
            self._last_trellis_oom_retry = True
            # Drop exception references and aggressively release CUDA caches before retry.
            exc = None
            self._aggressive_cuda_cleanup()

            default_num_samples = int(getattr(self.settings.trellis.params, "num_samples", 1) or 1)
            fallback = TrellisParams.Overrides(
                pipeline_type=TrellisPipeType.MODE_512,
                num_samples=default_num_samples,
                sparse_structure={"steps": 8},
                shape_slat={"steps": 6},
                tex_slat={"steps": 12},
            )
            self._last_trellis_pipeline_used = TrellisPipeType.MODE_512
            mesh_output = self.mesh_generator.generate(
                request=TrellisInput(
                    model=self.mesh_pipeline,
                    images=images_without_background,
                    seed=request.seed,
                    params=fallback
                ),
            )

        # Phase 2: perceptual refinement (PyTorch3D) before CuMesh conversion.
        if mesh_output and alpha_hint_mask is not None:
            refined_meshes = self.perceptual_refiner.refine(mesh_output.meshes, alpha_hint_mask)
            mesh_output = TrellisOutput(meshes=refined_meshes)

        return mesh_output, images_edited, images_without_background, alpha_hint_mask

    def convert_mesh_to_glb(self, request: GLBConverterInput) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            request: GLB conversion request.

        Returns:
            GLB file as bytes
        """
        start_time = time.time()
        def _is_cuda_oom(exc: BaseException) -> bool:
            msg = str(exc).lower()
            return (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg
                or "cuda" in msg
                or "cumesh" in msg
            )

        try:
            glb_mesh = self.glb_converter.convert(request).glb_mesh
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            logger.warning(f"GLB conversion OOM; retrying with lighter params: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            lighter_params = {
                "remesh": {
                    "enabled": False,
                    "decimation_target": 70000,
                },
                "subdivision": {
                    "iterations": 0,
                    "vertex_reproject": 0.0,
                },
                "texture": {
                    "texture_size": 1024,
                },
            }
            glb_mesh = self.glb_converter.convert(
                GLBConverterInput(mesh=request.mesh, params=lighter_params)
            ).glb_mesh

        buffer = io.BytesIO()
        glb_mesh.export(file_obj=buffer, file_type="glb", extension_webp=False)
        buffer.seek(0)
        
        logger.info(f"GLB conversion time: {time.time() - start_time:.2f}s")
        return buffer.getvalue()

    def prepare_outputs(
        self,
        images_edited: Iterable[ImageTensor],
        images_without_background: Iterable[ImageTensor],
        grid_views_from_num_samples: Optional[Iterable[ImageTensor]],
        glb_mesh: Optional[bytes]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Prepare output files: save to disk if configured and generate base64 strings if needed.

        Args:
            images_edited: List of edited images
            images_without_background: List of images with background removed
            glb_trellis_result: Generated GLB result (optional)
            grid_views_from_num_samples: List of grid views from num samples
        Returns:
            Tuple of (image_edited_base64, image_without_background_base64)
        """
        start_time = time.time()
        
        # Create grid images once for both save and send operations
        image_edited_grid = image_grid(images_edited)
        image_without_background_grid = image_grid(images_without_background)

        if grid_views_from_num_samples:
            grid_views_from_num_samples_grid = image_grid(grid_views_from_num_samples)
        else:
            grid_views_from_num_samples_grid = None

        # Save generated files if configured
        if self.settings.output.save_generated_files:
            save_files(glb_mesh, image_edited_grid, image_without_background_grid, grid_views_from_num_samples_grid)

        # Convert to PNG base64 for response if configured
        image_edited_base64 = None
        image_without_background_base64 = None
        grid_views_from_num_samples_base64 = None
        if self.settings.output.send_generated_files:
            image_edited_base64 = image_to_base64(image_edited_grid)
            image_without_background_base64 = image_to_base64(image_without_background_grid)
            grid_views_from_num_samples_base64 = image_to_base64(grid_views_from_num_samples_grid) if grid_views_from_num_samples_grid is not None else None

        logger.info(f"Output preparation time: {time.time() - start_time:.2f}s")

        return image_edited_base64, image_without_background_base64, grid_views_from_num_samples_base64

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Execute full generation pipeline with output types.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"Request received | Seed: {request.seed} | Prompt Type: {request.prompt_type.value}")

        # Generate mesh and get processed images
        mesh_output, images_edited, images_without_background, alpha_hint_mask = await self.generate_mesh(request)
        alpha_hard_mask = (alpha_hint_mask > 0.5).float() if alpha_hint_mask is not None else None

        glb_mesh = None
        grid_view_winner = None

        self._clean_gpu_memory()

        # Convert meshes to GLB
        glb_meshes: list[bytes] = []
        uv_infos: list[dict | None] = []
        if mesh_output:
            for i, mesh in enumerate(mesh_output.meshes):
                self.glb_converter.set_alpha_hint_mask(alpha_hint_mask)
                glb_output = self.convert_mesh_to_glb(
                    request=GLBConverterInput(
                        mesh=mesh,
                        params=request.glbconv_params,
                    )
                )
                glb_meshes.append(glb_output)
                uv_infos.append(getattr(self.glb_converter, "last_uv_unwrap_info", None))
                self._save_debug_phase_outputs(
                    seed=request.seed,
                    images_edited=images_edited,
                    images_without_background=images_without_background,
                    alpha_hint_mask=alpha_hint_mask,
                    alpha_hard_mask=alpha_hard_mask,
                    textures=getattr(self.glb_converter, "last_textures", None),
                    candidate_idx=i,
                )
        self.glb_converter.set_alpha_hint_mask(None)

        # Judge the meshes
        grid_views_from_num_samples = None
        duel_done = False
        duel_winner = None
        duel_explanation = None
        if self.duel_manager and self.judge_pipeline:
            grid_views_from_num_samples = self.renderer.render_grids(GridRendererInput(glb_bytes=glb_meshes)).grids
            judge_output = await self.duel_manager.judge_grid_views(
                request=JudgeInput(
                    pipeline=self.judge_pipeline,
                    grid_views=grid_views_from_num_samples,
                    prompt_image=pil_to_image_tensor(base64_to_image(request.prompt_image)),
                    seed=request.seed,
                )
            )
            grid_view_winner = judge_output.grid_view_winner
            glb_mesh = glb_meshes[judge_output.winner_index]
            duel_done = True
            duel_winner = judge_output.winner_index
            duel_explanation = getattr(judge_output, "explanation", None)
            winner_uv_info = uv_infos[judge_output.winner_index] if judge_output.winner_index < len(uv_infos) else None
        else:
            glb_mesh = glb_meshes[0]
            winner_uv_info = uv_infos[0] if uv_infos else None

        # Save generated files
        image_edited_base64, image_no_bg_base64, grid_views_from_num_samples_base64 = None, None, None
        if self.settings.output.save_generated_files or self.settings.output.send_generated_files:
            image_edited_base64, image_no_bg_base64, grid_views_from_num_samples_base64 = self.prepare_outputs(
                images_edited,
                images_without_background,
                grid_views_from_num_samples,
                glb_mesh
            )

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s")

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_mesh if glb_mesh is not None else None,
            grid_view_file_base64=image_to_base64(image_tensor_to_pil(grid_view_winner)) if grid_view_winner is not None else None,
            grid_views_from_num_samples_base64=grid_views_from_num_samples_base64,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64,
            trellis_oom_retry=bool(self._last_trellis_oom_retry),
            qwen_oom_retry=bool(self._last_qwen_oom_retry),
            qwen_edit_skipped=bool(self._last_qwen_edit_skipped),
            trellis_pipeline_used=(
                getattr(self._last_trellis_pipeline_used, "value", self._last_trellis_pipeline_used)
                if self._last_trellis_pipeline_used is not None
                else None
            ),
            uv_unwrap_mode=winner_uv_info.get("mode") if isinstance(winner_uv_info, dict) else None,
            uv_unwrap_reason=winner_uv_info.get("reason") if isinstance(winner_uv_info, dict) else None,
            cluster_count=winner_uv_info.get("num_charts") if isinstance(winner_uv_info, dict) else None,
            duel_done=duel_done,
            duel_winner=duel_winner,
            duel_explanation=duel_explanation,
        )
        
        return response
