from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from PIL import Image

from app.config import (
    BACKEND_BASE_URL,
    DEFAULT_MODEL_TIER,
    DEFAULT_MULTI_VIEW,
    DEFAULT_QUALITY_MODE,
    DEFAULT_TEXTURED,
    ENABLE_BG_REMOVE,
    MODEL_TIER_MO3D_1,
    MODEL_TIER_MO3D_PRO,
    OUTPUT_DIR,
)
from app.core.progress import set_done, set_error, update_job
from app.services.bg_remove import remove_bg_and_compose_white
from app.services.image_generation import ImageGenerationService
from app.services.mesh_generation import MeshGenerationService
from app.services.prompt_refiner import refine_prompt_for_3d_sd
from app.services.texture_service import TextureGenerationError, TextureService

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipelines: Optional[Dict[str, Any]] = None
_loading_error: Optional[BaseException] = None


def init_pipelines(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all AI pipelines once per process (thread-safe double-checked locking).

    cache_dir:
      Path to your local HuggingFace model cache.
      Example: "D:/Grademodels/model_cache"
      If omitted, falls back to HF_HOME env var.
    """
    global _pipelines, _loading_error

    if _pipelines is not None:
        return _pipelines
    if _loading_error is not None:
        raise RuntimeError(f"Pipeline load previously failed: {_loading_error}") from _loading_error

    with _lock:
        if _pipelines is not None:
            return _pipelines
        if _loading_error is not None:
            raise RuntimeError(f"Pipeline load previously failed: {_loading_error}") from _loading_error

        try:
            from app.core.loader import load_all_pipelines

            if cache_dir:
                os.environ.setdefault("HF_HOME", cache_dir)

            _pipelines = load_all_pipelines(cache_dir=cache_dir)
            return _pipelines

        except BaseException as exc:
            _loading_error = exc
            _pipelines = None
            raise


def get_pipelines(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return loaded pipelines, lazily initializing if needed."""
    return init_pipelines(cache_dir=cache_dir)


def is_ready() -> bool:
    """True only when pipelines are fully loaded with no errors."""
    return _pipelines is not None and _loading_error is None


def reset_pipelines() -> None:
    """
    Force-reset pipeline state (useful for testing or hot-reload scenarios).
    WARNING: Does NOT unload GPU memory. Use with care.
    """
    global _pipelines, _loading_error
    with _lock:
        _pipelines = None
        _loading_error = None


@dataclass
class PipelineOptions:
    model_tier: str = DEFAULT_MODEL_TIER
    textured: bool = DEFAULT_TEXTURED
    multi_view: bool = DEFAULT_MULTI_VIEW
    quality_mode: str = DEFAULT_QUALITY_MODE


def _normalize_model_tier(model_tier: Optional[str]) -> str:
    tier = (model_tier or DEFAULT_MODEL_TIER).strip().lower()
    if tier in (MODEL_TIER_MO3D_1, MODEL_TIER_MO3D_PRO):
        return tier
    return MODEL_TIER_MO3D_1


def _ensure_512_rgb(img: Image.Image) -> Image.Image:
    """Hunyuan runs most predictably with exactly 512x512 RGB input."""
    img = img.convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BICUBIC)
    return img


def _prepare_for_hunyuan(
    *,
    image: Image.Image,
    image_bytes: bytes,
    job_id: str,
    view_idx: int,
) -> Image.Image:
    """
    Background removal + resize for Hunyuan input.
    Respects ENABLE_BG_REMOVE flag from config.
    """
    label = f"view {view_idx + 1}" if view_idx >= 0 else "image"
    if ENABLE_BG_REMOVE:
        update_job(
            job_id,
            stage="bg_remove",
            percent=36,
            message=f"Removing background ({label})...",
        )
        bg = remove_bg_and_compose_white(image_bytes, pad_ratio=0.12, bg_tol=75)
        hunyuan_img = Image.open(io.BytesIO(bg.composited_rgb_png_bytes))
    else:
        update_job(
            job_id,
            stage="bg_remove",
            percent=36,
            message=f"BG removal disabled for {label}; using raw image.",
        )
        hunyuan_img = image
    return _ensure_512_rgb(hunyuan_img)


class GenerationOrchestrator:
    """Coordinates prompt refinement, image generation, mesh generation, and optional texturing."""

    def __init__(self) -> None:
        self.image_service = ImageGenerationService()
        self.mesh_service = MeshGenerationService()
        self.texture_service = TextureService()

    def run_text_job(
        self,
        *,
        job_id: str,
        prompt: str,
        preset: str,
        options: PipelineOptions,
    ) -> None:
        try:
            options = self._normalize_options(options)
            update_job(
                job_id,
                status="running",
                stage="refining",
                percent=2,
                message="Building 3D-friendly prompt...",
                model_tier=options.model_tier,
                textured_requested=options.textured,
                multi_view_requested=options.multi_view,
                quality_mode=options.quality_mode,
            )

            use_cuda = torch.cuda.is_available()
            refined = refine_prompt_for_3d_sd(prompt, preset=preset)
            positive_prompt = refined.positive
            negative_prompt = refined.negative
            image_stage = "sd" if options.model_tier == MODEL_TIER_MO3D_1 else "refining"
            update_job(
                job_id,
                refined_prompt_positive=positive_prompt,
                refined_prompt_negative=negative_prompt,
                stage=image_stage,
                percent=5,
                message="Prompt ready",
            )

            job_dir = OUTPUT_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            generation = self.image_service.generate_from_text(
                job_id=job_id,
                model_tier=options.model_tier,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                use_cuda=use_cuda,
                multi_view=options.multi_view,
                quality_mode=options.quality_mode,
            )
            for warning in generation.warnings:
                logger.warning("job=%s image generation warning: %s", job_id, warning)

            source_images = generation.images
            if not source_images:
                raise RuntimeError("Image generation produced no images.")

            primary_image_path = job_dir / "generated.png"
            source_images[0].save(primary_image_path)

            for idx, extra_img in enumerate(source_images[1:], start=2):
                extra_img.save(job_dir / f"generated_view_{idx}.png")

            processed_images: list[Image.Image] = []
            for idx, image in enumerate(source_images):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                processed = _prepare_for_hunyuan(
                    image=image,
                    image_bytes=buf.getvalue(),
                    job_id=job_id,
                    view_idx=idx,
                )
                processed_images.append(processed)
                suffix = "" if idx == 0 else f"_view_{idx + 1}"
                processed.save(job_dir / f"generated_bg_removed{suffix}.png")

            update_job(
                job_id,
                stage="hunyuan",
                percent=45,
                message="Generating mesh...",
            )
            mesh_output = self.mesh_service.generate_mesh(
                images=processed_images,
                model_tier=options.model_tier,
                multi_view=options.multi_view,
                use_cuda=use_cuda,
            )
            for warning in mesh_output.warnings:
                logger.warning("job=%s mesh generation warning: %s", job_id, warning)

            mesh_path = job_dir / "model.glb"
            mesh_output.mesh.export(str(mesh_path))

            texture_url = None
            textured_model_url = None
            texture_error = None

            if options.textured:
                update_job(
                    job_id,
                    stage="texturing",
                    percent=96,
                    message="Applying texture...",
                )
                textured_glb_path = job_dir / "model_textured.glb"
                try:
                    texture_result = self.texture_service.apply_texture(
                        mesh_path=mesh_path,
                        output_glb_path=textured_glb_path,
                        source_images=processed_images,
                    )
                    if texture_result.texture_glb_path:
                        textured_model_url = (
                            f"{BACKEND_BASE_URL}/outputs/{job_id}/{texture_result.texture_glb_path.name}"
                        )
                    texture_url = (
                        f"{BACKEND_BASE_URL}/outputs/{job_id}/{texture_result.texture_glb_path.name}"
                        if texture_result.texture_glb_path
                        else None
                    )
                    if texture_result.warning:
                        texture_error = texture_result.warning
                except TextureGenerationError as exc:
                    texture_error = str(exc)
                    logger.warning("job=%s texturing unavailable: %s", job_id, exc)
                except Exception as exc:
                    texture_error = str(exc)
                    logger.exception("job=%s texturing failed", job_id)

            set_done(
                job_id,
                image_url=f"{BACKEND_BASE_URL}/outputs/{job_id}/generated.png",
                model_glb_url=f"{BACKEND_BASE_URL}/outputs/{job_id}/model.glb",
                texture_url=texture_url,
                model_tier=options.model_tier,
                textured_requested=options.textured,
                multi_view_requested=options.multi_view,
                quality_mode=options.quality_mode,
                textured_model_glb_url=textured_model_url,
                texture_error=texture_error,
            )
        except Exception as exc:
            logger.exception("run_text_job failed for job=%s", job_id)
            set_error(job_id, str(exc))

    def run_image_job(
        self,
        *,
        job_id: str,
        image_bytes: bytes,
        preset: str,
        options: PipelineOptions,
    ) -> None:
        del preset  # kept for backward compatibility and future presets.
        try:
            options = self._normalize_options(options)
            update_job(
                job_id,
                status="running",
                stage="hunyuan",
                percent=5,
                message="Loading image...",
                model_tier=options.model_tier,
                textured_requested=options.textured,
                multi_view_requested=options.multi_view,
                quality_mode=options.quality_mode,
            )

            use_cuda = torch.cuda.is_available()
            job_dir = OUTPUT_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(job_dir / "input.png")

            processed_images = [
                _prepare_for_hunyuan(
                    image=image,
                    image_bytes=image_bytes,
                    job_id=job_id,
                    view_idx=0,
                )
            ]
            processed_images[0].save(job_dir / "input_bg_removed.png")

            if options.multi_view and options.model_tier == MODEL_TIER_MO3D_PRO:
                logger.info(
                    "job=%s requested multi_view for image input; single-image input kept as fallback",
                    job_id,
                )

            update_job(
                job_id,
                stage="hunyuan",
                percent=45,
                message="Generating mesh...",
            )
            mesh_output = self.mesh_service.generate_mesh(
                images=processed_images,
                model_tier=options.model_tier,
                multi_view=False,
                use_cuda=use_cuda,
            )

            mesh_path = job_dir / "model.glb"
            mesh_output.mesh.export(str(mesh_path))

            texture_url = None
            textured_model_url = None
            texture_error = None

            if options.textured:
                update_job(
                    job_id,
                    stage="texturing",
                    percent=96,
                    message="Applying texture...",
                )
                textured_glb_path = job_dir / "model_textured.glb"
                try:
                    texture_result = self.texture_service.apply_texture(
                        mesh_path=mesh_path,
                        output_glb_path=textured_glb_path,
                        source_images=processed_images,
                    )
                    if texture_result.texture_glb_path:
                        textured_model_url = (
                            f"{BACKEND_BASE_URL}/outputs/{job_id}/{texture_result.texture_glb_path.name}"
                        )
                    texture_url = (
                        f"{BACKEND_BASE_URL}/outputs/{job_id}/{texture_result.texture_glb_path.name}"
                        if texture_result.texture_glb_path
                        else None
                    )
                    if texture_result.warning:
                        texture_error = texture_result.warning
                except TextureGenerationError as exc:
                    texture_error = str(exc)
                    logger.warning("job=%s texturing unavailable: %s", job_id, exc)
                except Exception as exc:
                    texture_error = str(exc)
                    logger.exception("job=%s texturing failed", job_id)

            set_done(
                job_id,
                image_url=f"{BACKEND_BASE_URL}/outputs/{job_id}/input.png",
                model_glb_url=f"{BACKEND_BASE_URL}/outputs/{job_id}/model.glb",
                texture_url=texture_url,
                model_tier=options.model_tier,
                textured_requested=options.textured,
                multi_view_requested=options.multi_view,
                quality_mode=options.quality_mode,
                textured_model_glb_url=textured_model_url,
                texture_error=texture_error,
            )
        except Exception as exc:
            logger.exception("run_image_job failed for job=%s", job_id)
            set_error(job_id, str(exc))

    @staticmethod
    def _normalize_options(options: PipelineOptions) -> PipelineOptions:
        quality = (options.quality_mode or DEFAULT_QUALITY_MODE).strip().lower()
        if quality not in {"fast", "balanced", "quality"}:
            quality = "balanced"
        return PipelineOptions(
            model_tier=_normalize_model_tier(options.model_tier),
            textured=bool(options.textured),
            multi_view=bool(options.multi_view),
            quality_mode=quality,
        )


_orchestrator = GenerationOrchestrator()


def run_text_pipeline(
    job_id: str,
    prompt: str,
    preset: str,
    model_tier: str = DEFAULT_MODEL_TIER,
    textured: bool = DEFAULT_TEXTURED,
    multi_view: bool = DEFAULT_MULTI_VIEW,
    quality_mode: str = DEFAULT_QUALITY_MODE,
) -> None:
    """Backward-compatible public entrypoint for text jobs."""
    _orchestrator.run_text_job(
        job_id=job_id,
        prompt=prompt,
        preset=preset,
        options=PipelineOptions(
            model_tier=model_tier,
            textured=textured,
            multi_view=multi_view,
            quality_mode=quality_mode,
        ),
    )


def run_image_pipeline(
    job_id: str,
    image_bytes: bytes,
    preset: str = "product",
    model_tier: str = DEFAULT_MODEL_TIER,
    textured: bool = DEFAULT_TEXTURED,
    multi_view: bool = DEFAULT_MULTI_VIEW,
    quality_mode: str = DEFAULT_QUALITY_MODE,
) -> None:
    """Backward-compatible public entrypoint for image jobs."""
    _orchestrator.run_image_job(
        job_id=job_id,
        image_bytes=image_bytes,
        preset=preset,
        options=PipelineOptions(
            model_tier=model_tier,
            textured=textured,
            multi_view=multi_view,
            quality_mode=quality_mode,
        ),
    )
