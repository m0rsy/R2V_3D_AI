from __future__ import annotations

"""
Backward-compatible job entrypoints.

The main orchestration now lives in app.core.pipeline.
These wrappers preserve existing import paths used by the API layer.
"""

from app.config import (
    DEFAULT_MODEL_TIER,
    DEFAULT_MULTI_VIEW,
    DEFAULT_QUALITY_MODE,
    DEFAULT_TEXTURED,
)
from app.core.pipeline import run_image_pipeline, run_text_pipeline


def run_text_job(
    job_id: str,
    prompt: str,
    preset: str,
    model_tier: str = DEFAULT_MODEL_TIER,
    textured: bool = DEFAULT_TEXTURED,
    multi_view: bool = DEFAULT_MULTI_VIEW,
    quality_mode: str = DEFAULT_QUALITY_MODE,
) -> None:
    run_text_pipeline(
        job_id=job_id,
        prompt=prompt,
        preset=preset,
        model_tier=model_tier,
        textured=textured,
        multi_view=multi_view,
        quality_mode=quality_mode,
    )


def run_image_job(
    job_id: str,
    image_bytes: bytes,
    preset: str = "product",
    model_tier: str = DEFAULT_MODEL_TIER,
    textured: bool = DEFAULT_TEXTURED,
    multi_view: bool = DEFAULT_MULTI_VIEW,
    quality_mode: str = DEFAULT_QUALITY_MODE,
) -> None:
    run_image_pipeline(
        job_id=job_id,
        image_bytes=image_bytes,
        preset=preset,
        model_tier=model_tier,
        textured=textured,
        multi_view=multi_view,
        quality_mode=quality_mode,
    )
