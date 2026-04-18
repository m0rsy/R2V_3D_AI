from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from app.config import (
    HUNYUAN_BASIC_MODEL_ID,
    HUNYUAN_PRO_MODEL_ID,
    LOAD_PRO_HUNYUAN_PIPELINE,
    SD_MODEL_ID,
)


def _load_sd_pipeline(
    device: str,
    dtype: torch.dtype,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Load SD 1.5 or SDXL-Turbo pipeline from local cache (offline only).
    Applies memory optimizations where supported.
    """
    is_sdxl_turbo = "sdxl-turbo" in SD_MODEL_ID.lower()

    if is_sdxl_turbo:
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )

    for method in ("enable_xformers_memory_efficient_attention",):
        if hasattr(pipe, method):
            try:
                getattr(pipe, method)()
                print(f"[loader] SD: {method} enabled")
            except Exception as exc:
                print(f"[loader] SD: {method} skipped ({exc})")

    for method in ("enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, method):
            try:
                getattr(pipe, method)()
            except Exception:
                pass

    print(f"[loader] SD loaded -> {SD_MODEL_ID[:60]}  device={device}  dtype={dtype}")
    return pipe


def _load_hunyuan_pipeline(
    *,
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Load Hunyuan shape generation pipeline from local cache.
    """
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as exc:
        raise ImportError("hy3dgen is not installed. Run: pip install hy3dgen") from exc

    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    shape.to(torch.device(device))

    if hasattr(shape, "enable_xformers_memory_efficient_attention"):
        try:
            shape.enable_xformers_memory_efficient_attention()
            print("[loader] Hunyuan: xformers enabled")
        except Exception as exc:
            print(f"[loader] Hunyuan: xformers skipped ({exc})")

    print(f"[loader] Hunyuan loaded -> {model_id}  device={device}  dtype={dtype}")
    return shape


def load_all_pipelines(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all AI pipelines.

    Returns:
        {
            "sd": <StableDiffusionPipeline or AutoPipelineForText2Image>,
            "shape": <Hunyuan pipeline for mo3d-1>,
            "shape_pro": <optional Hunyuan pipeline for mo3d-pro>,
        }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[loader] device={device} dtype={dtype} cache_dir={cache_dir or '(env default)'}")
    if device == "cpu":
        print("[loader] Running on CPU; generation will be slow.")

    try:
        sd = _load_sd_pipeline(device=device, dtype=dtype, cache_dir=cache_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to load SD pipeline: {exc}") from exc

    try:
        shape = _load_hunyuan_pipeline(
            model_id=HUNYUAN_BASIC_MODEL_ID,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Hunyuan pipeline: {exc}") from exc

    shape_pro = None
    if LOAD_PRO_HUNYUAN_PIPELINE:
        try:
            if HUNYUAN_PRO_MODEL_ID == HUNYUAN_BASIC_MODEL_ID:
                shape_pro = shape
            else:
                shape_pro = _load_hunyuan_pipeline(
                    model_id=HUNYUAN_PRO_MODEL_ID,
                    device=device,
                    dtype=dtype,
                    cache_dir=cache_dir,
                )
        except Exception as exc:
            print(f"[loader] Pro Hunyuan load failed; continuing without it: {exc}")
            shape_pro = None

    assert sd is not None and callable(sd), "SD pipeline is None or not callable"
    assert shape is not None and callable(shape), "Hunyuan pipeline is None or not callable"

    print("[loader] All pipelines ready.")
    return {"sd": sd, "shape": shape, "shape_pro": shape_pro}
