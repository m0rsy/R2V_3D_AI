from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import torch
from PIL import Image

from app.config import (
    GEMINI_API_KEY,
    GEMINI_ENABLED,
    GEMINI_IMAGE_COUNT,
    GEMINI_MODEL,
    MODEL_TIER_MO3D_1,
    MODEL_TIER_MO3D_PRO,
    PRO_IMAGE_FALLBACK_TO_SD,
    SD_GUIDANCE_FAST,
    SD_GUIDANCE_PRO,
    SD_HEIGHT,
    SD_MODEL_ID,
    SD_STEPS_FAST,
    SD_STEPS_PRO,
    SD_WIDTH,
    SDXL_TURBO_GUIDANCE_FAST,
    SDXL_TURBO_STEPS_FAST,
)
from app.core.progress import update_job

logger = logging.getLogger(__name__)


class ImageGenerationError(RuntimeError):
    """Raised when image generation fails."""


@dataclass
class ImageGenerationResult:
    images: list[Image.Image]
    backend_used: str
    warnings: list[str] = field(default_factory=list)


def _sd_callback_builder(job_id: str, total_steps: int, base_percent: int, span_percent: int):
    """Progress callback for SD inference steps."""

    def _cb(step: int, timestep: int, latents):
        p = base_percent + int(((step + 1) / max(1, total_steps)) * span_percent)
        update_job(
            job_id,
            percent=min(p, base_percent + span_percent),
            message=f"SD step {step + 1}/{total_steps}",
        )
        return latents

    return _cb


class ImageGenerationService:
    """Routes text-to-image generation to mo3d-1 or mo3d-pro backends."""

    def generate_from_text(
        self,
        *,
        job_id: str,
        model_tier: str,
        prompt: str,
        negative_prompt: Optional[str],
        use_cuda: bool,
        multi_view: bool = False,
        quality_mode: str = "balanced",
    ) -> ImageGenerationResult:
        tier = (model_tier or MODEL_TIER_MO3D_1).strip().lower()
        if tier == MODEL_TIER_MO3D_PRO:
            return self._generate_pro(
                job_id=job_id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                use_cuda=use_cuda,
                multi_view=multi_view,
                quality_mode=quality_mode,
            )
        return self._generate_mo3d_1(
            job_id=job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_cuda=use_cuda,
        )

    def _generate_mo3d_1(
        self,
        *,
        job_id: str,
        prompt: str,
        negative_prompt: Optional[str],
        use_cuda: bool,
    ) -> ImageGenerationResult:
        # Local import avoids circular dependency with orchestration module.
        from app.core.pipeline import get_pipelines

        pipes = get_pipelines()
        sd_pipe = pipes["sd"]

        is_turbo = "sdxl-turbo" in SD_MODEL_ID.lower()
        steps = SDXL_TURBO_STEPS_FAST if is_turbo else SD_STEPS_FAST
        guidance = SDXL_TURBO_GUIDANCE_FAST if is_turbo else SD_GUIDANCE_FAST
        sd_negative = None if is_turbo else negative_prompt

        cb = _sd_callback_builder(job_id, total_steps=steps, base_percent=5, span_percent=30)
        common = dict(
            prompt=prompt,
            negative_prompt=sd_negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=SD_WIDTH,
            height=SD_HEIGHT,
            callback=cb,
            callback_steps=1,
        )

        if use_cuda:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                image = sd_pipe(**common).images[0]
        else:
            with torch.inference_mode():
                image = sd_pipe(**common).images[0]

        return ImageGenerationResult(images=[image], backend_used="stable-diffusion")

    def _generate_pro(
        self,
        *,
        job_id: str,
        prompt: str,
        negative_prompt: Optional[str],
        use_cuda: bool,
        multi_view: bool,
        quality_mode: str,
    ) -> ImageGenerationResult:
        warnings: list[str] = []

        if GEMINI_ENABLED:
            try:
                images = self._generate_with_gemini(
                    prompt=prompt,
                    multi_view=multi_view,
                    quality_mode=quality_mode,
                )
                if images:
                    return ImageGenerationResult(
                        images=images,
                        backend_used="gemini",
                        warnings=warnings,
                    )
            except Exception as exc:
                warnings.append(f"Gemini generation failed: {exc}")
                logger.exception("Gemini image generation failed for job=%s", job_id)

        if not PRO_IMAGE_FALLBACK_TO_SD:
            raise ImageGenerationError(
                "mo3d-pro image generation requires Gemini (fallback disabled)."
            )

        warnings.append("Falling back to Stable Diffusion for mo3d-pro image generation.")
        fallback = self._generate_pro_with_sd_fallback(
            job_id=job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_cuda=use_cuda,
            multi_view=multi_view,
        )
        fallback.warnings.extend(warnings)
        return fallback

    def _generate_pro_with_sd_fallback(
        self,
        *,
        job_id: str,
        prompt: str,
        negative_prompt: Optional[str],
        use_cuda: bool,
        multi_view: bool,
    ) -> ImageGenerationResult:
        from app.core.pipeline import get_pipelines

        pipes = get_pipelines()
        sd_pipe = pipes["sd"]

        steps = max(SD_STEPS_PRO, SD_STEPS_FAST)
        guidance = max(SD_GUIDANCE_PRO, SD_GUIDANCE_FAST)

        requested_views = max(1, GEMINI_IMAGE_COUNT if multi_view else 1)
        view_suffixes = [
            "front view",
            "three-quarter front view",
            "side view",
            "rear three-quarter view",
            "top-front view",
            "hero angle",
        ]

        images: list[Image.Image] = []
        for idx in range(requested_views):
            suffix = view_suffixes[idx % len(view_suffixes)] if multi_view else ""
            full_prompt = f"{prompt}, {suffix}".strip().strip(",")
            cb = _sd_callback_builder(
                job_id,
                total_steps=steps,
                base_percent=5,
                span_percent=30,
            )
            common = dict(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=SD_WIDTH,
                height=SD_HEIGHT,
                callback=cb,
                callback_steps=1,
            )
            if use_cuda:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    images.append(sd_pipe(**common).images[0])
            else:
                with torch.inference_mode():
                    images.append(sd_pipe(**common).images[0])

        return ImageGenerationResult(images=images, backend_used="stable-diffusion-fallback")

    def _generate_with_gemini(
        self,
        *,
        prompt: str,
        multi_view: bool,
        quality_mode: str,
    ) -> list[Image.Image]:
        if not GEMINI_API_KEY:
            raise ImageGenerationError("GEMINI_API_KEY is missing.")

        prompts = [prompt]
        if multi_view:
            prompts = [
                f"{prompt}, front view",
                f"{prompt}, three-quarter front view",
                f"{prompt}, side view",
                f"{prompt}, rear three-quarter view",
            ]

        try:
            from google import genai  # type: ignore
        except Exception as exc:
            raise ImageGenerationError(
                "Gemini client is not installed. Install package providing `google.genai`."
            ) from exc

        client = genai.Client(api_key=GEMINI_API_KEY)
        images: list[Image.Image] = []

        for p in prompts:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=(
                    "Generate a single centered product-style image for 3D reconstruction. "
                    f"Quality mode: {quality_mode}. Prompt: {p}"
                ),
            )
            if not getattr(response, "candidates", None):
                continue

            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                for part in parts:
                    inline_data = getattr(part, "inline_data", None)
                    data = getattr(inline_data, "data", None) if inline_data else None
                    if not data:
                        continue
                    img = Image.open(BytesIO(data)).convert("RGB")
                    images.append(img)
                if images:
                    break
            if images and not multi_view:
                break

        if not images:
            raise ImageGenerationError("Gemini did not return image bytes.")
        return images
