from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from app.config import (
    DEFAULT_MODEL_TIER,
    DEFAULT_MULTI_VIEW,
    DEFAULT_QUALITY_MODE,
    DEFAULT_TEXTURED,
    MODEL_TIER_MO3D_1,
    MODEL_TIER_MO3D_PRO,
)

ModelTier = Literal["mo3d-1", "mo3d-pro"]
QualityMode = Literal["fast", "balanced", "quality"]


class Generate3DRequest(BaseModel):
    prompt: str
    preset: str = "product"   # product | studio | photoreal | FAST | QUALITY
    model: ModelTier = DEFAULT_MODEL_TIER  # mo3d-1 (default) | mo3d-pro
    textured: bool = DEFAULT_TEXTURED
    multi_view: bool = DEFAULT_MULTI_VIEW
    quality_mode: QualityMode = DEFAULT_QUALITY_MODE

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("preset")
    @classmethod
    def preset_valid(cls, v: str) -> str:
        allowed = {"product", "studio", "photoreal", "FAST", "QUALITY"}
        return v if v in allowed else "product"

    @field_validator("model")
    @classmethod
    def model_valid(cls, v: str) -> str:
        value = (v or DEFAULT_MODEL_TIER).strip().lower()
        if value in (MODEL_TIER_MO3D_1, MODEL_TIER_MO3D_PRO):
            return value
        return MODEL_TIER_MO3D_1

    @field_validator("quality_mode")
    @classmethod
    def quality_mode_valid(cls, v: str) -> str:
        value = (v or DEFAULT_QUALITY_MODE).strip().lower()
        if value in {"fast", "balanced", "quality"}:
            return value
        return "balanced"


class JobStartResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str

    status: Literal["queued", "running", "done", "error"]
    stage: Literal[
        "queued",
        "refining",
        "sd",
        "bg_remove",
        "hunyuan",
        "texturing",
        "exporting",
        "done",
        "error",
    ]

    percent: int
    message: Optional[str] = None
    error: Optional[str] = None

    image_url: Optional[str] = None
    model_glb_url: Optional[str] = None
    texture_url: Optional[str] = None
    textured_model_glb_url: Optional[str] = None
    texture_error: Optional[str] = None

    voice_detected_language: Optional[str] = None
    voice_transcript_original: Optional[str] = None
    voice_text_english: Optional[str] = None
    voice_prompt_used: Optional[str] = None

    refined_prompt_positive: Optional[str] = None
    refined_prompt_negative: Optional[str] = None

    model_tier: Optional[ModelTier] = None
    textured_requested: Optional[bool] = None
    multi_view_requested: Optional[bool] = None
    quality_mode: Optional[QualityMode] = None
