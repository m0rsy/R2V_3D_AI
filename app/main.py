from __future__ import annotations

import os
import threading
import uuid

from dotenv import load_dotenv

# --------------------------------------------------
# ENVIRONMENT - MUST run before any model imports
# --------------------------------------------------
load_dotenv()

BASE_CACHE = os.getenv("R2V_MODEL_CACHE", r"D:/Grademodels/model_cache")
HY3D_CACHE = os.getenv("R2V_HY3D_CACHE", f"{BASE_CACHE}/hy3dgen")

os.environ["HY3DGEN_CACHE"] = HY3D_CACHE
os.environ["HY3DGEN_HOME"] = HY3D_CACHE
os.environ["HY3DGEN_MODEL_HOME"] = HY3D_CACHE
os.environ["XDG_CACHE_HOME"] = BASE_CACHE

os.environ["HF_HOME"] = BASE_CACHE
os.environ["HF_HUB_CACHE"] = f"{BASE_CACHE}/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{BASE_CACHE}/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

os.environ["USERPROFILE"] = BASE_CACHE
os.environ["HOME"] = BASE_CACHE

_tmp = f"{BASE_CACHE}/tmp"
os.environ["TEMP"] = _tmp
os.environ["TMP"] = _tmp
os.makedirs(_tmp, exist_ok=True)

os.environ.pop("TRANSFORMERS_CACHE", None)

# --------------------------------------------------
# TORCH PERFORMANCE FLAGS
# --------------------------------------------------
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[main] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("[main] No CUDA detected; running on CPU.")

# --------------------------------------------------
# APP IMPORTS (after env is set)
# --------------------------------------------------
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import (
    DEFAULT_MODEL_TIER,
    DEFAULT_MULTI_VIEW,
    DEFAULT_QUALITY_MODE,
    DEFAULT_TEXTURED,
    OUTPUT_DIR,
)
from app.core.pipeline import get_pipelines, init_pipelines, is_ready
from app.core.progress import cleanup_old_jobs, get_job, init_job, list_jobs, update_job
from app.schemas import Generate3DRequest, JobStartResponse, JobStatusResponse
from app.services.generation import run_image_job, run_text_job
from app.services.voice_service import transcribe_and_translate_to_english

app = FastAPI(
    title="AI 3D Generation Backend",
    description=(
        "Text/image/voice to 3D generation with tiered backends "
        "(mo3d-1 baseline and mo3d-pro)."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.on_event("startup")
def on_startup() -> None:
    """Load pipelines once and run a minimal SD warmup."""
    init_pipelines(cache_dir=BASE_CACHE)
    try:
        pipes = get_pipelines(cache_dir=BASE_CACHE)
        sd = pipes["sd"]
        warmup_prompt = (
            "single object, centered, pure white background, studio lighting, simple cube"
        )
        if torch.cuda.is_available():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                sd(
                    warmup_prompt,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                    width=512,
                    height=512,
                )
        else:
            with torch.inference_mode():
                sd(
                    warmup_prompt,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                    width=512,
                    height=512,
                )
        print("[main] Warmup complete (SD only).")
    except Exception as exc:
        print(f"[main] Warmup skipped: {exc}")


@app.get("/health", tags=["Meta"])
def health():
    return {
        "ok": True,
        "pipelines_ready": is_ready(),
        "cuda": torch.cuda.is_available(),
    }


@app.post("/api/generate-3d", response_model=JobStartResponse, tags=["Generate"])
async def start_text_job(payload: Generate3DRequest):
    """Start a text-to-3D generation job. Returns a job_id to poll."""
    job_id = uuid.uuid4().hex
    init_job(job_id)
    update_job(job_id, message="Queued", stage="queued", percent=0)

    threading.Thread(
        target=run_text_job,
        args=(
            job_id,
            payload.prompt,
            payload.preset,
            payload.model,
            payload.textured,
            payload.multi_view,
            payload.quality_mode,
        ),
        daemon=True,
    ).start()

    return JobStartResponse(job_id=job_id)


@app.post("/api/generate-3d-from-image", response_model=JobStartResponse, tags=["Generate"])
async def start_image_job(
    file: UploadFile = File(...),
    preset: str = Form("product"),
    model: str = Form(DEFAULT_MODEL_TIER),
    textured: bool = Form(DEFAULT_TEXTURED),
    multi_view: bool = Form(DEFAULT_MULTI_VIEW),
    quality_mode: str = Form(DEFAULT_QUALITY_MODE),
):
    """Upload an image and start a 3D generation job."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    job_id = uuid.uuid4().hex
    init_job(job_id)
    update_job(job_id, message="Image received. Queued.", stage="queued", percent=0)

    threading.Thread(
        target=run_image_job,
        args=(job_id, image_bytes, preset, model, textured, multi_view, quality_mode),
        daemon=True,
    ).start()

    return JobStartResponse(job_id=job_id)


@app.post("/api/voice-to-3d", response_model=JobStartResponse, tags=["Generate"])
async def start_voice_job(
    file: UploadFile = File(...),
    preset: str = Form("product"),
    model: str = Form(DEFAULT_MODEL_TIER),
    textured: bool = Form(DEFAULT_TEXTURED),
    multi_view: bool = Form(DEFAULT_MULTI_VIEW),
    quality_mode: str = Form(DEFAULT_QUALITY_MODE),
):
    """Upload audio, transcribe with Whisper, then run text-to-3D."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    job_id = uuid.uuid4().hex
    init_job(job_id)
    update_job(
        job_id,
        status="running",
        stage="refining",
        percent=1,
        message="Transcribing voice...",
    )

    try:
        vr = transcribe_and_translate_to_english(audio_bytes)
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            stage="error",
            percent=100,
            error=f"Transcription failed: {exc}",
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    prompt_used = (vr.text_english or vr.transcript_original or "").strip()
    if not prompt_used:
        update_job(
            job_id,
            status="error",
            stage="error",
            percent=100,
            error="Could not transcribe audio; empty result",
        )
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")

    update_job(
        job_id,
        voice_detected_language=vr.detected_language,
        voice_transcript_original=vr.transcript_original,
        voice_text_english=vr.text_english,
        voice_prompt_used=prompt_used,
        message="Voice transcribed. Starting generation...",
        stage="refining",
        percent=2,
    )

    threading.Thread(
        target=run_text_job,
        args=(job_id, prompt_used, preset, model, textured, multi_view, quality_mode),
        daemon=True,
    ).start()

    return JobStartResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
def job_status(job_id: str):
    """Poll job progress and results."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/jobs", tags=["Jobs"])
def list_all_jobs():
    """List all jobs (newest first). Useful for debugging."""
    return list_jobs()


@app.delete("/api/jobs/cleanup", tags=["Jobs"])
def cleanup_jobs(max_age_seconds: float = 3600.0):
    """Remove jobs older than max_age_seconds from memory."""
    removed = cleanup_old_jobs(max_age_seconds=max_age_seconds)
    return {"removed": removed}
