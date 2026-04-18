from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

# Allow overrides via environment (useful for Modal GPU vs local CPU switches)
_ENV_MODEL   = os.getenv("WHISPER_MODEL_SIZE")
_ENV_DEVICE  = os.getenv("WHISPER_DEVICE")
_ENV_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE")


@dataclass
class VoiceResult:
    detected_language: str
    transcript_original: str
    text_english: str


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_voice_model: WhisperModel | None = None
_voice_model_key: tuple[str, str, str] | None = None


def _cuda_is_usable() -> bool:
    """
    Return True only if CUDA is available AND a small tensor op succeeds.
    Catches environments that report CUDA but cannot actually use it.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def _resolve_settings() -> tuple[str, str, str]:
    """
    Resolve (model_size, device, compute_type) with safe fallbacks.
    - CUDA requested but not usable → CPU int8.
    - CPU compute_type coerced to int8 for speed + compatibility.
    """
    model_size   = (_ENV_MODEL   or WHISPER_MODEL_SIZE   or "medium").strip()
    device       = (_ENV_DEVICE  or WHISPER_DEVICE       or "cpu").lower().strip()
    compute_type = (_ENV_COMPUTE or WHISPER_COMPUTE_TYPE or "int8").lower().strip()

    if device in ("cuda", "gpu"):
        if _cuda_is_usable():
            device = "cuda"
            if compute_type not in ("float16", "float32", "int8_float16"):
                compute_type = "float16"
        else:
            print("[voice_service] CUDA requested but not usable → falling back to CPU int8")
            device, compute_type = "cpu", "int8"

    if device == "cpu" and compute_type not in ("int8", "int8_float16", "float32"):
        compute_type = "int8"

    return model_size, device, compute_type


def get_voice_model() -> WhisperModel:
    """
    Load Whisper once and reuse (lazy singleton).
    Automatically falls back to CPU int8 if the configured device fails.
    """
    global _voice_model, _voice_model_key

    model_size, device, compute_type = _resolve_settings()
    key = (model_size, device, compute_type)

    if _voice_model is not None and _voice_model_key == key:
        return _voice_model

    print(f"[voice_service] Loading Whisper: model={model_size}  device={device}  compute={compute_type}")

    try:
        _voice_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _voice_model_key = key
        return _voice_model
    except Exception as exc:
        print(f"[voice_service] Whisper load failed ({exc}) → retrying with CPU int8")
        _voice_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _voice_model_key = (model_size, "cpu", "int8")
        return _voice_model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_and_translate_to_english(audio_bytes: bytes) -> VoiceResult:
    """
    Transcribe `audio_bytes` (wav / mp3 / m4a — needs ffmpeg for non-wav).

    Steps:
      1. Transcribe in the original language.
      2. Translate to English (separate pass for accuracy).

    Returns VoiceResult with original transcript, language code, and English translation.
    """
    model = get_voice_model()

    # Write to a temp file — faster-whisper needs a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = Path(f.name)

    try:
        # Pass 1: original language transcription
        segs1, info1 = model.transcribe(
            str(tmp_path),
            task="transcribe",
            vad_filter=True,
            beam_size=5,
        )
        original = " ".join(s.text.strip() for s in segs1 if s.text.strip()).strip()
        lang     = (getattr(info1, "language", None) or "unknown").strip()

        # Pass 2: English translation
        segs2, _ = model.transcribe(
            str(tmp_path),
            task="translate",
            vad_filter=True,
            beam_size=5,
        )
        english = " ".join(s.text.strip() for s in segs2 if s.text.strip()).strip()

        # If translation is empty, fall back to original
        if not english and original:
            english = original

        print(f"[voice_service] lang={lang}  original='{original[:80]}'  english='{english[:80]}'")

        return VoiceResult(
            detected_language=lang,
            transcript_original=original,
            text_english=english,
        )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass