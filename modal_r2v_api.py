# modal_r2v_api.py
# R2V on Modal (GPU): text/image/voice -> SD -> bg remove (cached rembg) -> Hunyuan3D-2 (shape) -> GLB
#
# FIXED (deadline-safe):
# - Ensures Modal methods exist: text_to_3d/image_to_3d/voice_to_3d/ping
# - Fixes CLIP scoring bug (list-of-lists -> float(list))
# - Stops Ollama from returning tutorials/steps (strict system + stop tokens + detector + fallback)
# - Voice->3D uses same refiner as text->3D
# - Respects LOCAL_FILES_ONLY: if true -> offline mode (no HF internet calls)
# - Reduces wasted credits by removing random BUILD_ID

import io
import os
import re
import time
import uuid
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import modal

# ---------------- App/Volume ----------------
APP_NAME = "r2v-gpu-api"
VOLUME_NAME = "r2v-model-cache"

MODEL_MOUNT_PATH = "/model_cache"
OUTPUT_DIR = Path(MODEL_MOUNT_PATH) / "outputs"

HF_CACHE_ROOT = Path(MODEL_MOUNT_PATH) / ".cache" / "huggingface"
HF_HUB_DIR = HF_CACHE_ROOT / "hub"
TORCH_HOME_DIR = Path(MODEL_MOUNT_PATH) / ".cache" / "torch"

HY3D_CACHE_DIR = Path(MODEL_MOUNT_PATH) / ".cache" / "hy3dgen"
WHISPER_CACHE_DIR = Path(MODEL_MOUNT_PATH) / ".cache" / "faster_whisper"
OLLAMA_MODELS_DIR = Path(MODEL_MOUNT_PATH) / ".ollama"

app = modal.App(APP_NAME)
model_vol = modal.Volume.from_name(VOLUME_NAME)

# ✅ IMPORTANT: keep constant to avoid rebuilding image every run (saves credits)
BUILD_ID = "r2v"

# ✅ Modal Secret
R2V_SECRET = modal.Secret.from_name("r2v-env")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "curl",
        "zstd",
        "ca-certificates",
        "libgl1",
        "libgl1-mesa-glx",
        "libglx-mesa0",
        "libglib2.0-0",
        "libxrender1",
        "libsm6",
        "libxext6",
        "libopengl0",
        "ffmpeg",
    )
    .env({"BUILD_ID": BUILD_ID})
    .pip_install(
        "fastapi==0.115.0",
        "uvicorn==0.30.6",
        "pydantic==2.*",
        "pillow",
        "python-multipart",
        "numpy",
        "trimesh",
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "hy3dgen",
        "faster-whisper",
        "requests",
        "rembg",
        "onnxruntime",
    )
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
)

# ---------------- Cache env ----------------
def _hf_offline_if_local_only():
    if os.environ.get("LOCAL_FILES_ONLY", "false").lower() == "true":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _set_cache_env() -> str:
    os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_HUB_DIR)

    os.environ["TORCH_HOME"] = str(TORCH_HOME_DIR)
    os.environ["XDG_CACHE_HOME"] = str(Path(MODEL_MOUNT_PATH) / ".cache")
    os.environ["HOME"] = MODEL_MOUNT_PATH

    os.environ.setdefault("HY3DGEN_CACHE", str(HY3D_CACHE_DIR))
    os.environ.setdefault("HY3DGEN_HOME", str(HY3D_CACHE_DIR))
    os.environ.setdefault("HY3DGEN_MODEL_HOME", str(HY3D_CACHE_DIR))

    # Store Ollama models inside Volume
    os.environ["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)

    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    HF_HUB_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_HOME_DIR.mkdir(parents=True, exist_ok=True)
    HY3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _hf_offline_if_local_only()
    return str(HF_HUB_DIR)

# ---------------- Ollama settings ----------------
USE_OLLAMA_REFINER = os.environ.get("USE_OLLAMA_REFINER", "true").lower() == "true"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "60"))
OLLAMA_PULL_ON_START = os.environ.get("OLLAMA_PULL_ON_START", "false").lower() == "true"

NEG_FALLBACK = (
    "text, watermark, logo, signature, blurry, low quality, jpeg artifacts, "
    "cropped, cut off, out of frame, multiple objects, cluttered background, busy background, "
    "hands, holding, people, props, packaging, box, stand, pedestal, "
    "colored background, orange background, black background, gradient background, "
    "studio set, backdrop, wall, floor"
)

GLOBAL_NEGATIVE_STRONG = (
    "room corner, corner, wall corner, backdrop corner, v-shape corner, seam line, "
    "floor, wall, horizon line, studio backdrop, background geometry, "
    "two-tone background, split background, gradient, vignette, "
    "collage, multi panel, split screen, diptych, triptych, grid, "
    "duplicate, two objects, multiple objects, extra object, second object, "
    "fisheye, wide angle, extreme perspective, perspective distortion"
)

def _ensure_ollama_model_present(model_name: str) -> None:
    import subprocess
    try:
        subprocess.run(["ollama", "show", model_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except Exception:
        pass
    subprocess.run(["ollama", "pull", model_name], check=False)

def _start_ollama_server_if_enabled() -> None:
    if not USE_OLLAMA_REFINER:
        return

    import subprocess
    import requests

    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # wait ready
    for _ in range(200):
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.25)

    if OLLAMA_PULL_ON_START:
        _ensure_ollama_model_present(OLLAMA_MODEL)

def _looks_like_tutorial(text: str) -> bool:
    t = (text or "").lower()
    bad = [
        "step", "download", "install", "blender", "maya", "sketchup",
        "here's a guide", "here is a guide", "ctrl+", "file >", "extrude",
        "repeat this process", "go to"
    ]
    if any(b in t for b in bad):
        return True
    if len(t.split()) > 85:
        return True
    return False

def _cleanup_ollama_output(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL).strip()

    # remove markdown artifacts
    t = t.replace("**", " ")
    t = t.replace("###", " ")
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.M)
    t = re.sub(r"[*#>`_]+", " ", t)

    # remove common lead-ins
    t = re.sub(r"^(here.*?:\s*)", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(output.*?:\s*)", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(stable diffusion prompt.*?:\s*)", "", t, flags=re.IGNORECASE).strip()

    # collapse whitespace
    t = " ".join(t.split()).strip()

    # hard cut if it starts explaining
    for cut in [" step ", "download", "install", "blender", "maya", "sketchup", "guide", "tutorial"]:
        idx = t.lower().find(cut)
        if idx != -1 and idx > 35:
            t = t[:idx].strip()
            break

    return t

def _ollama_generate_one_line(user_prompt: str, preset: str) -> str:
    import requests

    preset_in = (preset or "3D RENDER").strip().upper()
    up = (user_prompt or "").strip()

    system = (
    "You are a Stable Diffusion prompt engineer for SINGLE OBJECT 3D asset images.\n"
    "Return ONLY ONE SINGLE LINE prompt: comma-separated visual descriptors.\n"
    "ABSOLUTE RULES:\n"
    "- NO steps, NO instructions, NO tutorial, NO explanations, NO markdown, NO bullets, NO headings.\n"
    "- NO quotes, NO JSON, NO colon-labeled fields.\n"
    "- DO NOT mention Blender/Maya/SketchUp or any software.\n"
    "STYLE RULES:\n"
    "- Single object only, centered composition.\n"
    "- Isolated product render, high-key studio lighting.\n"
    "- White seamless background (pure white), minimal soft shadow.\n"
    "- Clean silhouette, sharp focus, high detail.\n"
    "- Always include the exact phrases: isolated, white seamless background, product photo.\n"
    "OUTPUT:\n"
    "- Exactly one line. No newline.\n"
    "- 20 to 60 words max.\n"
)


    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": f"{preset_in}: {up}",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 160,
            "stop": [
                "\n", "###", "**",
                "Step", "step", "Steps", "steps",
                "1.", "2.", "3.",
                "- ", "* ", "•",
                "download", "install",
                "blender", "maya", "sketchup",
                "tutorial", "guide",
                "file >", "ctrl", "press",
                "to create", "you need", "you'll need", "you will need"
            ],
        },
    }

    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def refine_prompt_llm(user_prompt: str, preset: str) -> str:
    raw = _ollama_generate_one_line(user_prompt=user_prompt, preset=preset)
    cleaned = _cleanup_ollama_output(raw)

    if (not cleaned) or _looks_like_tutorial(cleaned):
        raw2 = _ollama_generate_one_line(user_prompt=f"{user_prompt} (ONE LINE SD PROMPT ONLY)", preset=preset)
        cleaned2 = _cleanup_ollama_output(raw2)
        if cleaned2 and not _looks_like_tutorial(cleaned2):
            return cleaned2

    if (not cleaned) or _looks_like_tutorial(cleaned):
        raise ValueError(f"Ollama returned non-prompt text: {cleaned[:140]}")
    return cleaned

def _fallback_refine(user_prompt: str, preset: str) -> str:
    p = re.sub(r"\s+", " ", (user_prompt or "").strip())
    preset_up = (preset or "3D RENDER").strip().upper()
    return (
        f"{preset_up}, {p}, single object, centered, studio lighting, pure white background, "
        "minimal shadow, clean silhouette, sharp focus, high detail"
    )

def refine_prompt_with_negative(user_prompt: str, preset: str = "3D RENDER") -> Tuple[str, str]:
    user_prompt = (user_prompt or "").strip()
    preset = (preset or "3D RENDER").strip()

    if USE_OLLAMA_REFINER:
        try:
            pos = refine_prompt_llm(user_prompt, preset)
        except Exception as e:
            print("🔥 Ollama refine failed (fallback):", repr(e))
            pos = _fallback_refine(user_prompt, preset)
    else:
        pos = _fallback_refine(user_prompt, preset)

    neg = f"{NEG_FALLBACK}, {GLOBAL_NEGATIVE_STRONG}"
    return pos, neg

# ---------------- Utility: list ollama models ----------------
@app.function(
    image=image,
    volumes={MODEL_MOUNT_PATH: model_vol},
    secrets=[R2V_SECRET],
    timeout=60 * 10,
)
def list_ollama_models() -> Dict[str, Any]:
    _set_cache_env()
    import subprocess
    import requests

    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(80):
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)

    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
    except Exception as e:
        out = f"ERROR: {e}"
    return {"ollama_list": out}

# ---------------- Prefetch (one-time warmup) ----------------
@app.function(
    image=image,
    volumes={MODEL_MOUNT_PATH: model_vol},
    secrets=[R2V_SECRET],
    gpu="any",
    timeout=60 * 60,
)
def prefetch_sd_hunyuan_clip() -> Dict[str, Any]:
    """
    One-time warmup:
      - starts Ollama
      - warms SD + Hunyuan + CLIP into Volume cache
    """
    cache_dir = _set_cache_env()
    _start_ollama_server_if_enabled()

    t0 = time.time()
    import torch
    from diffusers import StableDiffusionPipeline
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from transformers import CLIPModel, CLIPProcessor

    SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    HUNYUAN_MODEL_ID = os.environ.get("HUNYUAN_MODEL_ID", "tencent/Hunyuan3D-2")
    CLIP_MODEL_ID = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    sd = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID, torch_dtype=dtype, cache_dir=cache_dir, local_files_only=False, safety_checker=None
    ).to(device)
    _ = sd("a cube on white background", num_inference_steps=1, guidance_scale=0.0, width=256, height=256).images[0]
    del sd

    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HUNYUAN_MODEL_ID, torch_dtype=dtype, cache_dir=cache_dir, local_files_only=False
    )
    shape.to(device)
    del shape

    _ = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, local_files_only=False)
    _ = CLIPModel.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, local_files_only=False)

    model_vol.commit()
    return {"ok": True, "seconds": round(time.time() - t0, 2)}

# ---------------- Image helpers ----------------
def _ensure_512_rgb(pil_img):
    from PIL import Image
    img = pil_img.convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.BICUBIC)
    return img

def _place_on_square_and_resize(pil_rgb, out_size: int = 512, pad_ratio: float = 0.12):
    from PIL import Image
    import numpy as np

    rgb = np.array(pil_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    fg = np.where(np.max(255 - rgb, axis=2) > 8)
    if fg[0].size == 0:
        return pil_rgb.resize((out_size, out_size), Image.LANCZOS)

    y0, y1 = int(fg[0].min()), int(fg[0].max()) + 1
    x0, x1 = int(fg[1].min()), int(fg[1].max()) + 1
    bw, bh = (x1 - x0), (y1 - y0)
    pad_px = int(max(bw, bh) * pad_ratio)

    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(w, x1 + pad_px)
    y1 = min(h, y1 + pad_px)

    crop = pil_rgb.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    side = max(cw, ch, 8)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas.resize((out_size, out_size), Image.LANCZOS)

# ---------------- GPU Server ----------------
@app.cls(
    image=image,
    gpu="any",
    volumes={MODEL_MOUNT_PATH: model_vol},
    secrets=[R2V_SECRET],
    timeout=60 * 60,
    min_containers=1,
    max_containers=4,
    scaledown_window=60 * 10,
)
class R2VModelServer:
    ready: bool = False
    load_seconds: Optional[float] = None
    load_error: Optional[str] = None

    sd_pipe = None
    shape_pipe = None
    whisper_model = None

    clip_model = None
    clip_processor = None
    rembg_session = None

    @modal.enter()
    def load(self):
        t0 = time.time()
        try:
            cache_dir = _set_cache_env()
            _start_ollama_server_if_enabled()

            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            self.sd_pipe, self.shape_pipe = self._load_pipelines(cache_dir)
            self._load_clip(cache_dir)
            self._load_rembg_session()

            voice_enabled = os.environ.get("VOICE_ENABLED", "true").lower() == "true"
            if voice_enabled:
                self.whisper_model = self._load_whisper()

            self.ready = True
            self.load_error = None
            self.load_seconds = round(time.time() - t0, 2)
            print(f"✅ Pipelines loaded in {self.load_seconds}s")
        except Exception:
            self.ready = False
            self.load_seconds = round(time.time() - t0, 2)
            self.load_error = traceback.format_exc()
            print("🔥 PIPELINE LOAD FAILED:\n", self.load_error)

    def _load_pipelines(self, cache_dir: str):
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        HUNYUAN_MODEL_ID = os.environ.get("HUNYUAN_MODEL_ID", "tencent/Hunyuan3D-2")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        local_only = os.environ.get("LOCAL_FILES_ONLY", "true").lower() == "true"

        sd = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_only,
            safety_checker=None,
        ).to(device)

        sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
        sd.set_progress_bar_config(disable=True)

        try:
            sd.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            sd.vae.enable_slicing()
            sd.vae.enable_tiling()
        except Exception:
            pass

        shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN_MODEL_ID,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        shape.to(device)
        return sd, shape

    def _load_clip(self, cache_dir: str):
        import torch
        import time as _time
        from transformers import CLIPModel, CLIPProcessor

        model_id = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        local_only = os.environ.get("LOCAL_FILES_ONLY", "true").lower() == "true"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        last_err = None
        for attempt in range(1, 6):
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(
                    model_id, cache_dir=cache_dir, local_files_only=local_only
                )
                self.clip_model = CLIPModel.from_pretrained(
                    model_id, cache_dir=cache_dir, local_files_only=local_only
                ).to(device)
                self.clip_model.eval()
                if device == "cuda":
                    self.clip_model = self.clip_model.half()
                return
            except Exception as e:
                last_err = e
                if not local_only:
                    print(f"⚠️ CLIP load failed (attempt {attempt}/5): {repr(e)}")
                    _time.sleep(2.0 * attempt)
                else:
                    break

        raise RuntimeError(f"CLIP load failed. local_only={local_only}. last_err={repr(last_err)}")

    def _load_rembg_session(self):
        from rembg import new_session
        self.rembg_session = new_session(os.environ.get("REMBG_MODEL", "u2net"))

    def _load_whisper(self):
        from faster_whisper import WhisperModel
        import torch

        model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
            if compute_type not in ("float16", "float32", "int8_float16"):
                compute_type = "float16"
        else:
            device = "cpu"
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
            if compute_type not in ("int8", "float32", "int8_float16"):
                compute_type = "int8"

        return WhisperModel(model_size, device=device, compute_type=compute_type, download_root=str(WHISPER_CACHE_DIR))

    def _truncate_sd_77(self, text: str) -> str:
        t = (text or "").strip()
        tok = getattr(self.sd_pipe, "tokenizer", None)
        if tok is None:
            return t[:900]
        enc = tok(t, truncation=True, max_length=77, return_tensors=None)
        return tok.decode(enc["input_ids"], skip_special_tokens=True).strip()

    def _sd_text_to_images(
        self,
        prompt: str,
        negative_prompt: str,
        seed: Optional[int],
        steps: int,
        guidance: float,
        width: int,
        height: int,
        num_candidates: int,
    ):
        import torch

        images = []
        base_seed = int(seed) if seed is not None else None
        for i in range(int(num_candidates)):
            gen = None
            if base_seed is not None:
                gen = torch.Generator(device=self.sd_pipe.device).manual_seed(base_seed + i * 101)

            out = self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                generator=gen,
            )
            images.append(out.images[0])
        return images

    # ✅ FIXED: one text only -> returns list[float]
    def _clip_scores(self, text: str, images: List) -> List[float]:
        import torch
        device = self.clip_model.device

        inputs = self.clip_processor(
            text=[text],     # ✅ single text
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device.type == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.clip_model(**inputs)
            logits = out.logits_per_image  # [num_images, 1]
        return logits.squeeze(1).float().detach().cpu().tolist()

    def _pick_best_candidate(self, user_prompt: str, candidates: List):
        import numpy as np
        scores = self._clip_scores(user_prompt, candidates)
        best_i = 0
        best_s = -1e9
        for i, img in enumerate(candidates):
            rgb = np.array(img.convert("RGB"))
            obj_ratio = float((rgb.mean(axis=2) < 245).mean())
            penalty = 100.0 if obj_ratio < 0.04 else 0.0
            s = float(scores[i]) - penalty
            if s > best_s:
                best_s = s
                best_i = i
        return candidates[best_i], float(best_s)

    def _bg_is_nearly_white(self, pil_img) -> bool:
        import numpy as np
        rgb = np.array(pil_img.convert("RGB"))
        b = 14
        border = np.concatenate(
            [
                rgb[:b, :, :].reshape(-1, 3),
                rgb[-b:, :, :].reshape(-1, 3),
                rgb[:, :b, :].reshape(-1, 3),
                rgb[:, -b:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        return float(border.mean(axis=0).min()) >= 248.0

    def remove_bg_and_compose_white(self, img_bytes: bytes, out_size: int = 512, pad_ratio: float = 0.12) -> bytes:
        from PIL import Image
        from rembg import remove

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self._bg_is_nearly_white(img):
            final = _place_on_square_and_resize(img, out_size=out_size, pad_ratio=pad_ratio)
            buf = io.BytesIO()
            final.save(buf, format="PNG")
            return buf.getvalue()

        rgba_in = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        cut_out = remove(rgba_in, session=self.rembg_session)

        if isinstance(cut_out, (bytes, bytearray)):
            cut = Image.open(io.BytesIO(cut_out)).convert("RGBA")
        else:
            cut = cut_out.convert("RGBA")

        white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(white, cut).convert("RGB")

        final = _place_on_square_and_resize(comp, out_size=out_size, pad_ratio=pad_ratio)
        buf = io.BytesIO()
        final.save(buf, format="PNG")
        return buf.getvalue()

    def _extract_mesh(self, shape_result):
        if isinstance(shape_result, (list, tuple)):
            return shape_result[0]
        if isinstance(shape_result, dict) and "mesh" in shape_result:
            return shape_result["mesh"]
        if isinstance(shape_result, dict) and len(shape_result) > 0:
            return next(iter(shape_result.values()))
        return shape_result

    def _clean_mesh(self, mesh):
        import trimesh
        try:
            parts = mesh.split(only_watertight=False)
            if parts and len(parts) > 1:
                parts = sorted(parts, key=lambda m: float(m.area), reverse=True)
                mesh = parts[0]
        except Exception:
            pass

        try:
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.remove_infinite_values()
            mesh.merge_vertices()
        except Exception:
            pass

        if os.environ.get("MESH_SMOOTHING", "true").lower() == "true":
            iters = int(os.environ.get("MESH_SMOOTH_ITERS", "4"))
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=iters)
            except Exception:
                pass

        return mesh

    def _hunyuan_params(self, preset: str):
        p = (preset or "product").lower()
        if p == "quality":
            return (
                int(os.environ.get("HY_STEPS_QUALITY", "26")),
                float(os.environ.get("HY_GUIDANCE_QUALITY", "4.6")),
                int(os.environ.get("HY_OCTREE_QUALITY", "384")),
                int(os.environ.get("HY_CHUNKS_QUALITY", "192")),
            )
        return (
            int(os.environ.get("HY_STEPS_FAST", "16")),
            float(os.environ.get("HY_GUIDANCE_FAST", "4.2")),
            int(os.environ.get("HY_OCTREE_FAST", "256")),
            int(os.environ.get("HY_CHUNKS_FAST", "128")),
        )

    def _run_hunyuan_shape(self, pil_img_512, out_glb_path: Path, preset: str) -> None:
        import torch
        hy_steps, hy_guidance, hy_octree, hy_chunks = self._hunyuan_params(preset)
        hy_pbar = os.environ.get("HY_ENABLE_PBAR", "false").lower() == "true"

        device = self.shape_pipe.device if hasattr(self.shape_pipe, "device") else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        use_cuda = (getattr(device, "type", "") == "cuda") and torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        kwargs = dict(
            image=pil_img_512,
            num_inference_steps=hy_steps,
            guidance_scale=hy_guidance,
            octree_resolution=hy_octree,
            num_chunks=hy_chunks,
            enable_pbar=hy_pbar,
            output_type="trimesh",
        )

        if use_cuda:
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                shape_result = self.shape_pipe(**kwargs)
        else:
            with torch.inference_mode():
                shape_result = self.shape_pipe(**kwargs)

        mesh = self._extract_mesh(shape_result)
        mesh = self._clean_mesh(mesh)

        out_glb_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_glb_path))

    def _sd_defaults(self, preset: str, steps: Optional[int], guidance: Optional[float], candidates: Optional[int]):
        p = (preset or "product").lower()
        if p == "quality":
            s = int(steps or os.environ.get("SD_STEPS_QUALITY", "30"))
            g = float(guidance or os.environ.get("SD_GUIDANCE_QUALITY", "7.0"))
            n = int(candidates or os.environ.get("SD_NUM_CANDIDATES_QUALITY", "4"))
        else:
            s = int(steps or os.environ.get("SD_STEPS_FAST", "18"))
            g = float(guidance or os.environ.get("SD_GUIDANCE_FAST", "6.5"))
            n = int(candidates or os.environ.get("SD_NUM_CANDIDATES_FAST", "2"))

        s = max(8, min(50, s))
        g = max(0.0, min(12.0, g))
        n = max(1, min(6, n))
        return s, g, n

    def _voice_to_text_and_english(self, audio_bytes: bytes) -> Dict[str, Any]:
        import tempfile
        from pathlib import Path

        if self.whisper_model is None:
            raise RuntimeError("Whisper is not loaded. Set VOICE_ENABLED=true and redeploy.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = Path(f.name)

        try:
            beam = int(os.environ.get("WHISPER_BEAM", "5"))

            seg1, info1 = self.whisper_model.transcribe(
                str(tmp_path), task="transcribe", vad_filter=True, beam_size=beam
            )
            original = " ".join(s.text.strip() for s in seg1 if s.text).strip()
            lang = (getattr(info1, "language", None) or "unknown")

            seg2, _info2 = self.whisper_model.transcribe(
                str(tmp_path), task="translate", vad_filter=True, beam_size=beam
            )
            english = " ".join(s.text.strip() for s in seg2 if s.text).strip()

            return {"detected_language": lang, "transcript_original": original, "text_english": english}
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _text_to_3d_impl(
        self,
        user_prompt: str,
        preset: str,
        seed: Optional[int],
        steps: Optional[int],
        guidance: Optional[float],
        width: int,
        height: int,
        candidates: Optional[int],
    ) -> Dict[str, Any]:
        from PIL import Image

        job_id = uuid.uuid4().hex
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        image_path = job_dir / "generated.png"
        hunyuan_image_path = job_dir / "generated_bg_removed.png"
        glb_path = job_dir / "model.glb"

        ollama_preset = os.environ.get("OLLAMA_PRESET", "3D RENDER").strip()
        pos, neg = refine_prompt_with_negative(user_prompt, preset=ollama_preset)

        pos = self._truncate_sd_77(pos)
        neg = self._truncate_sd_77(neg)

        sd_steps, sd_guidance, sd_candidates = self._sd_defaults(preset, steps, guidance, candidates)

        candidates_imgs = self._sd_text_to_images(
            prompt=pos,
            negative_prompt=neg,
            seed=seed,
            steps=sd_steps,
            guidance=sd_guidance,
            width=int(width),
            height=int(height),
            num_candidates=sd_candidates,
        )

        best_img, best_score = self._pick_best_candidate(user_prompt, candidates_imgs)
        best_img.save(image_path)

        buf = io.BytesIO()
        best_img.save(buf, format="PNG")
        cleaned_png = self.remove_bg_and_compose_white(buf.getvalue(), out_size=512, pad_ratio=0.12)

        hy_img = Image.open(io.BytesIO(cleaned_png)).convert("RGB")
        hy_img = _ensure_512_rgb(hy_img)
        hy_img.save(hunyuan_image_path)

        self._run_hunyuan_shape(hy_img, glb_path, preset=preset)
        model_vol.commit()

        return {
            "job_id": job_id,
            "status": "SUCCEEDED",
            "mode": "text-to-3d",
            "prompt_user": user_prompt,
            "prompt_pos": pos,
            "prompt_neg": neg,
            "sd": {
                "steps": sd_steps,
                "guidance": sd_guidance,
                "width": int(width),
                "height": int(height),
                "candidates": sd_candidates,
                "clip_best_score": best_score,
            },
            "artifacts": {
                "image_path": str(image_path),
                "bg_removed_path": str(hunyuan_image_path),
                "glb_path": str(glb_path),
                "image_url": f"/outputs/{job_id}/generated.png",
                "bg_removed_url": f"/outputs/{job_id}/generated_bg_removed.png",
                "glb_url": f"/download/{job_id}/model.glb",
            },
        }

    def _image_to_3d_impl(self, image_bytes: bytes, preset: str) -> Dict[str, Any]:
        from PIL import Image

        job_id = uuid.uuid4().hex
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        image_path = job_dir / "input.png"
        hunyuan_image_path = job_dir / "input_bg_removed.png"
        glb_path = job_dir / "model.glb"

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.save(image_path)

        cleaned_png = self.remove_bg_and_compose_white(image_bytes, out_size=512, pad_ratio=0.12)
        hy_img = Image.open(io.BytesIO(cleaned_png)).convert("RGB")
        hy_img = _ensure_512_rgb(hy_img)
        hy_img.save(hunyuan_image_path)

        self._run_hunyuan_shape(hy_img, glb_path, preset=preset)
        model_vol.commit()

        return {
            "job_id": job_id,
            "status": "SUCCEEDED",
            "mode": "image-to-3d",
            "artifacts": {
                "image_path": str(image_path),
                "bg_removed_path": str(hunyuan_image_path),
                "glb_path": str(glb_path),
                "image_url": f"/outputs/{job_id}/input.png",
                "bg_removed_url": f"/outputs/{job_id}/input_bg_removed.png",
                "glb_url": f"/download/{job_id}/model.glb",
            },
        }

    def _voice_to_3d_impl(self, audio_bytes: bytes, preset: str, seed: Optional[int], steps: Optional[int], guidance: Optional[float]) -> Dict[str, Any]:
        vt = self._voice_to_text_and_english(audio_bytes)
        transcript = (vt.get("transcript_original") or "").strip()
        english = (vt.get("text_english") or "").strip()

        prompt_used = (english or transcript).strip()
        if not prompt_used:
            raise RuntimeError("Empty transcript from whisper.")

        # tiny whisper fix
        prompt_used = prompt_used.replace("white back", "white background")

        out = self._text_to_3d_impl(
            user_prompt=prompt_used,
            preset=preset,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=512,
            height=512,
            candidates=None,
        )
        out["mode"] = "voice-to-3d"
        out["voice"] = {
            "detected_language": vt.get("detected_language"),
            "transcript_original": transcript,
            "text_english": english,
            "prompt_used": prompt_used,
        }
        return out

    # ---------------- Modal methods (MUST EXIST) ----------------
    @modal.method()
    def ping(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "ready": self.ready,
            "load_seconds": self.load_seconds,
            "load_error": self.load_error,
            "use_ollama_refiner": USE_OLLAMA_REFINER,
            "ollama_model": OLLAMA_MODEL,
            "ollama_pull_on_start": OLLAMA_PULL_ON_START,
            "local_files_only": os.environ.get("LOCAL_FILES_ONLY", "true"),
        }

    @modal.method()
    def text_to_3d(
        self,
        prompt: str,
        preset: str = "product",
        seed: Optional[int] = 0,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        width: int = 512,
        height: int = 512,
        candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"Server not ready. load_error:\n{self.load_error}")
        return self._text_to_3d_impl(prompt, preset, seed, steps, guidance, width, height, candidates)

    @modal.method()
    def image_to_3d(self, image_bytes: bytes, preset: str = "product") -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"Server not ready. load_error:\n{self.load_error}")
        return self._image_to_3d_impl(image_bytes, preset)

    @modal.method()
    def voice_to_3d(
        self,
        audio_bytes: bytes,
        preset: str = "product",
        seed: Optional[int] = 0,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"Server not ready. load_error:\n{self.load_error}")
        return self._voice_to_3d_impl(audio_bytes, preset, seed, steps, guidance)

# ---------------- FastAPI Web (ASGI) ----------------
@app.function(
    image=image,
    volumes={MODEL_MOUNT_PATH: model_vol},
    secrets=[R2V_SECRET],
    min_containers=1,
    max_containers=4,
    scaledown_window=60 * 10,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    api = FastAPI(title="R2V Modal AI API", version="12.1 (deadline fixed)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

    server = R2VModelServer()

    class TextTo3DReq(BaseModel):
        prompt: str
        preset: str = "product"
        seed: Optional[int] = 0
        steps: Optional[int] = None
        guidance: Optional[float] = None
        width: int = 512
        height: int = 512
        candidates: Optional[int] = None

    @api.get("/")
    def root():
        return {
            "name": "R2V Modal AI API",
            "status": "online",
            "endpoints": [
                "/health",
                "/ready",
                "/docs",
                "/text-to-3d",
                "/image-to-3d",
                "/voice-to-3d",
                "/download/{job_id}/{filename}",
                "/outputs/{job_id}/generated.png",
            ],
            "ollama": {
                "enabled": USE_OLLAMA_REFINER,
                "model": OLLAMA_MODEL,
                "preset_env": os.environ.get("OLLAMA_PRESET", "3D RENDER"),
                "strict_one_line": True,
            },
        }

    @api.get("/health")
    def health():
        return {"ok": True}

    @api.get("/ready")
    def ready():
        try:
            return server.ping.remote()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    @api.post("/text-to-3d")
    def text_to_3d(req: TextTo3DReq):
        try:
            return server.text_to_3d.remote(
                req.prompt,
                req.preset,
                req.seed,
                req.steps,
                req.guidance,
                req.width,
                req.height,
                req.candidates,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/image-to-3d")
    async def image_to_3d(file: UploadFile = File(...), preset: str = Form("product")):
        try:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="Empty file")
            return server.image_to_3d.remote(data, preset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/voice-to-3d")
    async def voice_to_3d(
        file: UploadFile = File(...),
        preset: str = Form("product"),
        seed: int = Form(0),
        steps: Optional[int] = Form(None),
        guidance: Optional[float] = Form(None),
    ):
        try:
            audio = await file.read()
            if not audio:
                raise HTTPException(status_code=400, detail="Empty audio file")
            return server.voice_to_3d.remote(audio, preset, seed, steps, guidance)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/download/{job_id}/{filename}")
    def download(job_id: str, filename: str):
        model_vol.reload() 
        file_path = OUTPUT_DIR / job_id / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        media = "application/octet-stream"
        if filename.endswith(".glb"):
            media = "model/gltf-binary"
        elif filename.endswith(".png"):
            media = "image/png"
        return FileResponse(path=str(file_path), media_type=media, filename=filename)

    return api
