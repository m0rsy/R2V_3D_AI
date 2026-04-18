from pathlib import Path
import os

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# MODEL TIERS
# -------------------------------
MODEL_TIER_MO3D_1 = "mo3d-1"
MODEL_TIER_MO3D_PRO = "mo3d-pro"

AVAILABLE_MODEL_TIERS = (MODEL_TIER_MO3D_1, MODEL_TIER_MO3D_PRO)
DEFAULT_MODEL_TIER = os.getenv("DEFAULT_MODEL_TIER", MODEL_TIER_MO3D_1).strip().lower()
if DEFAULT_MODEL_TIER not in AVAILABLE_MODEL_TIERS:
    DEFAULT_MODEL_TIER = MODEL_TIER_MO3D_1

DEFAULT_TEXTURED = os.getenv("DEFAULT_TEXTURED", "false").lower() == "true"
DEFAULT_MULTI_VIEW = os.getenv("DEFAULT_MULTI_VIEW", "false").lower() == "true"
DEFAULT_QUALITY_MODE = os.getenv("DEFAULT_QUALITY_MODE", "quality").strip().lower()
if DEFAULT_QUALITY_MODE not in {"fast", "balanced", "quality"}:
    DEFAULT_QUALITY_MODE = "quality"

# -------------------------------
# MODELS (LOCAL PATHS / HF IDs)
# -------------------------------
SD_MODEL_ID = os.getenv(
    "SD_MODEL_ID",
    r"D:\Grademodels\model_cache\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)

# Hunyuan: use your local offline snapshot if available, else HF ID
HUNYUAN_MODEL_ID = os.getenv(
    "HUNYUAN_MODEL_ID",
    "tencent/Hunyuan3D-2"
)

# Alias for clarity in tier routing (kept same as baseline by default)
HUNYUAN_BASIC_MODEL_ID = os.getenv("HUNYUAN_BASIC_MODEL_ID", HUNYUAN_MODEL_ID)
HUNYUAN_PRO_MODEL_ID = os.getenv("HUNYUAN_PRO_MODEL_ID", HUNYUAN_MODEL_ID)
LOAD_PRO_HUNYUAN_PIPELINE = os.getenv("LOAD_PRO_HUNYUAN_PIPELINE", "false").lower() == "true"
ENABLE_PRO_MULTI_IMAGE = os.getenv("ENABLE_PRO_MULTI_IMAGE", "false").lower() == "true"

# Gemini (mo3d-pro image backend)
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image-preview")
GEMINI_IMAGE_COUNT = int(os.getenv("GEMINI_IMAGE_COUNT", "4"))
PRO_IMAGE_FALLBACK_TO_SD = os.getenv("PRO_IMAGE_FALLBACK_TO_SD", "true").lower() == "true"

# -------------------------------
# SD FAST SETTINGS
# -------------------------------
SD_WIDTH = int(os.getenv("SD_WIDTH", "512"))
SD_HEIGHT = int(os.getenv("SD_HEIGHT", "512"))
SD_STEPS_FAST = int(os.getenv("SD_STEPS_FAST", "15"))
SD_GUIDANCE_FAST = float(os.getenv("SD_GUIDANCE_FAST", "7.0"))

SDXL_TURBO_STEPS_FAST = int(os.getenv("SDXL_TURBO_STEPS_FAST", "3"))
SDXL_TURBO_GUIDANCE_FAST = float(os.getenv("SDXL_TURBO_GUIDANCE_FAST", "0.0"))

# Pro profile defaults (quality-first)
SD_STEPS_PRO = int(os.getenv("SD_STEPS_PRO", "28"))
SD_GUIDANCE_PRO = float(os.getenv("SD_GUIDANCE_PRO", "7.5"))

# Hunyuan defaults (mo3d-1 baseline preserved)
HY_STEPS_BASIC = int(os.getenv("HY_STEPS_BASIC", "8"))
HY_GUIDANCE_BASIC = float(os.getenv("HY_GUIDANCE_BASIC", "3.0"))
HY_OCTREE_BASIC = int(os.getenv("HY_OCTREE_BASIC", "128"))
HY_CHUNKS_BASIC = int(os.getenv("HY_CHUNKS_BASIC", "200"))

# Hunyuan pro defaults (quality-first)
HY_STEPS_PRO = int(os.getenv("HY_STEPS_PRO", "20"))
HY_GUIDANCE_PRO = float(os.getenv("HY_GUIDANCE_PRO", "4.5"))
HY_OCTREE_PRO = int(os.getenv("HY_OCTREE_PRO", "256"))
HY_CHUNKS_PRO = int(os.getenv("HY_CHUNKS_PRO", "256"))

# -------------------------------
# WHISPER (VOICE)
# -------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# -------------------------------
# BACKEND
# -------------------------------
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

# -------------------------------
# FEATURE FLAGS
# -------------------------------
# Set to "false" to skip bg removal (e.g. during testing)
ENABLE_BG_REMOVE = os.getenv("ENABLE_BG_REMOVE", "true").lower() == "true"

# Set to "false" to skip SD and go straight to Hunyuan from image
ENABLE_SD = os.getenv("ENABLE_SD", "true").lower() == "true"

# Texture backend:
# - "none": no texturing implementation (safe no-op)
# - "vertex-color": lightweight fallback using average image color on mesh vertices
TEXTURE_BACKEND = os.getenv("TEXTURE_BACKEND", "none").strip().lower()
