from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RefinedPrompt:
    positive: str
    negative: str


# ---------------------------------------------------------------------------
# Negative prompt — keyword style works best with SD 1.5
# ---------------------------------------------------------------------------
BASE_NEGATIVE = (
    "multiple objects, two objects, extra items, accessories, props, packaging, box, "
    "scene, environment, room, interior, outdoor, background objects, "
    "table, desk, floor, wall, surface, tabletop, wood texture, fabric texture, background texture, "
    "flat lay, top-down, overhead view, close-up, macro, zoomed in, "
    "stand, mount, pole, tripod, pedestal, platform, holder, "
    "people, person, hands, holding, arms, fingers, "
    "text, watermark, logo, label, brand text, "
    "cropped, cut off, out of frame, partial view, occlusion, "
    "harsh shadow, strong shadows, long shadow, dramatic lighting, rim light, "
    "glow, bloom, lens flare, reflections of environment, "
    "blurry, low quality, lowres, noisy, jpeg artifacts, "
    "bad anatomy, deformed, distorted"
)

# ---------------------------------------------------------------------------
# Quality tags per preset
# ---------------------------------------------------------------------------
_QUALITY_TAGS: dict[str, str] = {
    "QUALITY":   "ultra detailed, photorealistic, 8k, high resolution, crisp edges, sharp focus, clean geometry, fine surface detail",
    "FAST":      "photorealistic, high quality, sharp focus, clean geometry, clean edges",
    "product":   "photorealistic, high quality, sharp focus, clean product photo, clean geometry",
    "studio":    "studio photo, highly detailed, professional lighting, sharp focus",
    "photoreal": "hyperrealistic, 8k, photo, sharp focus, ultra detailed, lifelike",
}


def refine_prompt_for_3d_sd(user_prompt: str, preset: str = "product") -> RefinedPrompt:
    """
    Harden a user prompt for SD 1.5 → 3D pipeline.

    Strategy:
      • Keep prompt compact and keyword-heavy (SD 1.5 obeys this better than sentences).
      • Always: isolated single object, white bg, full frame, 3/4 view.
      • preset controls quality suffix.
    """
    up = (user_prompt or "").strip().rstrip(".,")
    if not up:
        up = "a single object"

    preset_key = (preset or "product").strip()
    quality = _QUALITY_TAGS.get(preset_key, _QUALITY_TAGS["product"])

    positive = (
        f"{up}, "
        "single object only, isolated object, centered composition, full object visible in frame, "
        "zoomed out with margin, three-quarter view angle, "
        "pure white seamless background, studio product photography, "
        "soft even studio lighting, minimal contact shadow only, "
        "no floor reflection, no wall, no table, no environment, "
        "clean sharp silhouette, high contrast edges, "
        "3D reconstruction friendly, suitable for photogrammetry, "
        f"{quality}"
    )

    return RefinedPrompt(positive=positive, negative=BASE_NEGATIVE)