from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from rembg import remove as rembg_remove, new_session


@dataclass
class BgRemoveResult:
    rgba_png_bytes: bytes           # RGBA transparent PNG
    composited_rgb_png_bytes: bytes # Composited on pure white, RGB PNG


# One shared rembg session (u2netp is fast and good enough for product images)
_REMBG_SESSION = new_session("u2netp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_light_uniform_bg(rgb: np.ndarray, thr_mean: int = 235, thr_std: float = 18.0) -> bool:
    """
    Heuristic: if the image border is bright and fairly uniform,
    it's likely a white/off-white studio background.
    Fast path is safe to use for these images.
    """
    h, w = rgb.shape[:2]
    b = 12  # slightly thicker border for more reliable sampling
    border = np.concatenate([
        rgb[:b,  :,  :].reshape(-1, 3),
        rgb[-b:, :,  :].reshape(-1, 3),
        rgb[:,  :b,  :].reshape(-1, 3),
        rgb[:, -b:,  :].reshape(-1, 3),
    ], axis=0)
    m = border.mean(axis=0)
    s = border.std(axis=0).mean()
    return bool((m.min() >= thr_mean) and (s <= thr_std))


def _estimate_bg_color(rgb: np.ndarray) -> np.ndarray:
    """Median border colour — robust against small logos or edge artefacts."""
    h, w = rgb.shape[:2]
    b = 12
    border = np.concatenate([
        rgb[:b,  :,  :].reshape(-1, 3),
        rgb[-b:, :,  :].reshape(-1, 3),
        rgb[:,  :b,  :].reshape(-1, 3),
        rgb[:, -b:,  :].reshape(-1, 3),
    ], axis=0)
    return np.median(border, axis=0)


def _floodfill_bg_mask(
    rgb: np.ndarray,
    bg: np.ndarray,
    tol: int = 55,
) -> np.ndarray:
    """
    BFS flood-fill from all border pixels.
    Any pixel close to `bg` colour (within `tol`) and reachable from the border
    is considered background (including soft shadows).
    Returns uint8 mask: 1 = background, 0 = foreground.
    """
    h, w = rgb.shape[:2]
    visited = np.zeros((h, w), dtype=np.uint8)
    mask    = np.zeros((h, w), dtype=np.uint8)
    bg16    = bg.astype(np.int16)

    def _close(i: int, j: int) -> bool:
        return int(np.max(np.abs(rgb[i, j].astype(np.int16) - bg16))) <= tol

    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        q.append((0,     x))
        q.append((h - 1, x))
    for y in range(h):
        q.append((y, 0))
        q.append((y, w - 1))

    while q:
        i, j = q.popleft()
        if visited[i, j]:
            continue
        visited[i, j] = 1
        if not _close(i, j):
            continue
        mask[i, j] = 1
        if i > 0:     q.append((i - 1, j))
        if i < h - 1: q.append((i + 1, j))
        if j > 0:     q.append((i, j - 1))
        if j < w - 1: q.append((i, j + 1))

    return mask


def _pad_bbox(
    x0: int, y0: int, x1: int, y1: int,
    w: int,  h: int,
    pad_px: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, x0 - pad_px),
        max(0, y0 - pad_px),
        min(w, x1 + pad_px),
        min(h, y1 + pad_px),
    )


def _place_on_square_and_resize(
    rgb_img: Image.Image,
    out_size: int,
    pad_ratio: float,
) -> Image.Image:
    """
    1. Crop tightly to non-white foreground.
    2. Add proportional padding.
    3. Place on a square white canvas.
    4. Resize to out_size × out_size (LANCZOS).
    """
    rgb = np.array(rgb_img)
    fg = np.where(np.max(255 - rgb, axis=2) > 3)

    if fg[0].size == 0:
        # Nothing visible — return blank
        return rgb_img.resize((out_size, out_size), Image.LANCZOS)

    y0, y1 = int(fg[0].min()), int(fg[0].max()) + 1
    x0, x1 = int(fg[1].min()), int(fg[1].max()) + 1
    h, w   = rgb.shape[:2]

    bw, bh = x1 - x0, y1 - y0
    pad_px = int(max(bw, bh) * pad_ratio)
    x0, y0, x1, y1 = _pad_bbox(x0, y0, x1, y1, w, h, pad_px)

    crop = rgb_img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    side = max(max(cw, ch), 8)

    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas.resize((out_size, out_size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_bg_and_compose_white(
    img_bytes: bytes,
    pad_ratio: float = 0.12,
    out_size: int = 512,
    # Fast-path knobs
    bg_tol: int = 55,
    edge_soften_radius: float = 1.5,
    # Optional post-enhancement (off by default — avoids crunchy edges)
    enhance: bool = False,
    enhance_contrast: float = 1.01,
    enhance_sharp: float = 1.02,
) -> BgRemoveResult:
    """
    Two-strategy background removal:

    FAST PATH  (white / off-white studio backgrounds):
        • Flood-fill background mask from border pixels.
        • Soft-blend background → pure white (Gaussian edge feathering).
        • Crop + pad + resize to stable 512×512.
        Ideal for product photos, SD-generated images.

    FALLBACK  (complex / mixed backgrounds):
        • rembg u2netp with alpha matting.
        • Composite onto white.
        • Same crop+pad+resize.

    Returns BgRemoveResult with both RGBA and white-composited RGB bytes.
    """
    orig = Image.open(BytesIO(img_bytes)).convert("RGB")
    rgb  = np.array(orig)

    # ------------------------------------------------------------------ FAST
    if _is_light_uniform_bg(rgb):
        bg       = _estimate_bg_color(rgb)
        bg_mask  = _floodfill_bg_mask(rgb, bg=bg, tol=int(bg_tol))

        # Soft-blend to white (anti-aliased edges, no jaggies)
        mask_img  = Image.fromarray((bg_mask * 255).astype(np.uint8), mode="L")
        mask_soft = mask_img.filter(ImageFilter.GaussianBlur(radius=float(edge_soften_radius)))
        alpha     = (np.array(mask_soft).astype(np.float32) / 255.0)[..., None]

        white   = np.full_like(rgb, 255, dtype=np.uint8)
        blended = (
            rgb.astype(np.float32) * (1.0 - alpha)
            + white.astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)

        cleaned = Image.fromarray(blended, mode="RGB")

        if enhance:
            cleaned = ImageEnhance.Contrast(cleaned).enhance(float(enhance_contrast))
            cleaned = ImageEnhance.Sharpness(cleaned).enhance(float(enhance_sharp))

        final_rgb = _place_on_square_and_resize(cleaned, out_size=out_size, pad_ratio=pad_ratio)

        rgba     = final_rgb.convert("RGBA")
        buf_rgba = BytesIO(); rgba.save(buf_rgba, format="PNG")
        buf_rgb  = BytesIO(); final_rgb.save(buf_rgb, format="PNG")

        return BgRemoveResult(
            rgba_png_bytes=buf_rgba.getvalue(),
            composited_rgb_png_bytes=buf_rgb.getvalue(),
        )

    # -------------------------------------------------------------- FALLBACK
    rgba_bytes = rembg_remove(
        img_bytes,
        session=_REMBG_SESSION,
        alpha_matting=True,
        alpha_matting_foreground_threshold=245,
        alpha_matting_background_threshold=8,
        alpha_matting_erode_size=10,
    )
    rgba = Image.open(BytesIO(rgba_bytes)).convert("RGBA")

    white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
    comp = Image.alpha_composite(white_bg.convert("RGBA"), rgba).convert("RGB")

    if enhance:
        comp = ImageEnhance.Contrast(comp).enhance(float(enhance_contrast))
        comp = ImageEnhance.Sharpness(comp).enhance(float(enhance_sharp))

    comp = _place_on_square_and_resize(comp, out_size=out_size, pad_ratio=pad_ratio)

    rgba2    = comp.convert("RGBA")
    buf_rgba = BytesIO(); rgba2.save(buf_rgba, format="PNG")
    buf_rgb  = BytesIO(); comp.save(buf_rgb,   format="PNG")

    return BgRemoveResult(
        rgba_png_bytes=buf_rgba.getvalue(),
        composited_rgb_png_bytes=buf_rgb.getvalue(),
    )