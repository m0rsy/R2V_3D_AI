from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import TEXTURE_BACKEND

logger = logging.getLogger(__name__)


class TextureGenerationError(RuntimeError):
    """Raised when texturing is requested but cannot be produced."""


@dataclass
class TextureResult:
    texture_url: str | None
    texture_glb_path: Path | None
    warning: str | None = None


class TextureService:
    """
    Optional texture stage.

    Current backends:
    - none: no implementation (explicitly disabled)
    - vertex-color: applies average source color to mesh vertices and exports GLB
    """

    def apply_texture(
        self,
        *,
        mesh_path: Path,
        output_glb_path: Path,
        source_images: list[Image.Image],
    ) -> TextureResult:
        backend = (TEXTURE_BACKEND or "none").strip().lower()
        if backend == "none":
            raise TextureGenerationError(
                "Texturing backend is disabled (TEXTURE_BACKEND=none)."
            )
        if backend == "vertex-color":
            self._apply_vertex_color(
                mesh_path=mesh_path,
                output_glb_path=output_glb_path,
                source_images=source_images,
            )
            return TextureResult(texture_url=None, texture_glb_path=output_glb_path)
        raise TextureGenerationError(f"Unsupported texture backend: {backend}")

    @staticmethod
    def _apply_vertex_color(
        *,
        mesh_path: Path,
        output_glb_path: Path,
        source_images: list[Image.Image],
    ) -> None:
        import trimesh

        if not source_images:
            raise TextureGenerationError("Texturing requires at least one source image.")

        mesh = trimesh.load(str(mesh_path), force="mesh")
        if mesh is None:
            raise TextureGenerationError("Failed to load mesh for texturing.")

        avg_rgb = TextureService._average_rgb(source_images[0])
        vertex_count = len(getattr(mesh, "vertices", []))
        if vertex_count <= 0:
            raise TextureGenerationError("Mesh has no vertices for texturing.")

        colors = np.tile(np.array([*avg_rgb, 255], dtype=np.uint8), (vertex_count, 1))
        mesh.visual.vertex_colors = colors
        mesh.export(str(output_glb_path))

    @staticmethod
    def _average_rgb(image: Image.Image) -> tuple[int, int, int]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        mean = rgb.reshape(-1, 3).mean(axis=0)
        return int(mean[0]), int(mean[1]), int(mean[2])
