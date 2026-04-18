from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field

import torch
from PIL import Image

from app.config import (
    ENABLE_PRO_MULTI_IMAGE,
    HY_CHUNKS_BASIC,
    HY_CHUNKS_PRO,
    HY_GUIDANCE_BASIC,
    HY_GUIDANCE_PRO,
    HY_OCTREE_BASIC,
    HY_OCTREE_PRO,
    HY_STEPS_BASIC,
    HY_STEPS_PRO,
    MODEL_TIER_MO3D_1,
    MODEL_TIER_MO3D_PRO,
)

logger = logging.getLogger(__name__)


class MeshGenerationError(RuntimeError):
    """Raised when mesh generation fails."""


@dataclass
class MeshGenerationResult:
    mesh: object
    backend_used: str
    warnings: list[str] = field(default_factory=list)


class MeshGenerationService:
    """Routes image-to-mesh generation to tier-specific Hunyuan backends."""

    def generate_mesh(
        self,
        *,
        images: list[Image.Image],
        model_tier: str,
        multi_view: bool,
        use_cuda: bool,
    ) -> MeshGenerationResult:
        if not images:
            raise MeshGenerationError("No images were provided for mesh generation.")

        tier = (model_tier or MODEL_TIER_MO3D_1).strip().lower()
        warnings: list[str] = []
        shape_pipe, backend_used = self._select_backend(tier=tier)

        if shape_pipe is None:
            raise MeshGenerationError("Hunyuan pipeline is not available.")

        params = self._tier_params(tier)
        kwargs = dict(
            num_inference_steps=params["steps"],
            guidance_scale=params["guidance"],
            octree_resolution=params["octree"],
            num_chunks=params["chunks"],
            enable_pbar=False,
            output_type="trimesh",
        )

        if (
            tier == MODEL_TIER_MO3D_PRO
            and multi_view
            and len(images) > 1
            and ENABLE_PRO_MULTI_IMAGE
            and self._supports_multi_image(shape_pipe)
        ):
            kwargs["images"] = images
        else:
            if tier == MODEL_TIER_MO3D_PRO and multi_view and len(images) > 1:
                warnings.append(
                    "Pro multi-image mesh path is not enabled/supported; using the first image."
                )
            kwargs["image"] = images[0]

        if use_cuda:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                shape_result = shape_pipe(**kwargs)
        else:
            with torch.inference_mode():
                shape_result = shape_pipe(**kwargs)

        mesh = self._extract_mesh(shape_result)
        if mesh is None:
            raise MeshGenerationError("Mesh is None after Hunyuan extraction.")

        mesh = self._smooth_mesh(mesh, tier=tier)
        return MeshGenerationResult(mesh=mesh, backend_used=backend_used, warnings=warnings)

    def _select_backend(self, *, tier: str):
        from app.core.pipeline import get_pipelines

        pipes = get_pipelines()
        if tier == MODEL_TIER_MO3D_PRO and pipes.get("shape_pro") is not None:
            return pipes["shape_pro"], "hunyuan-pro"
        return pipes.get("shape"), "hunyuan-basic"

    @staticmethod
    def _tier_params(tier: str) -> dict[str, float | int]:
        if tier == MODEL_TIER_MO3D_PRO:
            return {
                "steps": HY_STEPS_PRO,
                "guidance": HY_GUIDANCE_PRO,
                "octree": HY_OCTREE_PRO,
                "chunks": HY_CHUNKS_PRO,
            }
        return {
            "steps": HY_STEPS_BASIC,
            "guidance": HY_GUIDANCE_BASIC,
            "octree": HY_OCTREE_BASIC,
            "chunks": HY_CHUNKS_BASIC,
        }

    @staticmethod
    def _supports_multi_image(shape_pipe: object) -> bool:
        try:
            sig = inspect.signature(shape_pipe.__call__)
            return "images" in sig.parameters
        except Exception:
            return False

    @staticmethod
    def _extract_mesh(shape_result):
        if shape_result is None:
            raise MeshGenerationError("Hunyuan returned None.")
        if isinstance(shape_result, (list, tuple)):
            return shape_result[0]
        if isinstance(shape_result, dict) and "mesh" in shape_result:
            return shape_result["mesh"]
        if isinstance(shape_result, dict) and len(shape_result) > 0:
            return next(iter(shape_result.values()))
        return shape_result

    @staticmethod
    def _smooth_mesh(mesh, *, tier: str):
        import trimesh

        iterations = 6 if tier == MODEL_TIER_MO3D_PRO else 3
        lamb = 0.35 if tier == MODEL_TIER_MO3D_PRO else 0.3

        try:
            smoothed = trimesh.smoothing.filter_laplacian(
                mesh,
                lamb=lamb,
                iterations=iterations,
                implicit_time_integration=False,
                volume_constraint=True,
            )
            return smoothed
        except Exception as exc:
            logger.warning("Mesh smoothing failed, using raw mesh: %s", exc)
            return mesh
