# AI 3D Backend (mo3d-1 / mo3d-pro)

## 1. Project Overview
This project is a FastAPI backend for AI-assisted 3D generation from:
- Text prompts
- Uploaded images
- Voice input (Whisper transcription + translation)

The backend is now organized as a tiered architecture:
- `mo3d-1`: baseline/free tier (current Stable Diffusion + current Hunyuan flow)
- `mo3d-pro`: quality-focused tier (Gemini image backend abstraction + pro mesh routing abstractions)

The baseline behavior is preserved under `mo3d-1` by default.

## 2. Current Architecture Overview
Primary layers:
- `app/main.py`: API entrypoints and job thread launch.
- `app/core/pipeline.py`: orchestration layer (request flow coordination + backend selection).
- `app/core/loader.py`: model pipeline loading/cache lifecycle.
- `app/services/image_generation.py`: SD/Gemini image generation routing.
- `app/services/mesh_generation.py`: Hunyuan mesh generation routing.
- `app/services/texture_service.py`: optional texturing stage abstraction.
- `app/services/bg_remove.py`, `app/services/prompt_refiner.py`, `app/services/voice_service.py`: preserved supporting services.
- `app/schemas.py`: request/response validation and API schema contracts.
- `app/config.py`: centralized runtime config, backend toggles, and defaults.

## 3. Model Tiers
### mo3d-1 (default)
- Uses current Stable Diffusion flow for text-to-image.
- Uses current Hunyuan flow for mesh generation.
- Preserves baseline behavior and output conventions.
- Can run without texturing (default), or optional texture stage if enabled.

### mo3d-pro
- Routes image generation to a Gemini abstraction path.
- If Gemini is unavailable or unsupported in runtime, and fallback is enabled, it falls back to SD.
- Routes mesh generation through a pro-capable mesh abstraction, but may fall back to basic Hunyuan when pro pipeline is not loaded.
- Supports `multi_view` request flags in service interfaces, with current limitations noted below.
- Supports optional texture stage routing with limited current backends.

## 4. Feature List
- Text -> 3D generation jobs.
- Image -> 3D generation jobs.
- Voice -> 3D generation jobs (Whisper transcription/translation).
- Prompt refinement for 3D-friendly generation.
- Background removal preprocessing.
- Tier selection with `model: "mo3d-1" | "mo3d-pro"`.
- Optional texturing with non-fatal error handling.
- Progress polling with job metadata and URLs.

## 5. Request Flow / Pipeline
Orchestrated by `app/core/pipeline.py`:
1. Receive request/job options.
2. (Voice flow) transcribe audio to prompt.
3. Refine prompt.
4. Select tier backend (`mo3d-1` or `mo3d-pro`).
5. Generate one or more images.
6. Run preprocessing/background removal.
7. Generate mesh.
8. Optionally run texture stage when `textured=true`.
9. Persist artifacts and return job metadata.

If texturing fails:
- Geometry result is preserved.
- Job can still complete with `texture_error` metadata.

Current limitation:
- The image upload endpoint currently accepts a single image; true multi-image upload for explicit multi-view input is not implemented yet.

## 6. Directory Structure
```text
app/
  config.py
  main.py
  schemas.py
  core/
    loader.py
    pipeline.py
    progress.py
  services/
    bg_remove.py
    generation.py
    image_generation.py
    mesh_generation.py
    prompt_refiner.py
    texture_service.py
    voice_service.py
outputs/
requirements.txt
README.md
```

## 7. Setup Instructions
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Ensure local model caches/paths are configured (SD and Hunyuan).
4. Optional for `mo3d-pro` Gemini path:
   - install a Gemini client package compatible with `google.genai`
   - set `GEMINI_API_KEY`.

## 8. Environment Variables / Configuration
Important variables (see `app/config.py`):
- Tier routing:
  - `DEFAULT_MODEL_TIER` (`mo3d-1` default)
  - `DEFAULT_TEXTURED`, `DEFAULT_MULTI_VIEW`, `DEFAULT_QUALITY_MODE`
- SD:
  - `SD_MODEL_ID`, `SD_STEPS_FAST`, `SD_GUIDANCE_FAST`, `SD_STEPS_PRO`, `SD_GUIDANCE_PRO`
- Hunyuan:
  - `HUNYUAN_BASIC_MODEL_ID`, `HUNYUAN_PRO_MODEL_ID`
  - `LOAD_PRO_HUNYUAN_PIPELINE`
  - `ENABLE_PRO_MULTI_IMAGE`
  - `HY_*_BASIC`, `HY_*_PRO` tuning knobs
- Gemini:
  - `GEMINI_ENABLED`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_IMAGE_COUNT`
  - `PRO_IMAGE_FALLBACK_TO_SD`
- Texture:
  - `TEXTURE_BACKEND` (`none` or `vertex-color`)
- Core:
  - `ENABLE_BG_REMOVE`, `BACKEND_BASE_URL`

## 9. Run Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

## 10. API Usage Examples
### Text -> 3D
```bash
curl -X POST http://127.0.0.1:8000/api/generate-3d \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "modern desk lamp",
    "preset": "product",
    "model": "mo3d-1",
    "textured": false,
    "multi_view": false,
    "quality_mode": "balanced"
  }'
```

### Image -> 3D
```bash
curl -X POST http://127.0.0.1:8000/api/generate-3d-from-image \
  -F "file=@/path/to/image.png" \
  -F "preset=product" \
  -F "model=mo3d-pro" \
  -F "textured=true" \
  -F "multi_view=false" \
  -F "quality_mode=quality"
```

### Voice -> 3D
```bash
curl -X POST http://127.0.0.1:8000/api/voice-to-3d \
  -F "file=@/path/to/audio.wav" \
  -F "preset=product" \
  -F "model=mo3d-1" \
  -F "textured=false"
```

### Poll Job
```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

## 11. Notes on Texturing
- Texturing is an optional stage controlled by `textured`.
- `textured=false` skips the stage entirely.
- `textured=true` runs `TextureService`.
- Current texture backends:
  - `none`: disabled/no-op with explicit error metadata.
  - `vertex-color`: lightweight fallback coloring + `model_textured.glb`.
- Current texture implementation is not a full UV/material baking pipeline.
- Texturing failures are intentionally non-fatal so mesh output can still be used.
- Geometry generation remains independent from texturing.

## 11.1 Scaffolded vs Implemented
- Fully implemented today:
  - `mo3d-1` baseline text/image/voice -> mesh flow
  - Tier routing, schema options, and optional texture stage wiring
  - Non-fatal texture error handling and job metadata reporting
- Scaffolded / partial:
  - Gemini runtime path depends on SDK/runtime response compatibility and configuration
  - Pro mesh path may fall back to basic mesh backend if `shape_pro` is not loaded
  - `multi_view` is currently a service-level abstraction, not a true multi-image upload API flow
  - Texture backend is currently limited (`none` / `vertex-color`)

## 12. Notes on Future Extensions
- Add a production-grade texture backend (UV/material baking pipeline).
- Add robust multi-image upload endpoints for explicit multi-view inputs.
- Add stronger pro mesh backend integration once finalized.
- Add task queue + persistence (Redis/Celery/DB) for production job durability.
- Add automated tests for tier routing and failure behavior.

## 13. Development Roadmap / Phases
This roadmap is tracked in the checklist below and should be updated truthfully as changes land.

## 14. Phase Completion Checklist
### Phase 1 - Audit and preserve current baseline
- [x] inspect current pipeline
- [x] identify current SD flow
- [x] identify current Hunyuan flow
- [x] map request/response schema
- [x] preserve mo3d-1 behavior

### Phase 2 - Refactor architecture
- [x] separate image generation responsibilities
- [x] separate mesh generation responsibilities
- [x] add texture service abstraction
- [x] add backend/model routing
- [x] improve config structure

### Phase 3 - API and schema updates
- [x] add model selection
- [x] add textured flag
- [x] maintain backwards compatibility
- [x] update validation

### Phase 4 - mo3d-pro integration
- [x] add Gemini backend abstraction
- [x] add multi-view / multi-image support abstraction
- [x] add pro mesh routing
- [x] add optional texture routing

### Phase 5 - Documentation and cleanup
- [x] finalize README
- [x] add docstrings/comments
- [x] improve error handling/logging
- [ ] remove dead code if safe
- [x] summarize remaining TODOs

## 15. Guidance for Future Agents/Developers
- Preserve `mo3d-1` behavior first when changing pipelines.
- Avoid mixing image and mesh logic in one file; keep service boundaries clear.
- Do not make texturing a hard dependency for geometry completion.
- Keep request defaults backward compatible (`mo3d-1`, `textured=false`).
- Prefer explicit config/env flags over hidden hardcoded behavior.
- Add integration tests before deepening `mo3d-pro` implementations.

## Agent Working Notes / Next Steps
1. Implement a concrete Gemini image backend path and verify SDK compatibility in runtime.
2. Add endpoint support for multi-image upload (true multi-view inputs).
3. Plug in finalized pro Hunyuan multi-image pipeline and enable `ENABLE_PRO_MULTI_IMAGE`.
4. Add a real texturing backend (UV/material map generation) and preserve current fallback behavior.
5. Add regression tests for:
   - baseline `mo3d-1` outputs,
   - schema backward compatibility,
   - texturing failure partial-success behavior.
