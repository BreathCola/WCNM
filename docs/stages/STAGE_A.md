# Stage A — 3DGS to 2D Surfel Diffuse Foundation

## Goal

Evolve the clean 3DGS baseline from 3D ellipsoids into a diffuse 2D Gaussian
surfel representation and rasterizer producing:

```text
Cd / alpha / depth / position / normal / roughness / f0 / ks
```

The original 3DGS baseline must remain runnable. RT-GS mode must use the new
`DiffuseSurfelModel`.

## Tasks

### A-0 — Long-term project memory

- [x] Create `AGENTS.md`.
- [x] Create `docs/STATUS.md` and `docs/DECISIONS.md`.
- [x] Create `docs/stages/STAGE_A.md` through `STAGE_E.md`.
- [x] Keep root `RTGS_MASTER_PLAN.md` as the sole technical source.

### A-1 — Diffuse surfel representation

- [x] Add `scene/diffuse_surfel_model.py` with `DiffuseSurfelModel`.
- [x] Store `xyz [N,3]`, quaternion `rotation [N,4]`, `scaling_2d [N,2]`,
  `opacity_raw [N,1]`, `base_color_raw [N,3]`, `roughness_raw [N,1]`,
  `f0_raw [N,3]`, and `ks_raw [N,1]`.
- [x] Activate opacity, f0, and ks with sigmoid.
- [x] Activate roughness as `roughness_min + (1-roughness_min)*sigmoid(raw)`,
  with default `roughness_min=0.03`.
- [x] Derive tangent axes and normal from rotation plus 2D scale; normal must
  not be a geometry-independent learned vector.
- [x] Keep the original `GaussianModel` and baseline entry path runnable.

### A-2 — 2D surfel rasterization

- [x] Accept position, rotation, two tangent scales, opacity, color, and material
  attributes.
- [x] Rasterize RGB, expected depth, alpha, and surfel normal.
- [x] Composite all material maps with identical front-to-back alpha weights.
- [x] Support gradients to position, rotation, scale, opacity, base color,
  roughness, f0, and ks.
- [x] Normalize blended normals and face-forward them toward the camera.
- [x] Prefer world position reconstructed by unprojecting rasterized depth.

### A-3 — Renderer output and debug maps

- [x] Return `Cd [H,W,3]`, `alpha [H,W,1]`, `depth [H,W,1]`,
  `position [H,W,3]`, `normal [H,W,3]`, `roughness [H,W,1]`,
  `f0 [H,W,3]`, and `ks [H,W,1]`.
- [x] Save `diffuse_color.png`, `depth.png`, `normal.png`, `roughness.png`,
  `f0.png`, `ks.png`, and `alpha.png` for a fixed validation view.
- [x] Also save `ground_truth.png`; mask/overlay outputs are required only once
  the mask pipeline is introduced by its allowed stage.

### A-4 — Diffuse-only training losses

- [x] Enable baseline `L_rgb` (L1 plus D-SSIM).
- [x] Add normal-depth consistency `L_norm`, default weight `0.04`.
- [x] Add optional monocular normal `L_mono`, default weight `0.01`.
- [x] Add VGG-16 perceptual loss `L_perc`, default weight `0.01`.
- [x] Do not enable specular or transmittance depth losses.
- [x] Ensure all losses remain finite and missing optional priors are handled
  explicitly rather than silently fabricating targets.
- [x] Add an isolated, offline StableNormal preprocessing tool that emits the
  loader's float32 camera-space `.npy` contract and a provenance manifest.
- [x] Verify one real StableNormal prior through loading, camera-to-world
  conversion, positive `L_mono`, 200-step training, and checkpoint save.
- [x] Generate and strictly validate StableNormal priors for all 251 truck
  training images with no missing, duplicate, shape, dtype, or numerical errors.
- [ ] Obtain manual quality/direction approval for the fixed nine-view
  source/prior contact sheet; automated validity is not quality acceptance.

### A-5 — Training, checkpoint, and tests

- [x] RT-GS configuration selects diffuse surfels; baseline configuration still
  selects original 3DGS.
- [x] Save and restore model parameters, optimizer state, iteration, material
  parameters, and relevant Stage A configuration.
- [x] Add `tests/test_surfel_normal.py`.
- [x] Add `tests/test_material_compositing.py`.
- [x] Add `tests/test_renderer_output_contract.py`.
- [x] Add `tests/test_normal_depth_loss.py`.
- [x] Verify normal normalization and face-forward behavior.
- [x] Verify identical material alpha-compositing weights.
- [x] Verify roughness/f0/ks contain no NaN or Inf.
- [x] Verify RGB/depth/normal debug map saving.
- [x] Verify checkpoint resume.
- [x] Verify an original dataset can train in both RT-GS and baseline modes.

### A-6 — Target-video keyframe preprocessing

- [x] Add `tools/extract_video_keyframes.py` using `ffprobe`/`ffmpeg`.
- [x] Score uniformly sampled candidates for Laplacian sharpness, exposure,
  visual duplication, and temporal separation.
- [x] Preserve source resolution, emit continuous JPEG names, and record full
  selection provenance plus contact sheet and text report.
- [x] Support report-only dry-run, atomic writes, and fail-closed resume.
- [x] Add unit tests for naming, parameters, manifest, duplicate rejection, and
  dry-run image exclusion.
- [x] Verify dry-run and formal atomic extraction on a tiny synthetic FFmpeg
  video; this does not substitute for the real target video.
- [ ] Run dry-run and formal extraction on the real glass-dome target video;
  unit tests do not constitute a real extraction result.

## Explicitly forbidden in Stage A

- Reflection Gaussian or reflection model/optimizer paths.
- Transmittance Gaussian or transmittance model/optimizer paths.
- Gaussian ray tracing, brute-force production tracing, BVH, CUDA tracing, or
  OptiX.
- Transparent mesh extraction, two-hit intersection, or hit caches.
- Reflection/transmittance ray generation.
- Specular constraint or transmittance depth constraint.
- GGX, Fresnel, BRDF/BTDF weighting, or final D/R/T composition.
- Dummy later-stage outputs.

## Acceptance checklist

- [x] RT-GS configuration switches to 2D surfels.
- [x] Diffuse-only training runs.
- [x] Renderer returns every Stage A material map.
- [ ] Normal and depth visualizations are stable.
- [x] `L_norm`, `L_mono`, and `L_perc` run in their supported configurations.
- [x] No Reflection/Transmittance/ray-tracing code path exists.
- [x] Every Stage A debug image can be generated.
- [x] Checkpoints save and resume.
- [x] Original 3DGS baseline remains runnable.
- [x] Required Stage A tests pass.
- [x] `docs/STATUS.md`, this checklist, and `docs/DECISIONS.md` reflect verified
  reality.
- [ ] A rollback Git commit is created.
- [ ] Only after all boxes pass may `docs/STATUS.md` point to Stage B.

## Verified commands

```bash
git submodule update --init --recursive
conda run -n RT-GS pip install --no-build-isolation -e submodules/diff-surfel-rasterization
conda run -n RT-GS python -m pytest -q tests
conda run -n RT-GS python -m pytest -q tests/test_extract_video_keyframes.py
conda run -n RT-GS python -m pytest -q tests/test_generate_normal_priors.py tests/test_normal_prior_loading.py
conda run -n RT-GS python tools/validate_normal_priors.py \
  --scene data/tandt/truck --images images --priors normal_priors \
  --manifest manifest.json --expected-data-type indoor \
  --expected-resolution 768 --expected-steps 10

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n stablenormal \
  python tools/generate_normal_priors.py \
  --scene data/tandt/truck \
  --output data/tandt/truck/normal_priors \
  --stable-source "$HOME/opt/StableNormal" \
  --weights "$HOME/models/StableNormal/weights" \
  --dino-source "$HOME/models/StableNormal/dinov2/source" \
  --device cuda:0 --resolution 768 --steps 10 --image 000063.jpg

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n RT-GS \
  python train.py -s data/tandt/truck \
  -m output/stage_a_mono_smoke_20260628_231429 \
  --model_type surfel --normal_priors normal_priors \
  --normal_prior_space camera --lambda_mono 0.01 \
  --require_nonzero_mono --lambda_perc 0 --resolution 8 \
  --iterations 200 --test_iterations 200 --save_iterations 200 \
  --checkpoint_iterations 200 --debug_interval 200 --disable_viewer

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n stablenormal \
  python tools/generate_normal_priors.py \
  --scene data/tandt/truck \
  --output data/tandt/truck/normal_priors \
  --stable-source "$HOME/opt/StableNormal" \
  --weights "$HOME/models/StableNormal/weights" \
  --dino-source "$HOME/models/StableNormal/dinov2/source" \
  --device cuda:0 --resolution 768 --steps 10

CUDA_VISIBLE_DEVICES=0 conda run -n RT-GS python train.py \
  -s data/tandt/truck -m output/stage_a_3000 \
  --model_type surfel --resolution 8 --iterations 3000 \
  --test_iterations 3000 --save_iterations 3000 \
  --checkpoint_iterations 3000 --debug_interval 1000 --disable_viewer

CUDA_VISIBLE_DEVICES=0 conda run -n RT-GS python render.py \
  -m output/stage_a_densify --iteration 601 --skip_test --quiet

CUDA_VISIBLE_DEVICES=0 conda run -n RT-GS python train.py \
  -s data/tandt/truck -m output/baseline_smoke \
  --model_type 3dgs --resolution 8 --iterations 1 \
  --test_iterations 1 --save_iterations 1 \
  --checkpoint_iterations 1 --disable_viewer

CUDA_VISIBLE_DEVICES=0 conda run -n RT-GS python render.py \
  -m output/baseline_smoke --iteration 1 --skip_test --quiet
```

The 3,000-step run verifies the Stage A D-only schedule boundary and code paths,
but it is not a substitute for the unchecked target-resolution quality acceptance.
The real-prior smoke reported `L_mono=0.39248720` at iteration 1 and
`supervised_steps=1 nonzero_steps=1 max_L_mono=0.39248720`; one prior proves the
end-to-end path only and does not satisfy the unchecked normal/depth stability
or rollback acceptance items.
The full resumable run generated 250 new priors and skipped the one verified
existing prior. Strict validation reported 251 images, 251 priors, 251 manifest
entries, zero missing/unexpected/duplicate/abnormal artifacts, and maximum unit
length error `1.78813934e-07`. The fixed contact sheet is
`output/stage_a_normal_priors/representative_9_source_prior.png`; manual approval,
matched target-quality training, normal/depth stability, and rollback remain
unchecked.

## Superseded comparison — do not run

The previously documented Truck 30,000-step comparison was superseded on
2026-06-29 before execution. Truck is engineering smoke only. Define and run the
matched `lambda_mono=0` versus `0.01` protocol only after the glass-dome target
video, COLMAP reconstruction, and normal priors pass their own validation.
