# RT-GS Status

Current stage: Stage A — 3DGS to 2D Surfel Diffuse Foundation

Current branch: `master`

Last verified commit: `93d962371887801764332ddfc70d66680ceedd31`

## Repository structure

- Clean baseline 3DGS Python training/rendering entry points are present.
- Baseline Gaussian model: `scene/gaussian_model.py`.
- Baseline renderer: `gaussian_renderer/`.
- Baseline CUDA rasterizer remains `diff-gaussian-rasterization`.
- Stage A true 2D rasterizer is pinned as the recursively initialized
  `submodules/diff-surfel-rasterization` submodule.
- RT-GS mode is selected explicitly with `--model_type surfel`; default
  `--model_type 3dgs` preserves baseline behavior.
- Root `RTGS_MASTER_PLAN.md` is the sole technical specification.
- Stage memory files were initialized as the first Stage A task.

## Completed

- Read the complete root master plan and initialized all required memory files.
- Implemented `DiffuseSurfelModel` with 2D scales, rotation-derived normals,
  bounded diffuse/material parameters, optimizer/densification, PLY I/O, and
  versioned checkpoint state.
- Integrated the perspective-correct 2D surfel CUDA rasterizer.
- Implemented the HWC `Cd/alpha/depth/position/normal/roughness/f0/ks` contract,
  shared material alpha weights, expected-depth unprojection, normal
  normalization, and face-forward orientation.
- Added diffuse-only RGB, normal-depth, optional monocular-normal, and frozen
  VGG-16 perceptual losses.
- Added `.npy` normal-prior loading, validity masks, camera/world coordinate
  selection, and normalization.
- Added a dedicated-environment, fail-closed offline StableNormal generator with
  atomic HWC float32 priors, resumable validation, and `manifest.json` provenance.
- Added a default-off smoke-test guard that fails when no positive finite
  `L_mono` is observed, without changing normal training behavior.
- Generated and validated a real offline StableNormal prior for deterministic
  first-view `000063.jpg`: `000063.npy` is HWC `(546,979,3)`, camera-space
  float32, finite, and unit-normalized. Generation used indoor mode, resolution
  768, and 10 denoising steps.
- Verified the real prior end to end in a 200-step surfel smoke run. Iteration 1
  reported `L_mono=0.39248720`; the summary reported one supervised/nonzero
  step and the run saved `chkpnt200.pth` before exiting successfully.
- Completed offline StableNormal generation for all 251 truck images with the
  same indoor/resolution-768/10-step camera-space configuration. The resumable
  run generated 250 files and validated/skipped the existing `000063.npy`.
- Strictly re-read all priors and verified 251 images = 251 `.npy` files = 251
  manifest entries, with no missing, unexpected, duplicate, non-finite,
  wrong-dtype, wrong-shape, near-zero, or non-unit artifacts.
- Exported a fixed nine-view source/prior contact sheet spanning the sequence.
  Preliminary inspection shows structured, non-flat normals and visible
  high-frequency noise in foliage/thin structures; user quality approval remains
  pending and normal/depth stability is not accepted.
- Added fixed-view Stage A debug export and full-G-buffer export from `render.py`.
- Verified checkpoint save/resume, PLY reload/render, material/geometry gradients,
  densification, and the original 3DGS train/render paths.
- Ran a 3,000-step low-resolution D-only training pass on `data/tandt/truck`.

## Tests passed

- `conda run -n RT-GS python -m pytest -q tests`: 16 passed.
- Python compile check and `git diff --check` passed.
- Surfel one-step training, checkpoint resume, 601-step densification smoke,
  3,000-step D-only training, and 251-view G-buffer rendering passed.
- Original 3DGS one-step training and 251-view rendering passed.

## Known failures

- Full truck prior coverage is complete, but the nine-view contact sheet still
  requires user review and no matched target-quality training comparison has
  established whether the priors improve normal/depth stability.
- The 3,000-step run used `--resolution 8`; final-resolution/30,000-step quality
  acceptance has not run. The final normal map is finite and scene-aligned but
  retains visible high-frequency noise.
- No rollback commit has been created. Stage A must remain current until the
  remaining quality acceptance and commit are complete.

## Current metrics

- Dataset: `data/tandt/truck`, 3,000 iterations, resolution divisor 8.
- Validation train-view PSNR: 25.6296 dB; L1: 0.03405.
- Total training loss: 0.38218 at step 1 to 0.07305 at step 3,000.
- Normal-depth loss: 0.79331 to 0.25569; all logged values finite.
- VGG perceptual loss: 1.27682 to 0.51705; all logged values finite.
- Surfel count: 136,029 initial to 211,880 at step 3,000.
- Debug output: `output/stage_a_3000/debug/iteration_003000/`.
- Real-prior smoke: `output/stage_a_mono_smoke_20260628_231429`, 200 iterations,
  resolution divisor 8, `L_mono=0.39248720` on `000063.jpg`, one supervised and
  one nonzero monocular-normal step, checkpoint saved, exit code 0.
- Prior set: 251 files, 1.6 GiB, HWC camera-space float32, indoor, resolution
  768, 10 steps. Strict validation: 0 missing/unexpected/duplicates/abnormal;
  max unit-length error `1.78813934e-07`; component range
  `[-0.999984622, 0.999984622]`.
- Full resumable generation produced 250 new files in about 247 seconds including
  model load; one existing valid file was skipped.
- Prior debug outputs: `output/stage_a_normal_priors/generate.log`,
  `validate.log`, and `representative_9_source_prior.png`.

## Next exact task

- Obtain user review of the fixed nine-view source/prior contact sheet, then run
  the documented matched 30,000-step full-resolution Stage A comparison with
  `lambda_mono=0` versus `0.01`; do not change any other training setting.

## Blocked by

- Manual prior-direction/quality review and the matched `lambda_mono=0` versus
  `0.01` target-quality comparison remain intentionally unrun. Final
  normal/depth acceptance and rollback commit also remain incomplete.
