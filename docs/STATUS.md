# RT-GS Status

Current stage: Stage A — 3DGS to 2D Surfel Diffuse Foundation

Stage state: the TiHuBird StableNormal D-only baseline is conditionally accepted
and ready to be frozen on `baseline/stage-a-stablenormal`. Overall Stage A
closure is deferred while DiffusionRenderer priors are evaluated independently;
Stage B/C/D are not authorized.

Evidence code commit: `829dd82dc74f4c5dce640201448626df24afa25a`

## Repository and scene

- The original 3DGS baseline remains available through `--model_type 3dgs`.
- Stage A RT-GS uses `--model_type surfel` and the pinned 2D surfel rasterizer.
- Diffuse, Reflection, and Transmittance remain separate by design; only the
  Diffuse branch exists in Stage A.
- `data/TiHuBird` is the real acceptance scene. Truck artifacts are historical
  engineering smoke evidence only.
- TiHuBird has 111 final undistorted PINHOLE images at 3827x2152, 111/111 COLMAP
  registrations, and 84,989 sparse points.

## StableNormal baseline evidence

- Generated all 111 TiHuBird StableNormal priors in
  `data/TiHuBird/normal_priors/` and passed strict validation: 111 images, 111
  priors, 111 manifest entries, with empty missing, unexpected, duplicate, and
  abnormal lists. Evidence:
  `output/stage_a_tihubird_priors_validate_111.log`.
- Completed the full-prior 1,000-step smoke at
  `output/stage_a_tihubird_fullprior_1k_r2`. It reported 1,000 supervised and
  1,000 nonzero monocular-normal steps, saved the iteration-1,000 checkpoint,
  and printed `Training complete.`
- Completed the Gate 6 input freeze. The baseline and accepted retry records are
  byte-identical and pin code HEAD `829dd82`, `PYTHONHASHSEED=0`, 111 images,
  111 priors, the image-tree hash, prior-manifest hash, and all three COLMAP
  hashes.
- Gate 7 completed from iteration 0 to 30,000 with `lambda_mono=0` at
  `output/stage_a_tihubird_30k_mono0_r2`.
- The first Gate 8 attempt at
  `output/stage_a_tihubird_30k_mono001_r2` stopped around iteration 230 and has
  no usable checkpoint. It is excluded from comparison.
- Gate 8 retry1 restarted from iteration 0 and completed at
  `output/stage_a_tihubird_30k_mono001_r2_retry1`. Only retry1 is the accepted
  `lambda_mono=0.01` result.
- Gate 9 matched metrics and the global comparison are present at
  `output/stage_a_tihubird_30k_matched_metrics.txt` and
  `output/stage_a_tihubird_30k_matched_comparison.png`.
- Gate 9.5 completed the six-view and crop audit under
  `output/stage_a_tihubird_gate9p5/`.

## Matched 30,000-step metrics

| Iteration | `lambda_mono=0` L1 | `lambda_mono=0` PSNR | `lambda_mono=0.01` retry1 L1 | `lambda_mono=0.01` retry1 PSNR |
|---:|---:|---:|---:|---:|
| 7,000 | 0.0233018197119236 | 25.840939331054688 | 0.022160319611430168 | 26.222691726684573 |
| 15,000 | 0.022149086557328702 | 26.27469940185547 | 0.020849463716149333 | 26.722418594360352 |
| 30,000 | 0.01844301298260689 | 27.603187561035156 | 0.017842570878565313 | 27.78533058166504 |

## Baseline decision

- Conditionally accept the StableNormal result as the Stage A D-only baseline.
- Select `lambda_mono=0.01` for that baseline. Its retry1 result improves both
  L1 and PSNR over `lambda_mono=0` at all three matched evaluation nodes, and
  the global, six-view, and crop audits show no obvious RGB degradation.
- This is not final transparent-scene reconstruction. The D-only representation
  still mixes the glass surface, reflection, interior bird, and background.
  Its normal/depth are a foundation for later separation, not final transparent
  geometry or a final bird reconstruction.

## DiffusionRenderer prior extension

- DR-1A used the final undistorted TiHuBird training images for a 24-frame raw
  pilot at `output/stage_a_tihubird_dr_pilot_24/`.
- DR-1B generated raw priors from all 111 final training images at
  `output/stage_a_tihubird_dr_raw_111/`.
- Raw outputs contain `normal`, `depth`, `basecolor`, and `diffuse_albedo`.
  The 111 real-frame mappings and the final chunk's nine padding slots are
  recorded explicitly.
- No DiffusionRenderer normal adapter or normal-axis mapping has been selected
  or validated. No DiffusionRenderer prior loss has been connected, no training
  has used these priors, and the StableNormal baseline has not been replaced.
- `basecolor` and `diffuse_albedo` remain audit-only and are not training
  supervision.

## Tests and verified behavior

- `conda run -n RT-GS python -m pytest -q tests`: 29 passed at the Stage A code
  evidence commit.
- Original 3DGS and surfel training/render paths, checkpoint resume, PLY reload,
  densification, material gradients, and Stage A debug exports have passed their
  documented checks.
- Both accepted 30k runs saved checkpoints at 10k/20k/30k and PLY outputs at
  7k/15k/30k.

## Current boundary and next task

- Remain in Stage A. Stage B/C/D are forbidden without separate authorization.
- Freeze this StableNormal D-only baseline as a rollback branch and conduct the
  DiffusionRenderer normal adapter and axis audit only on the independent
  `feature/stage-a-diffrender-priors` branch.
- Do not overwrite `data/TiHuBird/normal_priors/` or either accepted StableNormal
  30k output directory.
- Final Stage A closure remains deferred until the DiffusionRenderer experiment
  is evaluated and an explicit closure decision is made.
