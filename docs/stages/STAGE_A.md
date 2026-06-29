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
  training images with no missing, duplicate, shape, dtype, or numerical errors;
  this is historical engineering evidence, not target-scene acceptance.
- [x] Generate, strictly validate, and visually approve the TiHuBird single prior
  `000000.npy`; verify the mono chain in the corrected 200-step smoke.
- [x] Generate and strictly validate all 111 TiHuBird priors.
- [x] Obtain manual quality/direction approval for the fixed TiHuBird nine-view
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
- [x] Verify the supplied formal TiHuBird extraction: 111 full-resolution
  keyframes, complete non-dry-run manifest, contiguous unique names, and exact
  manifest-to-`images_raw/` correspondence. This verifies the artifact and does
  not claim that this agent ran its extraction command.

### A-7 — TiHuBird COLMAP/SfM preparation

- [x] Confirm the actual loader requires `images/` plus `sparse/0/` and accepts
  only final SIMPLE_PINHOLE or PINHOLE cameras.
- [x] Preserve all 111 fixed `images_raw/` files byte-for-byte and use one shared
  camera for the single-device video.
- [x] Reuse the repository COLMAP flow in an isolated output workspace: OPENCV
  feature extraction, exhaustive guided matching, mapping, then full-resolution
  image undistortion.
- [x] Register 111/111 images in one model with 84,989 sparse points and one
  final PINHOLE camera at 3827x2152.
- [x] Verify final image names, `images.bin` names, and manifest names match
  exactly; verify every final image is decodable at the final camera dimensions.
- [x] Run `colmap model_analyzer`, finite point/pose checks, trajectory continuity
  diagnostics, and an isolated read-only RT-GS loader check.
- [x] Save complete commands, logs, validation reports, the pre-undistortion
  image backup, and trajectory diagnostic under
  `output/stage_a_tihubird_colmap/`.
- [x] Do not generate target priors or start training during SfM preparation.

### A-8 — Historical TiHuBird Stage A Operator Runbook

TiHuBird is the real acceptance scene; Truck is historical smoke only. The
commands below preserve the original ordered operator plan and must not be
rerun as if they were still pending. Actual execution is recorded in A-9: Gates
1–7 completed as planned; the first Gate 8 attempt stopped around iteration 230
without a checkpoint, so retry1 restarted from iteration 0 and is the only
accepted mono result; Gates 9 and 9.5 then completed. Final overall Stage A
closure is deferred while the independent DiffusionRenderer Stage A prior
extension is evaluated.

#### Gate 1 — Generate the remaining 110 priors

- Prerequisite: validated `000000.npy` and its manifest entry exist; Step 1 log
  does not exist; at least 12 GiB working space is available.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS &&
  set -euo pipefail
  mkdir -p output
  LOG="output/stage_a_tihubird_priors_remaining_110.log"
  PRIOR="data/TiHuBird/normal_priors/000000.npy"
  MANIFEST="data/TiHuBird/normal_priors/manifest.json"
  test ! -e "$LOG"
  test -s "$PRIOR"
  test -s "$MANIFEST"
  SHA256_BEFORE="$(sha256sum "$PRIOR" | cut -d ' ' -f1)"
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    NO_PROXY='*' no_proxy='*' \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n stablenormal \
    python tools/generate_normal_priors.py \
    --scene data/TiHuBird --images images \
    --output data/TiHuBird/normal_priors \
    --stable-source "$HOME/opt/StableNormal" \
    --weights "$HOME/models/StableNormal/weights" \
    --dino-source "$HOME/models/StableNormal/dinov2/source" \
    --torch-home "$HOME/.cache/stablenormal/torch" \
    --device cuda:0 --resolution 768 --steps 10 --seed 0 \
    2>&1 | sed -E 's/^\[[0-9]+\/[0-9]+\] WROTE /WROTE /' | tee "$LOG"
  SHA256_AFTER="$(sha256sum "$PRIOR" | cut -d ' ' -f1)"
  SKIPPED_EXISTING="$(grep -c '^SKIP verified ' "$LOG" || true)"
  NEWLY_WRITTEN="$(grep -c '^WROTE ' "$LOG" || true)"
  NPY_COUNT="$(find data/TiHuBird/normal_priors -maxdepth 1 -type f -name '*.npy' | wc -l)"
  MANIFEST_COUNT="$(grep -c '"output_file":' "$MANIFEST")"
  {
    echo "sha256_before=$SHA256_BEFORE"
    echo "sha256_after=$SHA256_AFTER"
    echo "skipped_existing=$SKIPPED_EXISTING"
    echo "newly_written=$NEWLY_WRITTEN"
    echo "npy_count=$NPY_COUNT"
    echo "manifest_entries=$MANIFEST_COUNT"
  } | tee -a "$LOG"
  test "$SHA256_BEFORE" = "$SHA256_AFTER"
  test "$SKIPPED_EXISTING" -eq 1
  test "$NEWLY_WRITTEN" -eq 110
  test "$NPY_COUNT" -eq 111
  test "$MANIFEST_COUNT" -eq 111
  grep -q '"output_file": "000000.npy"' "$MANIFEST"
  ! grep -q '^REGENERATE ' "$LOG"
  echo "STEP_1_PRIOR_GENERATION=PASS" | tee -a "$LOG"
  ```

- Success: unchanged SHA-256, one skip, 110 writes, 111 priors and manifest
  entries, no regenerate/error, final `STEP_1_PRIOR_GENERATION=PASS`.
- Return: skip line, both hashes, four counts, manifest path, final PASS line.
- Failure: stop; Gate 2 is forbidden.

#### Gate 2 — Full strict validation

- Prerequisite: Gate 1 PASS.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS && set -o pipefail &&
  conda run --no-capture-output -n RT-GS \
  python tools/validate_normal_priors.py \
    --scene data/TiHuBird --images images --priors normal_priors \
    --manifest manifest.json --unit-atol 0.0005 \
    --expected-data-type indoor --expected-resolution 768 --expected-steps 10 \
    2>&1 | tee output/stage_a_tihubird_priors_validate_111.log
  ```

- Success: 111/111/111, empty error lists, `STRICT_VALIDATION=PASS`.
- Return: complete JSON report and final PASS line.
- Failure: stop; Gate 3 is forbidden.

#### Gate 3 — Fixed nine-view source/prior contact sheet

- Prerequisite: Gate 2 PASS.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS &&
  conda run --no-capture-output -n RT-GS python - <<'PY'
  from pathlib import Path
  import numpy as np
  from PIL import Image, ImageDraw, ImageOps
  root = Path("data/TiHuBird")
  out = Path("output/stage_a_tihubird_normal_priors")
  out.mkdir(parents=True, exist_ok=True)
  names = ["000000", "000014", "000028", "000042", "000055",
           "000069", "000083", "000097", "000110"]
  sheet = Image.new("RGB", (2580, 810), "white")
  draw = ImageDraw.Draw(sheet)
  for i, stem in enumerate(names):
      src = Image.open(root / "images" / f"{stem}.jpg").convert("RGB")
      n = np.load(root / "normal_priors" / f"{stem}.npy", allow_pickle=False)
      assert n.shape == (2152, 3827, 3)
      prior = Image.fromarray(np.clip((n + 1) * 127.5, 0, 255).astype("uint8"))
      src = ImageOps.contain(src, (410, 230))
      prior = ImageOps.contain(prior, (410, 230))
      x, y = (i % 3) * 860, (i // 3) * 270
      sheet.paste(src, (x + 10, y + 30)); sheet.paste(prior, (x + 440, y + 30))
      draw.text((x + 10, y + 8), f"{stem}: source", fill="black")
      draw.text((x + 440, y + 8), f"{stem}: prior", fill="black")
  target = out / "representative_9_source_prior.png"
  sheet.save(target)
  print(target)
  PY
  ```

- Success: one readable 3x3 sheet with all nine labelled source/prior pairs.
- Return: output path and the image itself.
- Failure: stop; Gate 4 is forbidden.

#### Gate 4 — Human prior-quality review

- Prerequisite: Gate 3 output exists.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS &&
  test -s output/stage_a_tihubird_normal_priors/representative_9_source_prior.png &&
  sha256sum output/stage_a_tihubird_normal_priors/representative_9_source_prior.png
  ```

- Success: human explicitly approves direction, structure, noise, and
  cross-view consistency.
- Return: hash plus four-part approve/reject statement.
- Failure: rejection stops all training; Gate 5 is forbidden.

#### Gate 5 — 1,000-step full-prior mono debug

- Prerequisite: explicit Gate 4 approval.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS && set -euo pipefail
  OUT="output/stage_a_tihubird_fullprior_1k_r2"
  LOG="output/stage_a_tihubird_fullprior_1k_r2.log"
  test ! -e "$OUT"; test ! -e "$LOG"
  export PYTHONHASHSEED=0
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n RT-GS \
  python train.py -s data/TiHuBird -m "$OUT" \
    --model_type surfel --resolution 2 \
    --normal_priors normal_priors --normal_prior_space camera \
    --lambda_norm 0.04 --lambda_mono 0.01 --lambda_perc 0.01 \
    --require_nonzero_mono --iterations 1000 \
    --test_iterations 1000 --save_iterations 1000 \
    --checkpoint_iterations 1000 --debug_interval 250 --disable_viewer \
    2>&1 | tee "$LOG"
  ```

- Success: Training complete, large supervised/nonzero counts, no OOM/NaN/Inf,
  checkpoint and 250/500/750/1000 debug maps, no visible normal/depth collapse.
- Return: first mono line, summary, completion line, checkpoint, representative
  RGB/normal/depth maps.
- Failure: stop; Gate 6 is forbidden.

#### Gate 6 — Freeze long-run inputs

- Prerequisite: Gate 5 automated and human checks pass; both 30k output paths
  are unused.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS && set -euo pipefail
  A="output/stage_a_tihubird_30k_mono0_r2"
  B="output/stage_a_tihubird_30k_mono001_r2"
  test ! -e "$A"; test ! -e "$B"; mkdir -p "$A" "$B"
  RECORD="$(mktemp)"
  {
    echo "git_head=$(git rev-parse HEAD)"
    echo "pythonhashseed=0"
    echo "images_count=$(find data/TiHuBird/images -maxdepth 1 -type f -name '*.jpg' | wc -l)"
    echo "images_tree_sha256=$(find data/TiHuBird/images -maxdepth 1 -type f -name '*.jpg' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f1)"
    echo "priors_count=$(find data/TiHuBird/normal_priors -maxdepth 1 -type f -name '*.npy' | wc -l)"
    sha256sum data/TiHuBird/normal_priors/manifest.json
    sha256sum data/TiHuBird/sparse/0/cameras.bin \
      data/TiHuBird/sparse/0/images.bin data/TiHuBird/sparse/0/points3D.bin
  } | tee "$RECORD"
  cp "$RECORD" "$A/input_freeze.txt"
  cp "$RECORD" "$B/input_freeze.txt"
  rm "$RECORD"
  cmp "$A/input_freeze.txt" "$B/input_freeze.txt"
  ```

- Success: identical records; HEAD, exact image set/count, prior count/manifest
  hash, COLMAP hashes, and `PYTHONHASHSEED=0` are frozen.
- Return: both records and successful equality result.
- Failure: stop; Gates 7–8 are forbidden.

#### Gate 7 — 30k matched baseline (`lambda_mono=0`)

- Prerequisite: Gate 6 PASS; run A contains only `input_freeze.txt`.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS && set -euo pipefail
  export PYTHONHASHSEED=0
  OUT="output/stage_a_tihubird_30k_mono0_r2"
  LOG="output/stage_a_tihubird_30k_mono0_r2.log"
  test -s "$OUT/input_freeze.txt"; test ! -e "$LOG"
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n RT-GS \
  python train.py -s data/TiHuBird -m "$OUT" \
    --model_type surfel --resolution 2 \
    --normal_priors normal_priors --normal_prior_space camera \
    --lambda_norm 0.04 --lambda_mono 0 --lambda_perc 0.01 \
    --iterations 30000 --test_iterations 7000 15000 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 10000 20000 30000 \
    --debug_interval 1000 --disable_viewer 2>&1 | tee "$LOG"
  ```

- Success: no `--require_nonzero_mono`; Training complete; test/save at
  7k/15k/30k, checkpoints at 10k/20k/30k; no OOM/NaN/Inf.
- Return: command, freeze record, three metrics, saves/checkpoints, completion.
- Failure: stop; Gate 8 is forbidden.

#### Gate 8 — 30k matched mono (`lambda_mono=0.01`)

Execution note: the command below was the original attempt. It stopped around
iteration 230 and produced no usable checkpoint. The accepted result is the
from-scratch retry at `output/stage_a_tihubird_30k_mono001_r2_retry1`, recorded
in A-9. Never use the interrupted directory in formal metrics.

- Prerequisite: Gate 7 PASS; freeze record still matches run A.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS && set -euo pipefail
  export PYTHONHASHSEED=0
  OUT="output/stage_a_tihubird_30k_mono001_r2"
  LOG="output/stage_a_tihubird_30k_mono001_r2.log"
  test -s "$OUT/input_freeze.txt"; test ! -e "$LOG"
  cmp output/stage_a_tihubird_30k_mono0_r2/input_freeze.txt "$OUT/input_freeze.txt"
  CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n RT-GS \
  python train.py -s data/TiHuBird -m "$OUT" \
    --model_type surfel --resolution 2 \
    --normal_priors normal_priors --normal_prior_space camera \
    --lambda_norm 0.04 --lambda_mono 0.01 --lambda_perc 0.01 \
    --iterations 30000 --test_iterations 7000 15000 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 10000 20000 30000 \
    --debug_interval 1000 --disable_viewer 2>&1 | tee "$LOG"
  ```

- Success: only output path and `lambda_mono` differ from Gate 7; no
  `--require_nonzero_mono`; all required nodes complete without OOM/NaN/Inf.
- Return: same evidence as Gate 7 plus config comparison.
- Failure: stop; Gate 9 is forbidden.

#### Gate 9 — Extract metrics and comparison image

- Prerequisite: Gates 7–8 PASS.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS &&
  { echo 'A lambda_mono=0';
    grep -E '\[ITER (7000|15000|30000)\] Evaluating train' output/stage_a_tihubird_30k_mono0_r2.log;
    echo 'B lambda_mono=0.01';
    grep -E '\[ITER (7000|15000|30000)\] Evaluating train' output/stage_a_tihubird_30k_mono001_r2.log;
  } | tee output/stage_a_tihubird_30k_matched_metrics.txt &&
  conda run --no-capture-output -n RT-GS python - <<'PY'
  from pathlib import Path
  from PIL import Image, ImageDraw, ImageOps
  root = Path("output")
  runs = [("A mono=0", root/"stage_a_tihubird_30k_mono0_r2"),
          ("B mono=.01", root/"stage_a_tihubird_30k_mono001_r2")]
  sheet = Image.new("RGB", (1920, 660), "white"); draw = ImageDraw.Draw(sheet)
  for row, node in enumerate((7000, 15000, 30000)):
      for mi, name in enumerate(("diffuse_color.png", "normal.png", "depth.png")):
          for ri, (label, run) in enumerate(runs):
              im = Image.open(run/"debug"/f"iteration_{node:06d}"/name).convert("RGB")
              im = ImageOps.contain(im, (320, 180)); x=(mi*2+ri)*320; y=row*220
              sheet.paste(im, (x, y+30)); draw.text((x+5,y+7), f"{node} {name} {label}", fill="black")
  target=root/"stage_a_tihubird_30k_matched_comparison.png"; sheet.save(target); print(target)
  PY
  ```

- Success: six matched metrics and one labelled 7k/15k/30k RGB/normal/depth
  comparison image.
- Return: metric file and comparison image.
- Failure: stop; Gate 10 is forbidden.

#### Gate 10 — Human acceptance

- Prerequisite: Gate 9 evidence complete and matched inputs/configurations proven.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS &&
  sha256sum output/stage_a_tihubird_30k_matched_metrics.txt \
    output/stage_a_tihubird_30k_matched_comparison.png
  ```

- Success: human explicitly accepts or rejects RGB quality, normal/depth
  stability, and the measured effect of `lambda_mono` at all three nodes.
- Return: hashes and signed-off findings for each node.
- Failure: rejection or ambiguity stops Stage A; Gate 11 is forbidden.

#### Gate 9.5 — Completed multi-view and crop audit

- [x] Export the six-view comparison to
  `output/stage_a_tihubird_gate9p5/comparisons/tihubird_30k_six_view_comparison.png`.
- [x] Export per-view crops for `000000`, `000010`, `000028`, `000055`,
  `000083`, and `000110`, plus `crops/contact_sheet_crops.png`.
- [x] Compare GT, `lambda_mono=0`, and `lambda_mono=0.01` retry1 RGB together
  with the two runs' normal/depth outputs.
- [x] Record that no obvious RGB degradation is visible in the global,
  six-view, or crop audit.

#### Gate 11 — Final overall Stage A closure (deferred)

- Status: not executed as overall Stage A closure. The StableNormal D-only
  baseline is conditionally accepted and frozen separately, while final Stage A
  closure remains deferred for the independent DiffusionRenderer prior
  experiment. Stage B/C/D remain forbidden.
- Historical command retained below for provenance only; do not execute it as
  part of the baseline snapshot.
- Command:

  ```bash
  cd /home/hanglee/桌面/RT-GS &&
  "${EDITOR:-vi}" docs/STATUS.md docs/DECISIONS.md docs/stages/STAGE_A.md &&
  git diff --check &&
  git diff -- docs/STATUS.md docs/DECISIONS.md docs/stages/STAGE_A.md &&
  git add docs/STATUS.md docs/DECISIONS.md docs/stages/STAGE_A.md &&
  git diff --cached --check &&
  git commit -m "docs(stage-a): record TiHuBird acceptance rollback" &&
  git rev-parse HEAD
  ```

- Success: truthful final conclusions, clean checks, rollback commit hash; Stage
  remains A until a separate explicit Stage B transition approval.
- Return: final diff summary, checks, commit hash.
- Failure: stop and remain in Stage A; no Stage B work.

### A-9 — Verified StableNormal baseline execution record

- [x] Gate 1/2: all 111 TiHuBird StableNormal priors exist and strict validation
  reports 111 images, 111 priors, 111 manifest entries, and
  `STRICT_VALIDATION=PASS`.
- [x] Gate 3/4: fixed-view prior inspection and human quality/direction review
  completed.
- [x] Gate 5: full-prior 1,000-step smoke completed with 1,000 supervised and
  1,000 nonzero steps, checkpoint save, and `Training complete.`
- [x] Gate 6: the accepted runs have byte-identical input-freeze records.
- [x] Gate 7: `lambda_mono=0` completed 30,000 iterations at
  `output/stage_a_tihubird_30k_mono0_r2`.
- [x] Gate 8: the first mono attempt stopped around iteration 230 with no usable
  checkpoint. `output/stage_a_tihubird_30k_mono001_r2_retry1` restarted from
  iteration 0 and completed 30,000 iterations; only retry1 is accepted.
- [x] Gate 9: matched metrics and global comparison completed.
- [x] Gate 9.5: six-view and crop audits completed without obvious RGB
  degradation.

Matched metrics:

| Iteration | `lambda_mono=0` L1 | `lambda_mono=0` PSNR | `lambda_mono=0.01` retry1 L1 | `lambda_mono=0.01` retry1 PSNR |
|---:|---:|---:|---:|---:|
| 7,000 | 0.0233018197119236 | 25.840939331054688 | 0.022160319611430168 | 26.222691726684573 |
| 15,000 | 0.022149086557328702 | 26.27469940185547 | 0.020849463716149333 | 26.722418594360352 |
| 30,000 | 0.01844301298260689 | 27.603187561035156 | 0.017842570878565313 | 27.78533058166504 |

Decision: conditionally accept the StableNormal D-only baseline and select
`lambda_mono=0.01`. It improves L1 and PSNR at all three matched nodes, and the
global/multi-view/crop audit shows no obvious RGB degradation. This does not
solve transparent decomposition: D-only still mixes the glass surface,
reflection, interior bird, and background. Its normal/depth are a foundation,
not final transparent geometry or final bird reconstruction.

### A-10 — DiffusionRenderer raw-prior Stage A extension

- [x] DR-1A generated a 24-frame raw pilot from final undistorted TiHuBird
  training images under `output/stage_a_tihubird_dr_pilot_24/`.
- [x] DR-1B generated raw `normal`, `depth`, `basecolor`, and
  `diffuse_albedo` for all 111 final training images under
  `output/stage_a_tihubird_dr_raw_111/`.
- [x] Record all 111 real mappings and the last chunk's nine padding slots
  separately.
- [ ] Implement a normal adapter.
- [ ] Select and validate a normal axis mapping.
- [ ] Connect any DiffusionRenderer prior loss or run training with it.

The raw buffers do not replace the StableNormal baseline. `basecolor` and
`diffuse_albedo` remain audit-only. All adapter, test, and experiment work must
continue on `feature/stage-a-diffrender-priors` without entering Stage B/C/D.

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
- [x] StableNormal D-only normal/depth visualizations passed the matched global,
  six-view, and crop audit for the conditional baseline.
- [x] `L_norm`, `L_mono`, and `L_perc` run in their supported configurations.
- [x] No Reflection/Transmittance/ray-tracing code path exists.
- [x] Every Stage A debug image can be generated.
- [x] Checkpoints save and resume.
- [x] Original 3DGS baseline remains runnable.
- [x] Required Stage A tests pass.
- [x] `docs/STATUS.md`, this checklist, and `docs/DECISIONS.md` reflect verified
  reality.
- [x] TiHuBird Gates 1–9 and the added Gate 9.5 audit have complete evidence;
  the interrupted first Gate 8 attempt is excluded and retry1 is accepted.
- [x] The StableNormal D-only baseline is conditionally accepted with
  `lambda_mono=0.01` and frozen by the baseline snapshot commit containing this
  record.
- [ ] Overall Stage A closure remains deferred during the independent
  DiffusionRenderer prior experiment.
- [ ] Only after an explicit future closure decision may `docs/STATUS.md` point
  to Stage B; Stage B/C/D are currently forbidden.

## Verified commands

```bash
git submodule update --init --recursive
conda run -n RT-GS pip install --no-build-isolation -e submodules/diff-surfel-rasterization
conda run -n RT-GS python -m pytest -q tests
conda run -n RT-GS python -m pytest -q tests/test_extract_video_keyframes.py
conda run -n RT-GS python -m pytest -q tests/test_generate_normal_priors.py tests/test_normal_prior_loading.py
colmap feature_extractor \
  --database_path output/stage_a_tihubird_colmap/work/database.db \
  --image_path data/TiHuBird/images_raw \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 --FeatureExtraction.gpu_index 0 \
  --default_random_seed 0
colmap exhaustive_matcher \
  --database_path output/stage_a_tihubird_colmap/work/database.db \
  --FeatureMatching.use_gpu 1 --FeatureMatching.gpu_index 0 \
  --FeatureMatching.guided_matching 1 --default_random_seed 0
colmap mapper \
  --database_path output/stage_a_tihubird_colmap/work/database.db \
  --image_path data/TiHuBird/images_raw \
  --output_path output/stage_a_tihubird_colmap/work/mapper \
  --Mapper.ba_global_function_tolerance 0.000001 \
  --Mapper.random_seed 0 --default_random_seed 0
colmap image_undistorter \
  --image_path data/TiHuBird/images_raw \
  --input_path output/stage_a_tihubird_colmap/work/mapper/0 \
  --output_path output/stage_a_tihubird_colmap/work/undistorted \
  --output_type COLMAP --copy_policy copy --max_image_size -1
colmap model_analyzer --path data/TiHuBird/sparse/0
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

The historical Truck 3,000-step run verifies the Stage A D-only schedule
boundary and code paths, but it is not target-scene quality evidence.
The real-prior smoke reported `L_mono=0.39248720` at iteration 1 and
`supervised_steps=1 nonzero_steps=1 max_L_mono=0.39248720`; one prior proves the
end-to-end path only and is superseded as target-scene evidence by the completed
111-prior TiHuBird smoke and matched 30k comparison.
The full resumable run generated 250 new priors and skipped the one verified
existing prior. Strict validation reported 251 images, 251 priors, 251 manifest
entries, zero missing/unexpected/duplicate/abnormal artifacts, and maximum unit
length error `1.78813934e-07`. The fixed contact sheet is
`output/stage_a_normal_priors/representative_9_source_prior.png`; this Truck
artifact remains historical smoke only.
The TiHuBird COLMAP run registered 111/111 images in one model and produced
84,989 points with 1.098552 px mean reprojection error. Full logs and the smooth
camera-trajectory diagnostic are in `output/stage_a_tihubird_colmap/`. This
satisfies SfM data preparation. TiHuBird prior validation, matched 30k training,
and the StableNormal D-only conditional baseline audit are now complete as
recorded in A-9.

## Superseded comparison — do not run

The previously documented Truck 30,000-step comparison was superseded before
execution. Truck is engineering smoke only. The matched TiHuBird
`lambda_mono=0` versus `0.01` retry1 protocol has completed and must not be
rerun or replaced by Truck evidence.
