# RT-GS Status

Current stage: Stage B — Differentiable Ray Tracing and Reflection

Current branch: `feature/stage-b-reflection-dr-c03`

Stage state: Stage A is formally closed. Stage B-0 was approved by the user and
B-1a/B-1b/B-1c implementation is present with synthetic/CUDA tests passing.
The first user-operated TiHuBird smoke restored D and initialized R, then
aborted before its first optimization step on an ASCII-locale CUDA JIT path
error. The user-operated `smoke_2_retry1` then completed 2/2 steps from the
fixed code and produced a valid Stage B checkpoint, independent D/R PLYs, and
real debug maps. After the first diagnostic resume stopped on the RNG
map-location bug, `resume_diag_15003_retry1` completed and verified B-1d at
global 15,003 / R local 3. The first controlled health pilot was interrupted
with SIGINT after global 15,019 / R local 19 and classified
`PILOT_ABORTED_FOR_PROFILE`: ordinary steps took roughly 50--60 seconds, so its
partial output is preserved and must never be used as a resume source. Nsight
then isolated 160 long candidate advanced-index backward kernels as 97.8% of a
step. Commit `c87bb66` replaces only that R-parameter gather backward with a
grouped custom CUDA reduction. A matched two-step profile from the original
15,003/3 checkpoint reduced global 15,004 from 50.461 s to 1.572 s with exact
15004/15005 scalar losses. The fresh controlled 100-step retry then completed
from global 15,003 / R local 3 to global 15,103 / R local 103 without OOM,
non-finite values, count changes, or performance collapse. It is accepted as a
health pilot, not as Stage B acceptance or authorization for long training.
Stage B is neither complete nor accepted. The frozen StableNormal baseline and
C03 initialization artifacts remain unchanged. Stage C and Stage D have not
started.

Stage B branch point: `772c0c0e1c9fec012a10795101e874e2bc065c44`

Stage B implementation commits:

```text
7ad0b8881b9b2a00e9f766b68be7ff051c74281e  rendering/path decisions
bb5eb072e757a32735ad378b38a3d4b489ac1a57  B-1a model/math/checkpoint foundation
30dd8be5d05d6da9081b0a1f4b80a555234b74f3  B-1b fail-closed CUDA LBVH
953334009a8db3d63cc68179842687ba306186c2  B-1c train/render/debug integration
c6b918442eb1891eeba1c5914a8db0b85763ec4b  checkpoint schedule metadata fix
f1e90e780eb4777ddeeece70bc393e0b21b080db  on-disk checkpoint resume test
36b7fdfdd98ec7f3d04d256cf76830189a6a10a9  ASCII-only CUDA JIT staging fix
8cd03e861f89487472144d5c9d4fcd7d62915e19  B-008 diagnostics decision
9d69a0d2a8ac402744ee638c64822b8d4e601da9  B-1d observability implementation
47bc8422a7c024ad7946a053f8a21e74d9af2851  B-009 RNG device contract
9156b1eed7ecab41b873ef796d27783289723f6d  CUDA checkpoint RNG restore fix
c87bb66934ddfcf86173c77dc9dcd724837ca8ae  grouped candidate-gradient reduction
```

Stage A code evidence commit: `829dd82dc74f4c5dce640201448626df24afa25a`

## Repository and scene

- The original 3DGS baseline remains available through `--model_type 3dgs`.
- RT-GS uses `--model_type surfel` and the pinned 2D surfel rasterizer.
- Diffuse and Reflection now have separate Stage B models, optimizers,
  schedulers, densification state, checkpoint namespaces, PLY exports, and debug
  outputs. No Transmittance model or Stage C/D path exists.
- `data/TiHuBird` is the real acceptance scene. Truck artifacts are historical
  engineering smoke evidence only.
- TiHuBird has 111 final undistorted PINHOLE images at 3827x2152, 111/111 COLMAP
  registrations, and 84,989 sparse points.

## Frozen StableNormal Stage A baseline

- The frozen rollback branch is `baseline/stage-a-stablenormal` at
  `3e9f62ae8b34eaaf77ce67ba86a077041aaa5f04`.
- The accepted D-only output is
  `output/stage_a_tihubird_30k_mono001_r2_retry1/` with
  `lambda_mono=0.01`. Its 30,000-step checkpoint and point cloud remain present
  and must not be overwritten, moved, or modified.
- Generated all 111 TiHuBird StableNormal priors in
  `data/TiHuBird/normal_priors/` and passed strict validation: 111 images, 111
  priors, 111 manifest entries, with empty missing, unexpected, duplicate, and
  abnormal lists. Evidence:
  `output/stage_a_tihubird_priors_validate_111.log`.
- The matched `lambda_mono=0` and `lambda_mono=0.01` retry1 runs completed to
  30,000 iterations. The first `lambda_mono=0.01` attempt stopped around
  iteration 230 without a usable checkpoint and remains excluded.
- Gate 9 matched metrics and the global comparison are present at
  `output/stage_a_tihubird_30k_matched_metrics.txt` and
  `output/stage_a_tihubird_30k_matched_comparison.png`; Gate 9.5 six-view and
  crop evidence remains under `output/stage_a_tihubird_gate9p5/`.

## Matched StableNormal 30,000-step metrics

| Iteration | `lambda_mono=0` L1 | `lambda_mono=0` PSNR | `lambda_mono=0.01` retry1 L1 | `lambda_mono=0.01` retry1 PSNR |
|---:|---:|---:|---:|---:|
| 7,000 | 0.0233018197119236 | 25.840939331054688 | 0.022160319611430168 | 26.222691726684573 |
| 15,000 | 0.022149086557328702 | 26.27469940185547 | 0.020849463716149333 | 26.722418594360352 |
| 30,000 | 0.01844301298260689 | 27.603187561035156 | 0.017842570878565313 | 27.78533058166504 |

## DiffusionRenderer C03 exploratory result

- DiffusionRenderer priors were evaluated independently on
  `feature/stage-a-diffrender-priors`; they never replaced or modified the
  StableNormal baseline.
- C03 completed 15,000 D-only iterations at
  `output/stage_a_tihubird_drnormal_c03_15k/`. The log
  `output/stage_a_tihubird_drnormal_c03_15k.log` records saves at 7,000 and
  15,000, checkpoint saves at 7,000/10,000/15,000, and `Training complete.`
- Its 15,000-step train metrics are L1 `0.021616848371922973` and PSNR
  `26.401429367065433`. The matched StableNormal retry1 15,000-step metrics are
  L1 `0.020849463716149333` and PSNR `26.722418594360352`.
- C03 trained stably and did not show the front glass surface swallowing the
  bird subject. The RGB/normal/depth audit nevertheless did not establish a
  clear D-only advantage over the frozen StableNormal baseline. Therefore C03
  will not be extended to 30,000 D-only iterations.
- The comparison image is
  `output/stage_a_tihubird_drnormal_c03_15k/comparison_vs_stablenormal_15k.png`.
- The C03 15,000-step checkpoint is a readable `rtgs_stage_a` checkpoint at
  iteration 15,000 with 346,118 finite Diffuse surfels. Its usable initialization
  artifacts are:

  ```text
  output/stage_a_tihubird_drnormal_c03_15k/chkpnt15000.pth
  output/stage_a_tihubird_drnormal_c03_15k/point_cloud/iteration_15000/
  ```

- These C03 artifacts are read-only Stage B initialization candidates. C03 is
  not a proven replacement for the StableNormal D-only configuration.

## Tests and verified behavior

- `MAX_JOBS=4 conda run --no-capture-output -n RT-GS python -m pytest -q
  tests` → 84 passed on 2026-07-01. This includes the complete Stage A
  regression; Stage B model/ray/BRDF/checkpoint/render/debug/mask/JIT staging
  and candidate-gather equivalence tests; and eleven DR proposal/review audit,
  output, fail-closed, and training-isolation contracts.
- The CUDA extension compiled successfully for PyTorch 2.0.1 + CUDA 11.8 on an
  RTX 3090. CUDA/oracle consistency, chunking, refit/rebuild, and nonzero
  finite-difference gradients for xyz/rotation/scaling/opacity/color/ray
  origin/ray direction passed.
- Original 3DGS and surfel training/render paths, checkpoint resume, PLY reload,
  densification, material gradients, and Stage A debug exports have passed their
  documented checks.
- Synthetic Stage B rendering produced real reflection hit/color/alpha/depth,
  D/F/G/fr/wr, and diffuse/reflection contribution tensors and debug files in
  pytest temporary directories only.
- The user-operated `smoke_2` attempt created only initialization/config/event
  artifacts and stopped at 0/2 steps before any checkpoint or D/R PLY save. Its
  attached log identifies `UnicodeEncodeError` while PyTorch writes the CUDA JIT
  Ninja file from the non-ASCII repository path. Codex did not run this smoke or
  modify its partial output.
- A fresh forced-C-locale CUDA compile reported preferred encoding
  `ANSI_X3.4-1968` and loaded the extension successfully from the new ASCII-only
  staging path. No real manual soft-mask set exists yet.
- The user-operated `smoke_2_retry1` completed at global iteration 15,002 and R
  local step 2. Its `rtgs_stage_b` checkpoint contains 346,118 Diffuse and 4,096
  Reflection surfels, separate optimizer namespaces, and no non-finite tensor in
  recursive D/R state inspection. The independent D/R PLYs and both debug nodes
  are present. C03 source checkpoint and PLY hashes remain unchanged.
- The reviewed fixed view had 128,582 valid rays and 128,579 final alpha hits.
  Reflection color/alpha/depth were spatially structured, but the physical
  reflection contribution quantized to black and microfacet D display-clamped
  to white. B-1d now preserves these physical images while adding p99/log
  companion views, raw statistics, candidate/exact-intersection counts, CUDA
  phase timing, and peak-memory metadata only on requested debug renders.
- The first B-1d resume attempt loaded all cameras and the Stage B checkpoint,
  then stopped at 0 steps because `map_location="cuda"` moved the CPU PyTorch
  RNG ByteTensor onto CUDA. Its partial output contains only configuration,
  camera/input, and event artifacts; no checkpoint or D/R PLY. The corrected
  loader validates RNG uint8 tensors, restores contiguous CPU copies for both
  CPU and CUDA generators, passes an on-disk CUDA-map-location test, and restored
  the real user checkpoint RNG read-only.
- The corrected user resume completed from global 15,002 / R local 2 to global
  15,003 / R local 3 and wrote a finite checkpoint plus separate D/R PLYs. Its
  fixed view reports candidate p50/p95/p99 `111/205/256`, exact-intersection
  p50/p95/p99 `28/64/76`, about 152 ms raytrace time for 128,582 rays, and about
  319 MiB incremental peak allocation. Candidate p99 is 6.25% of the field, so
  the LBVH is not behaving as a 4,096-surfel brute-force scan.
- Microfacet D is finite and tightly distributed over `1.270–1.276`; its nearly
  white visualization is correct. Reflection contribution is nonzero and
  spatially structured in the companion map. No Reflection count or initial
  scale adjustment is justified before the controlled pilot.
- The aborted `health100` output contains completed TensorBoard scalars through
  global 15,019, but it is neither healthy nor blocked evidence and is not a
  resume source. The original read-only
  `resume_diag_15003_retry1/chkpnt15003.pth` remains the sole pilot resume point
  (SHA-256
  `ad92c7d75312cc5df60b7a1dd5d762d8e7d65b8de0f90264d742f969baf7e144`).
- The matched Nsight comparison uses global 15,004 bounded by consecutive D
  raster forwards. Wall time fell from `50.461 s` to `1.572 s`; the R backward
  envelope fell from `49.524 s` to `0.547 s`. The old 160 long
  `indexing_backward_kernel` calls (`49.356 s`) are absent; the nine remaining
  unrelated short calls total `0.430 ms`. The new 32 grouped reducers total
  `253.612 ms`. R forward, D rasterization, VGG, and optimizer timing remained
  in the same sub-second regime. The 15004/15005 losses are exactly unchanged:
  `0.08710507303476334` and `0.06420626491308212`.
- The new profile's 200 ms external sampler observed a 15,067 MiB device-total
  peak (302 MiB idle baseline, approximately 14,765 MiB attributable to the
  process). This is higher than the earlier separately observed 10,627 MiB
  total / 10,320 MiB process peak and remains a short-pilot risk; it does not
  negate removal of the measured backward bottleneck.
- The only post-repair health pilot completed at
  `output/stage_b_tihubird_reflection_dr_c03_health100_candidate_reduce_g15003_15103_retry1/`.
  Its 95 non-debug iteration intervals have p50/p95/mean
  `0.939/1.092/0.944 s`. Mean training raytrace forward and candidate-backward
  envelope are about `228.7 ms` and `643.0 ms`; all 100 iterations used 32
  candidate backward calls. Four evaluation/debug nodes and the accepted final
  15,103 debug node are complete with all companion maps.
- D/R counts remained `346,118/4,096`; no densify/prune event changed topology.
  Candidate p50/p95/p99 moved from `126/227/285` at 15,025 to `146/265/341` at
  15,103, while exact intersections moved from `33/72/86` to `38/83/109`.
  This is gradual growth, not an order-of-magnitude acceleration failure.
  Reflection contribution remained finite and essentially fully nonzero; its
  mean/p99 increased from `1.08e-4/5.58e-4` to `1.85e-4/1.12e-3`.
- A read-only fixed-view render of the final checkpoint found 128,582 valid
  surface pixels with ks min/mean/p1/p5/p50/p95/p99/max
  `0.08796/0.09726/0.09321/0.09468/0.09730/0.10000/0.10189/0.10640`;
  both `ks<0.01` and `ks>0.9` fractions are zero. The field did not collapse.
- Exact allocator evidence is peak allocated `18,094,459,392` bytes
  (16.85 GiB) and peak reserved `20,333,985,792` bytes (18.94 GiB). Current
  allocated memory after compute is non-monotonic (only 48.5% of transitions
  nondecreasing, 1.36--1.82 GB), so no live-tensor leak is evident. Reserved
  memory is 96.0% nondecreasing and rose from 12.51 GB to 20.33 GB as PyTorch
  cached larger per-view temporary blocks. This is a material long-run headroom
  risk even though the pilot did not OOM.
- Final checkpoint `chkpnt15103.pth` is `rtgs_stage_b` at global 15,103 / R
  local 103 with SHA-256
  `6f43ff1335f99e03f8ee08e4575ad4c91b29189cbe23a67942b23557adb87354`.
  Recursive inspection covered 63 tensors / 19,912,887 elements and found no
  NaN/Inf. It records `lambda_spec=0` and no specular-mask manifest.

## DR glass-mask proposal audit (not supervision)

- The strict input audit passed for all 111 real TiHuBird views. Source RGB is
  JPEG RGB uint8 at 3827x2152; stored DR RGB, normal, depth, basecolor, and
  diffuse_albedo are PNG RGB uint8 at 704x384. Source hashes, ordered stems,
  dimensions, manifest records, generation-time alignment validation, and all
  artifact paths/hashes were checked. Raw slots 111--119 are the nine final
  chunk padding records and are excluded from every real-view mapping.
- DR normal is decoded as `rgb / 127.5 - 1`, then normalized only for
  continuity cues. Its component/axis convention remains unconfirmed; no
  semantic axis interpretation is used. The 111 real normal maps have no
  non-finite or decoded-near-zero pixel. DR depth is a per-frame relative RGB
  visualization with no documented invalid sentinel or COLMAP/metric scale;
  only within-view boundaries and continuity are used.
- Nine automatic review proposals were generated for stems 000000, 000014,
  000028, 000042, 000055, 000069, 000083, 000097, and 000110 at
  `output/stage_b_tihubird_dr_glass_proposal_9views_v2/`. Each contains the
  requested RGB/DR/boundary/proposal/uncertainty views and metadata; the root
  contains the strict audit manifest, contact sheet, proposal index, and human
  review contract.
- These files are explicitly automatic drafts with no training role. No
  `reviewed_soft` directory, formal training mask manifest, mask loader input,
  `lambda_spec` run, or new scene training was created. Visual inspection finds
  the enclosure outline in all nine views, while every view conservatively
  flags possible bird inclusion because RGB texture exists behind the glass.
  Human review is required before any 111-view expansion decision.
- The user accepted the nine-view method and authorized a frozen-method
  111-view expansion. Commit `a2ee732` remains the proposal algorithm reference;
  no proposal formula, threshold, morphology, connectivity, or per-frame
  parameter changed. The completed review package is
  `output/stage_b_tihubird_dr_glass_proposal_111_v1/`.
- The package passed strict validation for ordered stems 000000--000110, source
  size 3827x2152, single-channel uint8 proposals containing both 0 and 255,
  input/output SHA-256 records, finite DR normals, nondegenerate area, and no
  padding intersection. It contains 111 review-queue entries, ten chronological
  overlay pages, ten risk-priority overlay pages, four boundary-review crops per
  frame, distribution plots/data, and an automatic anomaly list.
- Proposal area min/mean/p50/p95/max is
  `0.13856/0.22100/0.21222/0.31571/0.35572`. Uncertainty fraction above
  140/255 has min/mean/p50/p95/max
  `0.001783/0.003679/0.003812/0.004875/0.005206`. Five frames carry the
  background-inclusion review flag: 000012, 000013, 000039, 000040, and 000041.
  Fifteen frames carry the reflection-misclassification flag; all 111 retain the
  conservative bird-behind-glass flag.
- Visual inspection of all ten chronological pages found continuous enclosure
  outlines without a systematic table/wall capture or enclosure loss. The five
  automatic anomalies still align visually and remain first-priority human
  review items. No automatic proposal was deleted, rewritten, or promoted to a
  reviewed mask.

## Current boundary and next exact task

- Do not extend the health pilot or start long training. Preserve its checkpoint,
  metrics, diagnostics, D/R PLYs, and allocator evidence.
- `lambda_spec=0` must keep `--specular_masks` empty and emits no mask/overlay.
  The immediate next action is user review of the 111-entry risk-sorted queue,
  beginning with 000041, 000012, 000040, 000039, and 000013, not training.
  Automatic proposals must be manually corrected/confirmed before any
  `reviewed_soft` directory or formal training manifest exists; even then
  `lambda_spec>0` needs separate approval.
- Stage B solves reflection only. It must not claim transmission,
  bird/background separation, transparent mesh, or two-hit geometry.
- Do not start a matched StableNormal-versus-C03 Stage B comparison in parallel.
  It may be designed later if the reflection hypothesis requires it.
- Stage C and Stage D remain forbidden until their own acceptance and explicit
  stage transitions.

Pending before acceptance: a real 111/111 manual soft-mask set and verified
`L_spec` behavior. Peak reserved-memory headroom must remain visible in any later
training decision. Stage B acceptance cannot advance before that evidence
exists.
