# RT-GS Engineering Decisions

This log records implementation choices required where the paper or master plan
does not fully specify behavior. New entries must use the decision template from
`RTGS_MASTER_PLAN.md`.

Stage A decisions and the authorized Stage B entry decision are recorded below.

## A-001 — Single master-plan location

Date: 2026-06-28

Question: The master plan text mentions a future `docs/RTGS_MASTER_PLAN.md`, but
the repository task defines root `RTGS_MASTER_PLAN.md` as the unique source.

Chosen implementation: Keep only root `RTGS_MASTER_PLAN.md` and point all memory
files to it.

Alternatives: Copy the plan into `docs/`, creating two authoritative-looking
files.

Why: A duplicate can drift and directly conflicts with the explicit single-source
requirement.

Paper fidelity: No rendering impact.

Impact: Contributors must read the root file.

Required ablation: None.

## A-002 — True 2D rasterizer and material-pass layout

Date: 2026-06-28

Question: How should Stage A obtain true 2D ray–surfel rasterization and composite
more than three learned material channels?

Chosen implementation: Pin the official 2DGS `diff-surfel-rasterization` at
commit `e0ed0207b3e0669960cfad70852200a4a5847f61`. Render three CUDA passes with
identical geometry, sorting, opacity, and camera state: `Cd`, packed
`roughness/ks`, and `f0`. Material passes use a zero background.

Alternatives: Flatten a 3D covariance; maintain a local eight-channel CUDA fork;
or use a slow PyTorch compositor.

Why: The pinned implementation performs the perspective-correct ray/2D-splat
intersection. Reusing the unmodified kernel is reproducible, and identical passes
give each map exactly the same front-to-back alpha weights. Tests verify the
single-surfel equality `material_map = alpha * material` and all parameter
gradients.

Paper fidelity: Geometry and compositing follow 2DGS/Stage A; the multi-pass
execution is an engineering layout choice and does not change image formation.

Impact: Stage A rasterization performs three sorted CUDA passes per view.

Required ablation: Profile a fused multi-channel kernel before performance
release; no image-quality ablation is required.

## A-003 — Diffuse material parameterization and initialization

Date: 2026-06-28

Question: The master plan fixes roughness/f0/ks activation but does not fully
specify base-color activation or initial material values.

Chosen implementation: Store direct RGB `base_color_raw` and activate it with
sigmoid. Initialize base color from the COLMAP point color, opacity to `0.1`,
roughness to `0.5`, RGB f0 to `0.04`, and ks to `0.1`. Use a shared material
learning-rate default of `0.0025`.

Alternatives: Spherical harmonics for diffuse color; unconstrained base color;
or random material initialization.

Why: Stage A explicitly requires `base_color_raw [N,3]`; bounded direct RGB is
the minimal diffuse representation. Conservative material values keep all maps
finite before later-stage shading exists.

Paper fidelity: Stage A-compatible engineering default; the exact initial values
are not claimed as paper values.

Impact: Diffuse color is view-independent in RT-GS surfel mode. The baseline
3DGS SH representation is unchanged.

Required ablation: Material initialization may be ablated in Stage E if it
materially changes final convergence.

## A-004 — Surfel orientation initialization

Date: 2026-06-28

Question: How should surfel rotation be initialized when COLMAP points usually
contain zero normals?

Chosen implementation: Align local +Z to valid input point normals. For points
without valid normals, initialize a normalized random quaternion under the
repository's seeded RNG.

Alternatives: Identity orientation for every surfel or a separate learned normal.

Why: Identity creates a strong global orientation bias; a separate normal would
violate the requirement that normals derive from rotation/tangent geometry.

Paper fidelity: Consistent with geometry-derived normals; initialization is an
engineering choice.

Impact: Early normal maps are noisy and are regularized by `L_norm` during
training.

Required ablation: Compare random versus point-normal initialization when input
normals are available.

## A-005 — Depth, position, normal, and material-map semantics

Date: 2026-06-28

Question: Which depth should define the Stage A surface, and should material maps
be divided by accumulated alpha?

Chosen implementation: Use expected camera-z depth (`sum(w_i * z_i) / alpha`),
unproject it to world position, normalize the alpha-blended normal, and
face-forward it toward the camera. Keep roughness/f0/ks premultiplied exactly as
`sum(w_i * attribute_i)`, without alpha division, as written in the master plan.

Alternatives: Median depth; unnormalized accumulated depth; alpha-normalized
material maps.

Why: Expected depth is differentiable and directly supports position
unprojection. Premultiplied material maps match the explicit compositing formula.

Paper fidelity: Matches the Stage A output and compositing contract.

Impact: Background material pixels are zero; valid surface normals have unit
length and face the camera.

Required ablation: Expected versus median depth can be compared if later mesh
quality requires it, but median depth is not implemented in Stage A configuration.

## A-006 — Monocular normal prior convention

Date: 2026-06-28

Question: The master plan does not state the coordinate space or invalid-pixel
encoding of `.npy` normal priors.

Chosen implementation: Default to camera-space HWC or CHW normals, selectable
with `--normal_prior_space camera|world`. Resize with bilinear interpolation,
renormalize, treat non-finite/near-zero vectors as invalid, transform camera-space
priors to world space, and face-forward before cosine loss. Missing files disable
`L_mono` for that view rather than fabricate targets.

Alternatives: Assume world space only; require every frame; fill missing priors
with zeros and include them in the loss.

Why: Camera-space is the common monocular-prediction convention, while explicit
selection avoids a hidden coordinate assumption. Masking missing data prevents
false supervision.

Paper fidelity: The loss formula is preserved; coordinate/loading behavior is an
engineering choice.

Impact: The included datasets have no normal-prior directory, so the 3,000-step
smoke run records zero `L_mono`; loader and loss execution are covered by tests.

Required ablation: Camera-space versus world-space only when evaluating a prior
source that can emit both conventions.

## A-007 — VGG perceptual loss definition

Date: 2026-06-28

Question: Which VGG-16 layers and normalization implement the underspecified
feature loss?

Chosen implementation: Frozen ImageNet VGG-16 features after blocks ending at
indices 4, 9, 16, and 23, with ImageNet input normalization. Average L1 feature
distance across the four blocks.

Alternatives: One VGG layer, unnormalized inputs, LPIPS, or untrained VGG.

Why: Multi-scale frozen VGG features are a standard, deterministic realization
of the required VGG-16 L1 feature loss.

Paper fidelity: The master plan specifies VGG-16 feature L1 but not exact layers;
this is an engineering definition.

Impact: The pretrained weight file is required on first use and is cached by
TorchVision.

Required ablation: Perceptual loss on/off is sufficient for later evaluation.

## A-008 — Baseline switch and checkpoint compatibility

Date: 2026-06-28

Question: How can RT-GS Stage A coexist with the original 3DGS baseline without
silently loading the wrong checkpoint type?

Chosen implementation: `--model_type 3dgs` remains the default and uses the
untouched `GaussianModel`/rasterizer/checkpoint tuple. `--model_type surfel`
selects `DiffuseSurfelModel` and a versioned `rtgs_stage_a` dictionary containing
model type, material tensors, optimizer/exposure state, iteration, and relevant
loss configuration.

Alternatives: Replace the baseline model globally or infer model type from PLY
attributes.

Why: An explicit switch preserves reproducibility and rejects incompatible
checkpoint formats early.

Paper fidelity: No image-formation impact.

Impact: Training and rendering commands must specify `--model_type surfel` for
RT-GS Stage A.

Required ablation: None.

## A-009 — Offline StableNormal prior artifact contract

Date: 2026-06-28

Question: How should real monocular priors be generated without adding diffusion
dependencies to the RT-GS training environment or introducing hidden online
model access?

Chosen implementation: Generate priors ahead of training from the dedicated
StableNormal environment with `tools/generate_normal_priors.py`. Use the existing
loader's flat `<image-stem>.npy` naming rule and write HWC float32, finite,
unit-length camera-space normals. Decode StableNormal's RGB output with
`rgb / 127.5 - 1`, renormalize every pixel, and reject non-finite or near-zero
vectors. Run StableNormal in indoor mode without segmentation, force all model
and DINOv2 loads to local paths, block Python network access, write each prior
atomically, and record provenance in `manifest.json`. Camera-to-world conversion
and face-forward orientation remain in the training path because generation has
no scene geometry at each pixel.

Alternatives: Store PNG priors; emit CHW or world-space arrays; install
StableNormal into the RT-GS environment; permit Hugging Face or Torch Hub
fallback downloads; or run object/outdoor segmentation during preprocessing.

Why: Float32 `.npy` avoids additional quantization after RGB decoding and is the
format already validated by the Stage A loader. Keeping generation external
isolates incompatible dependencies, while fail-closed offline loading makes the
artifact reproducible and prevents accidental model substitution.

Paper fidelity: This is a Stage A preprocessing and artifact-provenance choice;
the monocular cosine loss is unchanged.

Impact: A scene can contain a resumable, auditable `normal_priors/` directory.
Missing priors still leave a view unsupervised rather than fabricating targets.
The default-off `--require_nonzero_mono` smoke-test guard reports supervised and
nonzero steps and fails the run if no positive finite monocular loss is observed;
it does not alter sampling or the loss when disabled.

Required ablation: Satisfied by the matched TiHuBird 30,000-step
`lambda_mono=0` versus `0.01` comparison recorded in A-010 and A-013.

## A-010 — Matched monocular-normal comparison protocol

Date: 2026-06-28

Status: Completed on 2026-06-29 for TiHuBird. Truck remains historical
engineering smoke only.

Question: How should the effect of the completed StableNormal prior set be
measured without confounding it with resolution, initialization, schedules, or
debug/checkpoint differences?

Chosen implementation: After all 111 TiHuBird priors pass strict and manual
quality validation, run two fresh 30,000-iteration TiHuBird trainings using the
candidate `--resolution 2`. Run A uses `lambda_mono=0`; run B uses
`lambda_mono=0.01`. Apart from the unavoidable output/log paths, the only
configuration difference is `lambda_mono`.

Both runs must use exactly the same `data/TiHuBird` scene, final undistorted
`images/`, `sparse/0/`, complete `normal_priors/` path, model initialization,
repository deterministic seed, `lambda_norm=0.04`, `lambda_perc=0.01`, 30,000
iterations, resolution, optimizer/densification schedule, evaluation and save
nodes at 7,000/15,000/30,000, and checkpoint nodes at
10,000/20,000/30,000. Both runs load the same priors so the zero-weight run
exercises the same data path. Both launch environments explicitly export
`PYTHONHASHSEED=0`.

Before either long run starts, record the same frozen-input identity in each
run's new output directory: repository HEAD, SHA-256 of the prior manifest and
all three COLMAP binary files, and image/prior counts. The two records must match
exactly except for their enclosing output paths. Any input change between runs
invalidates the comparison.

Neither 30,000-step run may use `--require_nonzero_mono`: adding it only to run B
would introduce a second configuration difference, while adding it to run A
would fail by design because its weighted `L_mono` is not the acceptance
variable. `--require_nonzero_mono` is restricted to short mono-chain and
full-prior debug smokes.

Alternatives: Compare against Truck; use resolution divisor 1, 4, or 8; omit
loading priors in the zero-weight run; change initialization or seed; use
different schedules/nodes; or enable `--require_nonzero_mono` in a long run.

Why: Loading the same data in both runs preserves memory and data-path behavior,
while deterministic initialization and identical schedules isolate the
monocular-normal coefficient.

Paper fidelity: This is a Stage A engineering evaluation protocol. It does not
change the loss formula, model, renderer, or training implementation.

Verified outcome: all 111 priors passed strict validation, and the full-prior
1,000-step smoke completed. Gate 6 produced byte-identical frozen-input records
for the two accepted runs. Gate 7 (`lambda_mono=0`) completed to 30,000. The
first Gate 8 attempt stopped around iteration 230 without a usable checkpoint;
`retry1` restarted from iteration 0 and completed to 30,000, and only retry1 is
used in the comparison. Both accepted runs saved checkpoints at
10,000/20,000/30,000 and evaluation/PLY nodes at 7,000/15,000/30,000.

The matched metrics were:

| Iteration | `lambda_mono=0` L1 | `lambda_mono=0` PSNR | `lambda_mono=0.01` retry1 L1 | `lambda_mono=0.01` retry1 PSNR |
|---:|---:|---:|---:|---:|
| 7,000 | 0.0233018197119236 | 25.840939331054688 | 0.022160319611430168 | 26.222691726684573 |
| 15,000 | 0.022149086557328702 | 26.27469940185547 | 0.020849463716149333 | 26.722418594360352 |
| 30,000 | 0.01844301298260689 | 27.603187561035156 | 0.017842570878565313 | 27.78533058166504 |

Gate 9 produced the matched metrics/global comparison, and Gate 9.5 completed
the six-view and crop audit. These checks support a conditional StableNormal
D-only baseline decision; they do not close all of Stage A or authorize a later
stage.

Required ablation: Completed for the StableNormal D-only baseline. A separate
DiffusionRenderer prior experiment remains a Stage A extension and must not be
mixed into this matched comparison.

Documentation-only audit note (2026-06-29): source review confirmed that an
all-image invocation of `generate_normal_priors.py` strictly validates and
prints `SKIP verified` for an existing valid prior without replacing its `.npy`.
Its manifest update merges entries by output filename and writes the merged JSON
atomically, preserving prior provenance fields while updating verification and
invocation metadata. This audit modified only `STATUS.md`, `DECISIONS.md`, and
`STAGE_A.md`; it ran no prior generation, training, COLMAP, rendering, or tests
and created no Git commit.

The sole operator procedure for this protocol is the ordered 11-gate TiHuBird
Operator Runbook in `docs/stages/STAGE_A.md`. Chat-history commands, the former
A–I summary, and Truck commands are not alternative runbooks. Each gate must
pass and return its requested evidence before the next command is authorized.

## A-011 — Static-scene video keyframe selection

Date: 2026-06-29

Question: How should a raw glass-dome video be converted into a deterministic,
COLMAP-ready Stage A image sequence without resizing, silently mixing older
frames, or selecting blurred and redundant frames?

Chosen implementation: Use `tools/extract_video_keyframes.py`. Require local
`ffprobe` and `ffmpeg`, probe the first video stream, and uniformly sample JPEG
candidates with FFmpeg's `fps` filter. Candidate timestamps use the deterministic
uniform grid `candidate_index / candidate_fps`. Score every candidate with
Laplacian variance, under/overexposed pixel ratios, a 32x32 grayscale visual
feature, and temporal separation. Select greedily with a seeded tie-breaker and
weights `0.70 quality + 0.20 visual novelty + 0.10 temporal coverage`; reject
exposure violations, candidates closer than the configured time gap, and visual
similarity at or above the configured duplicate threshold. Sort selected frames
back into time order and name them `000000.jpg`, `000001.jpg`, and so on.

Write full-resolution JPEGs at quality 98 by default. Store `keyframes.json`,
`contact_sheet.jpg`, and `selection_report.txt` beside the `images/` directory.
Dry-run writes only those metadata/preview artifacts. Formal extraction writes
an atomic in-progress manifest, atomically installs every image, updates progress
after each frame, and marks the manifest complete only at the end. Existing
files require matching video fingerprint, parameters, output names, dimensions,
and manifest state; unexpected or unmanifested files are fatal. Selection
shortfall is reported and fails instead of silently relaxing thresholds.

Alternatives: Fixed-stride extraction without quality checks; OpenCV video
decoding; resizing candidates; perceptual neural embeddings; silently appending
to an existing image directory; or automatically relaxing filters to reach the
requested count.

Why: FFmpeg handles real-world codecs and timestamps more robustly than an
application-level decoder. The lightweight deterministic metrics require no new
model dependency, preserve source resolution, and provide auditable reasons for
every selection/rejection. Fail-closed resume semantics prevent mixed datasets.

Paper fidelity: This is Stage A input preprocessing and does not change the
renderer, representation, losses, or training schedule.

Impact: Candidate extraction requires temporary high-quality JPEG storage and
selection is quadratic in candidate/selected count. A real video run is still
required before claiming operational extraction success.

Required ablation: None. Inspect the dry-run contact sheet and report before
formal extraction; adjust target count or thresholds explicitly if selection is
short.

## A-012 — TiHuBird single-camera COLMAP preparation

Date: 2026-06-29

Question: How should the fixed TiHuBird video keyframes be reconstructed into a
COLMAP scene that preserves the raw sequence and is accepted by the actual
RT-GS/3DGS loader?

Chosen implementation: Reuse the operation sequence in the repository's
`convert.py`, but run its COLMAP commands explicitly in
`output/stage_a_tihubird_colmap/` because the script hard-codes an `input/`
source and writes/moves files in the scene root. Extract SIFT features from the
fixed `images_raw/` sequence with one shared OPENCV camera, use exhaustive
guided matching and incremental mapping, and retain the single successful
reconstruction. Do not silently fall back to per-frame intrinsics.

Run COLMAP image undistortion at full resolution because
`scene/dataset_readers.py` accepts only final SIMPLE_PINHOLE or PINHOLE cameras.
Install the 111 undistorted images as `images/` and only the required
`cameras.bin`, `images.bin`, and `points3D.bin` as `sparse/0/`. Preserve
`images_raw/` byte-for-byte and keep the prior raw copy of `images/` in the
ignored COLMAP log directory for recovery.

Verified result: COLMAP 3.14.0.dev0 registered 111/111 images in one model with
one final PINHOLE camera at 3827x2152, 84,989 points, 651,521 observations,
mean track length 7.665945, and mean reprojection error 1.098552 px. Final image
names, model image names, and keyframe manifest names match exactly. The RT-GS
loader read all 111 cameras and all finite sparse points from an isolated
read-only check scene.

Alternatives: Invoke `convert.py` after renaming/moving the formal raw images;
use per-frame cameras; train directly from the distorted OPENCV model; use
sequential matching instead of the repository's exhaustive flow; or overwrite
the raw keyframes in place.

Why: Explicit isolated commands preserve the fixed input and complete logs
while retaining the repository's established reconstruction flow. A shared
camera matches the single-device video capture. Undistortion is mandatory for
the loader rather than a subjective quality choice.

Paper fidelity: This is Stage A data preparation only. It changes no renderer,
loss, model representation, training schedule, or later-stage feature.

Impact: Later TiHuBird priors and training must use the undistorted 3827x2152
`images/`, not `images_raw/`. The sparse model contains a low-density distant
point tail, plausibly including background or reflected features; camera motion
is smooth, but this remains a visual-quality risk rather than grounds for
silently changing the camera model.

Required ablation: None. If later scene review exposes a reconstruction defect,
report it and discuss an explicit remapping configuration before changing the
shared-camera assumption.

## A-013 — Conditional StableNormal D-only baseline selection

Date: 2026-06-29

Question: Which matched TiHuBird Stage A run should be frozen as the verified
StableNormal D-only baseline before independent DiffusionRenderer experiments?

Chosen implementation: Conditionally accept the completed StableNormal
`lambda_mono=0.01` retry1 result as the Stage A D-only baseline. Preserve the
matched `lambda_mono=0` run as its ablation and exclude the interrupted first
`lambda_mono=0.01` attempt from all formal comparison.

Evidence: `output/stage_a_tihubird_priors_validate_111.log` reports 111 images,
111 priors, 111 manifest entries, and `STRICT_VALIDATION=PASS`.
`output/stage_a_tihubird_fullprior_1k_r2.log` reports 1,000 supervised/nonzero
steps, checkpoint save, and `Training complete.` The accepted 30k logs are
`output/stage_a_tihubird_30k_mono0_r2.log` and
`output/stage_a_tihubird_30k_mono001_r2_retry1.log`; their input-freeze records
are byte-identical. The first mono log ends around iteration 230 and its output
directory contains no checkpoint. Gate 9 evidence is
`output/stage_a_tihubird_30k_matched_metrics.txt` plus
`output/stage_a_tihubird_30k_matched_comparison.png`. Gate 9.5 evidence is the
six-view comparison and six crop audits under
`output/stage_a_tihubird_gate9p5/`.

Why: Retry1 improves L1 and PSNR over `lambda_mono=0` at all three matched
7k/15k/30k nodes. The global, six-view, and crop audits show no obvious RGB
degradation. The conclusion is conditional because D-only still entangles the
glass surface, reflection, interior bird, and background.

Alternatives: Select `lambda_mono=0`; use the interrupted first mono attempt;
or claim that this comparison solves transparent-scene decomposition.

Paper fidelity: This selects a Stage A baseline configuration; it changes no
renderer, representation, loss formula, or schedule.

Impact: The frozen baseline uses `lambda_mono=0.01`. Its normal/depth are a
foundation for later reflection/transmittance separation, not final transparent
geometry or a final bird reconstruction. Overall Stage A closure remains
deferred and Stage B/C/D remain unauthorized.

Required ablation: The matched `lambda_mono=0` run is complete and retained.

## A-014 — DiffusionRenderer raw-prior experiment boundary

Date: 2026-06-29

Question: What DiffusionRenderer evidence may be recorded before a normal
adapter or any training integration exists?

Chosen implementation: Preserve the StableNormal baseline unchanged and treat
DiffusionRenderer as an independent Stage A prior extension. DR-1A generated a
24-frame raw pilot from final undistorted TiHuBird training images under
`output/stage_a_tihubird_dr_pilot_24/`. DR-1B generated all 111 real frames
under `output/stage_a_tihubird_dr_raw_111/`, with `normal`, `depth`,
`basecolor`, and `diffuse_albedo` raw PNGs. Its manifest records five chunks,
111 real mappings, and nine final-chunk padding slots separately.

Alternatives: Reuse the earlier GLINT inputs/outputs; overwrite
`data/TiHuBird/normal_priors/`; assume a normal coordinate mapping; or connect
raw buffers directly to a loss.

Why: Final-image-domain provenance and explicit padding separation are required
before semantic conversion. Keeping artifacts isolated prevents accidental
replacement of the accepted StableNormal baseline.

Paper fidelity: Raw-prior generation changes no RT-GS model, renderer, loss, or
training schedule.

Impact: No normal adapter exists, no normal axis mapping is selected or
validated, no DiffusionRenderer prior loss is connected, and no training has
used these raw outputs. `basecolor` and `diffuse_albedo` remain audit-only.

Required ablation: None at the raw-generation step. Adapter/axis validation and
any later training comparison must occur only on the independent
`feature/stage-a-diffrender-priors` branch.

## A-015 — Stop DiffusionRenderer C03 at 15k and close Stage A

Date: 2026-06-30

Question: Does the independent DiffusionRenderer C03 D-only experiment warrant
continuing from 15,000 to 30,000 iterations before Stage B begins?

Chosen implementation: Stop C03 at 15,000 iterations and do not run C03 30k.
Keep the StableNormal `lambda_mono=0.01` retry1 30k result as the frozen Stage A
D-only baseline. Close Stage A under explicit user authorization and advance the
project to Stage B.

Evidence: `output/stage_a_tihubird_drnormal_c03_15k.log` records a completed
15,000-step run, saves at 7,000/15,000, checkpoints at
7,000/10,000/15,000, and final train metrics L1
`0.021616848371922973`, PSNR `26.401429367065433`. The C03 checkpoint reports
format `rtgs_stage_a`, iteration 15,000, and 346,118 Diffuse surfels with finite
core parameter and densification tensors. The 15,000-step PLY and
`comparison_vs_stablenormal_15k.png` are present. C03 trained stably and did not
show the bird being swallowed by the front glass surface, but its D-only
RGB/normal/depth audit did not establish a clear advantage over StableNormal.

Alternatives: Continue C03 D-only training to 30,000 iterations; replace the
StableNormal baseline with C03; or defer Stage B indefinitely for more D-only
prior comparisons.

Why: Extending a configuration that still has no Reflection Gaussian cannot
directly test the intended value of the DiffusionRenderer front-interface prior.
The existing 15,000-step evidence is sufficient for the D-only question and
does not justify displacing the verified StableNormal baseline.

Paper fidelity: This is an experimental-stage and initialization decision. It
does not change the representation, rendering formulas, losses, or training
schedule.

Impact: No C03 30k run is authorized. Stage A is complete; the StableNormal
baseline and both output trees remain read-only. Stage B may test reflection but
must not reinterpret C03 as a proven best D-only prior.

Required ablation: A matched StableNormal-versus-C03 Stage B comparison may be
run later if needed, but it is explicitly not part of the first Stage B
experiment and no second training is started now.

## B-001 — Stage B reflection hypothesis starts from the C03 15k D candidate

Date: 2026-06-30

Question: Which existing Diffuse state may initialize the first Stage B
reflection experiment, and what conclusion may be drawn from that choice?

Chosen implementation: Use
`output/stage_a_tihubird_drnormal_c03_15k/chkpnt15000.pth` as the fixed Diffuse
initialization candidate for the first Stage B reflection hypothesis. Preserve
its associated `point_cloud/iteration_15000/` as read-only recovery/export
evidence. A new Reflection field must be independent from D and must not modify
the Stage A output directory.

Alternatives: Initialize D from the frozen StableNormal 30k checkpoint; train a
fresh D; extend C03 to 30k; or launch both C03- and StableNormal-initialized Stage
B runs together.

Why: C03 is a plausible interface-prior candidate whose intended benefit can be
tested only after an actual Reflection branch exists. Using it for the first
hypothesis isolates that question without spending another D-only long run or
starting two experiments simultaneously.

Paper fidelity: The master plan requires independent D and R fields but does not
prescribe which accepted D checkpoint initializes the first reflection
experiment. This is an engineering experiment choice.

Impact: C03 15k remains only an initialization candidate, not a new D-only
baseline. The frozen StableNormal branch
`baseline/stage-a-stablenormal` and output
`output/stage_a_tihubird_30k_mono001_r2_retry1/` remain unchanged. Stage B is
limited to Reflection Gaussian, differentiable tracing, reflection rays,
microfacet reflection, and the minimum specular constraint needed for that
hypothesis.

Required ablation: If the first Stage B result warrants it, compare C03 and
StableNormal under a matched Stage B protocol in a later explicitly authorized
experiment. Do not run that comparison concurrently with the first smoke.
