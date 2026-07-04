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

## B-002 — Gaussian raytrace value, hit, miss, and background semantics

Date: 2026-06-30

Question: What exactly do the Stage B Gaussian ray tracer outputs mean, and how
is a reflection miss converted into a color without consulting training images?

Chosen implementation: Ray directions are normalized inside the public
`raytrace` wrapper. Reflection surfels are two-sided. A candidate contributes
only when the ray is not parallel to its plane, its normalized ray distance is
positive, and its local elliptical radius satisfies `r2 <= 9`, i.e. the fixed
three-standard-deviation support used to build its finite AABB. Its effective
opacity is:

```python
a_i = sigmoid(opacity_raw_i) * exp(-0.5 * r2_i)
```

Contributions are sorted by increasing ray distance and composed front to back:

```python
w_i = a_i * product(1 - a_j for j before i)
color = sum(w_i * sigmoid(color_raw_i))
alpha = sum(w_i)
expected_depth = sum(w_i * t_i) / max(alpha, 1e-8)
hit = alpha > 1e-4
```

`color` is foreground-premultiplied and contains no background. A miss returns
exact zero color, alpha, and depth, with `hit=False`. The generic ray tracer
never reads a ground-truth image, target pixel, camera image, or training-image
color. `--ray_background scene` means exactly the deterministic renderer
background already selected from `dataset.white_background`: `[1,1,1]` for a
white-background scene and `[0,0,0]` otherwise. Reflection shading expands the
raytrace foreground to a background-composited radiance with:

```python
Cr = color + (1.0 - alpha) * renderer_background
```

Alternatives: Return background-composited color from `raytrace`; sample a GT
pixel on miss; use an image-dependent environment lookup; use an unbounded
Gaussian support; or silently truncate a fixed top-K candidate list.

Why: Premultiplied foreground plus explicit alpha is the reusable contract in
the master plan and keeps background policy outside the tracer. A fixed renderer
background is reproducible and cannot leak the training target. Three-sigma
support provides a finite conservative AABB, while two-pass candidate allocation
will fail instead of silently dropping intersections.

Paper fidelity: The master plan specifies color/alpha/expected-depth/hit and a
scene-scaled origin epsilon but does not publish cutoff, hit threshold, two-sided
behavior, or miss compositing. These are Stage B engineering definitions.

Impact: `reflection_color.png` visualizes `Cr`; raw premultiplied raytrace color
remains separately available for tests. Expected depth is distance along the
normalized reflection ray, not camera-z depth. CUDA failure is fatal for
training; the Python brute-force implementation is test-oracle-only.

Required ablation: Compare the three-sigma cutoff and hit threshold if reflection
coverage is visibly unstable. Never replace miss background with target-image
color in an ablation.

## B-003 — Stage A G-buffer decoding and full Stage B microfacet composite

Date: 2026-06-30

Question: How are Stage A premultiplied buffers decoded for BRDF evaluation, and
what numerical and alpha/background rules define the Stage B final image?

Chosen implementation: Let `A` be the Stage A diffuse alpha and `B` the constant
renderer background. The Stage A color rasterizer has already composited its
foreground over `B`, whereas roughness, f0, and ks were rendered with a zero
background. For valid pixels, recover conditional surface quantities with:

```python
denom = clamp(A, min=1e-4)
Cd_premul = Cd_rendered - (1.0 - A) * B
Cd_surface = clamp(Cd_premul / denom, 0.0, 1.0)
roughness = clamp(roughness_premul / denom, roughness_min, 1.0)
f0 = clamp(f0_premul / denom, 0.0, 1.0)
ks = clamp(ks_premul / denom, 0.0, 1.0)
```

The Stage A normal is used directly because it is already normalized and
face-forward. It is never alpha-divided or alpha-normalized again.

The optional PBRT roughness remap uses the documented PBRT polynomial before
GGX evaluation:

```python
x = log(max(roughness, 1e-3))
alpha_ggx = 1.62142 + 0.819955*x + 0.1734*x**2 \
            + 0.0171201*x**3 + 0.000640711*x**4
```

When remapping is disabled, `alpha_ggx = roughness`. In both modes it is
clamped to `[roughness_min, 1]`. Stage B uses `wi=d_ref`, `wo=-d_cam`, Schlick
Fresnel, the master-plan GGX/Trowbridge-Reitz D, and separable exact Smith GGX:

```python
G1(NoX) = 2*NoX / (NoX + sqrt(alpha_ggx**2
                               + (1-alpha_ggx**2)*NoX**2))
G = G1(NoV) * G1(NoL)
fr = D * G * F / (4*NoV*NoL + 1e-6)
wr = fr * NoL
```

Raw `NoV` and `NoL` must both be strictly positive; otherwise D/F/G/fr/wr and
the reflection contribution are zero. Inside that gate, dot products are
clamped to `[1e-6,1]`, square-root and BRDF denominators use `1e-6`, D is clamped
nonnegative without an artificial upper cap, F and G to `[0,1]`, and fr/wr
nonnegative without an artificial upper cap. Any non-finite intermediate is a
hard error. The full surface and image composite is:

```python
C_surface = (1.0 - ks) * Cd_surface + ks * wr * Cr
C = A * C_surface + (1.0 - A) * B
```

`C` is clamped to `[0,1]` only at the final RGB output used by the existing
sRGB-domain reconstruction loss and image writer. D/F/G/fr/wr remain available
unclamped except for the bounds above and are visualized with display-only
clamping.

Alternatives: Feed premultiplied material maps directly to GGX; alpha-divide the
normal; omit the background subtraction from Cd; use split-sum; clamp BRDF
weights to one; or add reflection after background compositing without the
diffuse alpha gate.

Why: Conditional material values are required for a stable physical BRDF at
partially covered pixels. Separating the surface composite from the final
alpha-over prevents background pixels from receiving reflection and preserves
the existing Stage A renderer-background convention.

Paper fidelity: D, F, G, fr, wr, and the Stage B reflection-only composition
follow the master plan. PBRT mapping, safe premultiplied decoding, exact Smith
form, epsilons, and output clamp are engineering details not fully published.

Impact: Checkpoints must store `roughness_remap`, `roughness_min`, the numerical
epsilon, the material-alpha threshold, and `bsdf_weight_mode`, which is fixed to
`brdf_times_cosine` for this Stage B path.

Required ablation: PBRT remap on/off is required if roughness stability or
reflection sharpness materially changes. Split-sum remains forbidden in Stage B
and is reserved for the explicit Stage E ablation.

## B-004 — Valid diffuse surface gate and reflection-ray generation

Date: 2026-06-30

Question: Which diffuse pixels may create reflection rays and enter the BRDF?

Chosen implementation: A pixel is a valid Stage B diffuse surface only when all
of the following hold:

```text
diffuse alpha > 1e-4
depth is finite and strictly positive
position is finite
normal is finite and has norm > 1e-6
roughness/f0/ks buffers are finite
the face-forward normal has dot(normal, wo) > 0
```

For valid pixels only:

```python
d_cam = normalize(position - camera_center, eps=1e-8)
wo = -d_cam
d_ref = normalize(d_cam - 2*dot(d_cam, normal)*normal, eps=1e-8)
ray_eps = ray_epsilon_scale * scene_radius
origin = position + ray_eps * d_ref
```

The default `ray_epsilon_scale` is `1e-4`, and `scene_radius` is
`Scene.cameras_extent`. Invalid pixels are compacted out before ray tracing and
BRDF evaluation, then receive exactly the renderer background in the final
alpha-over image. No placeholder reflection result is generated for them.

Alternatives: Trace every image pixel; use diffuse alpha as a transparent mask;
reflect an unnormalized direction; offset along the normal rather than the ray;
or admit non-finite/zero normals.

Why: Compaction prevents undefined rays and unnecessary BVH work. The direction
and scene-scaled epsilon match the master-plan convention. Diffuse alpha remains
a surface-validity signal only and is not reinterpreted as a transparent-object
mask.

Paper fidelity: Reflection direction and epsilon follow the master plan. The
finite-value and alpha thresholds are engineering safety gates.

Impact: Debug metadata must report valid-ray count and fraction. A view with no
valid surface pixels returns the constant renderer background and empty, real
reflection buffers without calling the tracer.

Required ablation: Revisit the diffuse-alpha threshold only if edge coverage is
visibly unstable; do not couple it to the Reflection hit threshold implicitly.

## B-005 — Independent Reflection initialization, schedule, densification, and checkpoint

Date: 2026-06-30

Question: How does the first Stage B run restore C03 Diffuse state while creating
and maintaining a separate Reflection field?

Chosen implementation: `--diffuse_init_checkpoint` is accepted only for a new
Stage B run and must contain `format=rtgs_stage_a`; `--start_checkpoint` resumes
only `format=rtgs_stage_b`, and the two options are mutually exclusive. The C03
Diffuse model, optimizer, exposure optimizer, and densification state are
restored at global iteration 15,000 without modifying the source artifact.
Reflection starts at local step zero and uses its own optimizer and scheduler.

The mandatory `random_bbox` initializer samples seeded uniform positions over
the complete Diffuse xyz AABB, without quantile cropping. It requires an explicit
surfel count and records seed, actual count, and exact bbox. Rotations are seeded
normalized random quaternions, initial color is `0.5`, initial opacity is `0.01`,
and both tangent scales are half the nominal uniform-volume spacing:

```python
spacing = cbrt(max(product(bbox_extent), eps) / reflection_count)
scale_u = scale_v = 0.5 * spacing
```

Reflection densification accumulates per-surfel xyz gradient norm and ray-hit
weight/count. It clones small high-gradient surfels, splits large high-gradient
surfels in their tangent plane, and prunes low-opacity or persistently unhit
surfels only after an explicit warmup. These tensors and all R optimizer state
remain separate from D. Topology changes increment an R topology version and
force a BVH rebuild; ordinary parameter updates require a BVH refit.

Stage B checkpoints use a versioned `rtgs_stage_b` dictionary with separate
`diffuse` and `reflection` namespaces, global and reflection iterations, both
model/optimizer/densification states, initialization provenance and source hash,
renderer/raytracer/BRDF/loss configuration, and RNG state. Derived BVH nodes are
not serialized and must be rebuilt fail-closed on resume. PLY exports use
separate Diffuse and Reflection paths under the new Stage B output directory.

Alternatives: Reuse D tensors as R; initialize R from the C03 PLY; reset D's
optimizer; use a shared global R scheduler; crop the AABB to a foreground box;
or serialize stale BVH nodes.

Why: The master plan requires physically separate fields and state. The Stage A
checkpoint, unlike PLY, contains the optimizer/exposure/densification state.
Local R time prevents a global 15k start from skipping all Reflection warmup and
densification. Full AABB initialization retains the external environment.

Paper fidelity: Independent D/R fields are required. Initialization values,
local schedule, densification statistics, and checkpoint layout are engineering
definitions.

Impact: The first smoke may use a small explicit R count but is structural only.
The large C03 AABB can make uniform initialization sparse; its actual reflection
coverage must be inspected by the user before any quality conclusion.

Required ablation: Reflection count/initial scale and random_bbox versus the
optional uniform-grid mode may be ablated later. Do not launch a concurrent
StableNormal-initialized Stage B run during the first experiment.

## B-006 — CUDA LBVH division of responsibility and fail-closed execution

Date: 2026-06-30

Question: Which parts of Stage B ray tracing live in the custom CUDA extension,
how is differentiability retained, and what happens when the extension is
unavailable?

Chosen implementation: Build a complete balanced hierarchy over Morton-sorted
Reflection leaf AABBs on CUDA tensors. Each leaf AABB conservatively encloses
the oriented three-sigma elliptical surfel support. The custom C++/CUDA
extension performs stack-based ray/AABB traversal in two passes: the first
counts every candidate and the second allocates/fills the exact flat candidate
array. It has no fixed top-K truncation. Parameter updates refit all leaf and
internal AABBs; Reflection topology-version changes rebuild Morton ordering and
the hierarchy.

Exact two-sided plane intersection, local ellipse evaluation, Gaussian opacity,
depth sorting, and front-to-back compositing operate on CUDA PyTorch tensors
gathered from the BVH candidates. Candidate topology is discrete, while all
continuous hit math remains under autograd. This supplies gradients to xyz,
rotation, scaling, opacity, color, ray origin, and ray direction away from
discrete candidate/support boundaries.

The production `raytrace` API requires CUDA model/ray tensors and a successfully
loaded or JIT-compiled extension. Compilation, loading, input, or traversal
failure raises immediately. It never imports or invokes the brute-force oracle
as a fallback. `raytracer/reference.py` remains test-only.

Alternatives: Implement a custom analytical backward in CUDA; traverse all
surfels; use an OptiX SDK dependency; use a fixed candidate cap; build a CPU BVH;
or silently call the Python oracle on failure.

Why: Separating discrete acceleration from differentiable exact CUDA-tensor math
keeps the first implementation auditable and permits direct oracle and
finite-difference checks. Two-pass allocation preserves correctness, and
fail-closed behavior prevents an accidental production complexity explosion.

Paper fidelity: The master plan requires CUDA/OptiX production acceleration,
chunking, synchronization, and all listed gradients. The Morton-balanced tree
and autograd division are engineering choices.

Impact: The extension compiled successfully with PyTorch 2.0.1/CUDA 11.8 on the
current RTX 3090. Tests matched the brute-force oracle and finite differences
for all required gradient inputs. Performance and memory on full-resolution
TiHuBird rays remain unverified until the user smoke.

Required ablation: None for correctness. Profile candidate counts, chunk size,
refit/rebuild time, and peak memory before selecting a long-run configuration.

## B-007 — ASCII-only CUDA JIT staging for non-ASCII repository paths

Date: 2026-06-30

Question: How can the fail-closed CUDA LBVH JIT compile reproducibly when the
repository path contains non-ASCII characters and the operator's Python process
uses an ASCII locale for text files?

Chosen implementation: Continue to prefer an explicitly installed
`rtgs_bvh_cuda` module. When JIT compilation is required, read the two canonical
CUDA source files from the repository and atomically stage their exact bytes
under this default ASCII-only cache layout:

```text
/tmp/rtgs-bvh-jit-<uid>/<build-fingerprint>/sources/
/tmp/rtgs-bvh-jit-<uid>/<build-fingerprint>/build/
```

The fingerprint covers both source contents plus the PyTorch version, PyTorch
CUDA version, Python ABI tag and executable, CUDA home, and C++ ABI setting.
Pass only the staged ASCII source paths and the explicit ASCII build directory
to `torch.utils.cpp_extension.load`. `RTGS_BVH_JIT_ROOT` may override the cache
root for controlled environments, but a resolved non-ASCII override is rejected
before compilation. Compilation and loading remain fail-closed, with no
brute-force production fallback.

Alternatives: Require the operator to rename or move the repository; require a
UTF-8 locale; install the extension manually before every run; or retry through
the Python brute-force oracle.

Why: The first user-operated TiHuBird smoke restored D, initialized R, and then
failed before its first optimization step because PyTorch wrote a Ninja file in
the repository-derived build path using ASCII encoding. Keeping generated build
inputs in a content-addressed ASCII path removes that locale/path interaction
without weakening the tracer's production contract or changing canonical CUDA
sources.

Paper fidelity: This is a build-path and reproducibility decision. It changes no
ray, surfel, BRDF, compositing, loss, or training semantics.

Impact: The failed `smoke_2` directory is evidence of an aborted attempt, not a
training result; it contains no Stage B checkpoint or D/R PLY save. A forced-C-
locale compile reported `ANSI_X3.4-1968` and successfully built and loaded the
hashed CUDA module from the ASCII staging directory. The operator must use a new
output path for the retry so the failed attempt is preserved.

Required ablation: None. Retain an ASCII-locale compile/load regression and the
existing CUDA/oracle/gradient tests.

## B-008 — Debug-only raytrace observability after the first successful smoke

Date: 2026-06-30

Question: How should Stage B distinguish useful Reflection coverage from an
over-broad random field, and preserve physically meaningful debug values when
8-bit visualization makes a nonzero contribution black or an unbounded GGX D
term white?

Chosen implementation: Add an explicit diagnostics request that is disabled on
ordinary training renders and enabled only for fixed debug/offline Stage B
renders. Reuse the exact LBVH offsets to report per-ray AABB candidate counts,
and report per-ray exact plane/ellipse intersection counts from the existing
differentiable intersection pass. Do not change candidate selection, hit
thresholds, compositing, gradients, or optimizer inputs.

Diagnostics record chunk count, Reflection surfel count, BVH rebuild/refit
deltas, raytrace wall time, CUDA time for BVH synchronization, traversal, and
exact intersection/compositing, and peak allocated CUDA memory. Timing uses
CUDA events and synchronizes only when diagnostics are explicitly requested.
The ordinary training render remains unsynchronized by this instrumentation.

Keep every existing physical-scale debug PNG unchanged. Add display-only
companions:

```text
reflection_contribution_vis.png  linear scale by finite valid-pixel p99
microfacet_D_log.png             log1p scale by finite valid-pixel p99
ray_candidate_count.png          linear scale by valid-ray p99
ray_exact_intersection_count.png linear scale by valid-ray p99
```

Every display scale and raw finite/count/min/mean/p50/p95/p99/max statistic is
written to `reflection_metadata.json`. Scaling never feeds back into rendering,
losses, checkpoints, or training. Statistics use only renderer outputs and
raytrace bookkeeping; no ground-truth, target pixel, or training-image color is
read to define a scale.

Alternatives: Replace the physical PNGs with normalized images; globally enable
synchronizing timers during training; infer acceleration quality from the final
hit mask alone; add a second CUDA traversal; or change the Reflection
initialization before measuring it.

Why: The successful user `smoke_2_retry1` completed two steps and produced real
Reflection color/alpha/depth plus a valid Stage B checkpoint, but 128,579 of
128,582 valid rays passed the final alpha hit threshold. Its physical
`reflection_contribution.png` quantized to black while 6,632 final-image channel
values still increased by one 8-bit level, and `microfacet_D.png` saturated to
white. These observations prove the path is not a dummy, but the current maps
cannot reveal whether the nearly complete coverage is efficient or over-broad.

Paper fidelity: This is Stage B debug and performance instrumentation only. It
does not modify the RT-GS representation, reflection rays, Gaussian support,
BRDF, final composite, losses, schedules, or checkpoint contract.

Impact: A new user-operated one-step resume smoke is required after the
instrumentation passes synthetic/CUDA regression. It must use a new output
directory and resume the read-only `smoke_2_retry1/chkpnt15002.pth`; Codex does
not run that TiHuBird command.

Required ablation: None. If candidate p50/p95 approaches the full Reflection
count or exact intersections remain unexpectedly dense, measure an explicitly
documented Reflection scale/count initialization change in a later short pilot;
do not silently alter the current initialization first.

## B-009 — Canonical CPU ByteTensor contract for checkpoint RNG state

Date: 2026-06-30

Question: How should Stage B restore CPU and CUDA RNG states when model and
optimizer tensors are loaded directly onto CUDA with `map_location="cuda"`?

Chosen implementation: RNG byte tensors are logically device-independent
checkpoint metadata. `capture_rng_state` continues to serialize the CPU PyTorch
state and the per-device CUDA states as the ByteTensors returned by PyTorch.
Before calling either `torch.set_rng_state` or `torch.cuda.set_rng_state_all`,
the restore path validates that every state is a uint8 tensor and moves it to a
contiguous CPU ByteTensor. Model, optimizer, and densification tensors continue
to obey the requested checkpoint `map_location`.

Do not change the Stage B checkpoint version: existing checkpoints already
contain the correct bytes, and only the loader incorrectly allowed
`map_location` to move those bytes onto CUDA. Invalid or non-uint8 RNG entries
fail closed rather than being cast silently.

Alternatives: Load the complete checkpoint on CPU and manually migrate every
model/optimizer tensor; disable RNG restoration on production resume; store RNG
bytes outside the checkpoint; or cast arbitrary tensors to uint8.

Why: The first user-operated B-1d resume read the 15,002 checkpoint successfully
but stopped before iteration 15,003 because `torch.load(...,
map_location="cuda")` moved `rng_state["torch"]` to CUDA and
`torch.set_rng_state` requires a CPU ByteTensor. The partial resume output has no
checkpoint or D/R PLY and remains failed evidence.

Paper fidelity: This is deterministic checkpoint-resume plumbing only. It
changes no model state, random sequence bytes, renderer, ray tracer, BRDF, loss,
schedule, or Stage B diagnostic definition.

Impact: Add an on-disk CUDA-map-location round-trip that restores RNG instead of
disabling it, plus a read-only restoration check of the user checkpoint. The
operator retry must use a new output directory and preserve the failed attempt.

Required ablation: None.

## B-010 — Close B-1d diagnostics and authorize only a 100-step health pilot

Date: 2026-06-30

Question: Does the successful diagnostic resume show an LBVH or BRDF/debug
failure that must be corrected before a short Reflection health pilot?

Chosen implementation: Accept the B-1d observability gate using the
user-operated output
`output/stage_b_tihubird_reflection_dr_c03_resume_diag_15003_retry1/`.
Its checkpoint is `rtgs_stage_b` at global iteration 15,003 and Reflection local
step 3, with 346,118 Diffuse and 4,096 Reflection surfels and no non-finite D/R
checkpoint state.

The fixed debug view contains 128,582 valid rays. LBVH AABB candidate counts
have p50/p95/p99 `111/205/256`; exact plane/ellipse intersections have
p50/p95/p99 `28/64/76`. Candidate p99 is 6.25% of the 4,096-surfel field, and
the mean exact/candidate fraction is about 25.8%. Raytrace wall time is about
152 ms for 128,582 rays; traversal and exact intersection/compositing CUDA times
are about 69.6 ms and 75.7 ms. Incremental peak allocation is 333,931,008 bytes,
about 319 MiB.

The raw microfacet D values lie in the narrow finite interval `1.270–1.276`.
Its nearly white physical/log display is therefore an honest low-variance map,
not a visualization or BRDF failure. Reflection contribution remains finite and
nonzero with mean `6.73e-5`, p99 `3.79e-4`, maximum `8.40e-4`, and nonzero
fraction about 99.998%; its p99-scaled companion is spatially informative.

Authorize only a controlled additional 100-step `lambda_spec=0` health pilot
from this exact 15,003/3 checkpoint. Do not change Reflection count, initial
scale, initialization mode, BRDF, BVH, densification, or any optimizer/loss/data
hyperparameter. Debug nodes must be exactly +25/+50/+75/+100, and checkpoint plus
independent D/R PLY must be written only at +100. If the current CLI cannot
express that schedule exactly, the pilot is blocked and no code workaround or
training run is authorized.

Alternatives: Treat near-complete final alpha hits as BVH brute force; change R
scale/count before measurement; start a long run; enable a fabricated mask; or
advance to Stage C/D.

Why: Candidate density, exact-intersection density, timing, and memory show that
the LBVH prunes the 4,096-surfel field materially and does not degenerate to a
full scan. The remaining question is short-horizon numerical and optimization
health, not acceleration correctness or 100-step image quality.

Paper fidelity: This closes a Stage B engineering diagnostic gate and defines a
short experiment boundary. It changes no representation, rendering formula,
loss, schedule, or checkpoint contract.

Impact: Stage B remains unaccepted. A real 111/111 manual soft-mask set and
`L_spec` behavior are still unverified. Stage C and Stage D remain forbidden.

Required ablation: None before the controlled health pilot.

## B-011 — Group Reflection candidate-parameter gradients by surfel ID

Date: 2026-07-01

Question: How should Stage B remove the measured Reflection candidate
advanced-index backward bottleneck without changing raytrace, shading, loss, or
training semantics?

Chosen implementation: Pack each Reflection surfel's raw `xyz` (3), quaternion
(4), 2D log-scale (2), opacity logit (1), and color logits (3) into one
13-channel view. A single custom CUDA gather preserves the LBVH candidate array
exactly, including order and repeated IDs. The same pointwise activations used by
the model are applied after gather: quaternion normalization and matrix
conversion, scale exponential, and opacity/color sigmoid. Exact plane/ellipse
intersection, depth sort, Gaussian alpha, front-to-back composite, BRDF, loss,
ray chunking, and all model/training settings are unchanged.

During backward, each chunk creates one candidate-position array, radix-sorts
candidate ID plus original position once, and launches one block per Reflection
surfel. That block binary-searches its ID range and reduces all 13 gradient
channels together in shared memory before one write per surfel/channel. No
candidate performs a contended global atomic scatter, and all attributes reuse
the same ID grouping. This custom path covers only Reflection parameter gather;
ray origin and direction remain in the ordinary differentiable exact-hit graph,
so gradients to Diffuse position/normal are not detached.

Alternatives: Replace advanced indexing with `index_select`/`gather`; run five
independent `index_add_` reductions; atomically scatter every candidate; alter
the candidate set, chunk size, Reflection count/scale, or mask rays.

Why: The original matched Nsight interval contained 160 long
`indexing_backward_kernel` calls from five parameter families across 32 chunks.
They totaled 49.356 s and occupied 97.8% of the 50.461 s global-15,004 step. In
the matched profile after commit `c87bb66`, no long indexing backward kernel
remains: nine unrelated short calls total 0.430 ms. The 32 grouped reducers total
253.612 ms, the complete R backward envelope is 0.547 s, and the step is 1.572
s. Global 15,004 and 15,005 losses are bit-identical to the old profile.

Paper fidelity: This changes only the implementation of mathematically
identical repeated-index gradient accumulation for Reflection parameters. It
does not change the continuous renderer or any Stage B experiment semantics.

Impact: Repeated-ID forward/gradient tests, complete raytrace tests, a full
Stage B training-style loss comparison, and all existing regressions pass (73
tests). The external 200 ms sampler observed a 15,067 MiB device-total peak in
the new profile, compared with an earlier separately observed 10,627 MiB total.
Because those memory observations are not one identical measurement method, the
difference is recorded as an unresolved short-pilot risk rather than a proven
allocation regression. Do not change model/training settings before measuring
it in the next fresh health pilot.

Required ablation: None. Stage B remains unaccepted; complete the controlled
health pilot from the original 15,003/3 checkpoint and later verify the real
111/111 manual soft-mask path before any stage transition.

## B-012 — Accept the post-repair 100-step run as a health pilot only

Date: 2026-07-01

Question: After removing the candidate-gradient bottleneck, does 100-step
continuous training remain numerically and operationally healthy, and what does
the allocator evidence permit next?

Chosen implementation: Run exactly one fresh pilot from the original read-only
global 15,003 / Reflection-local 3 checkpoint through global 15,103 / local 103.
Keep resolution 8, 4,096 Reflection surfels, chunk size 4,096, all renderer,
loss, optimizer, scheduler, densification, and RNG state unchanged. Keep
`lambda_spec=0` and load no mask. Use the existing 15,025/15,050/15,075/15,100
debug/evaluation nodes plus the accepted final 15,103 debug behavior; save only
the final checkpoint and independent D/R PLYs.

The completed run has ordinary-step p50/p95/mean wall times
`0.939/1.092/0.944 s`. D/R counts stay 346,118/4,096, every training iteration
uses 32 candidate-gradient reductions, and no densify/prune count change occurs.
All TensorBoard scalars, renderer checks, five diagnostic metadata sets, and the
final recursive checkpoint inspection are finite. Candidate p50/p95/p99 grows
from `126/227/285` to `146/265/341`; exact intersections grow from `33/72/86`
to `38/83/109`. Reflection contribution remains essentially fully nonzero and
increases rather than collapsing. Final valid-surface ks is centered at 0.09726
with range 0.08796--0.10640 and no values below 0.01 or above 0.9.

Exact PyTorch allocator peaks, reconstructed across the diagnostic timer's
intentional peak-stat resets, are 18,094,459,392 allocated bytes and
20,333,985,792 reserved bytes. Current live allocated memory varies with view
and is not monotonic, so the evidence does not indicate a retained live-tensor
leak. Reserved memory is nearly monotonic because the caching allocator retains
larger temporary blocks, reaching 20.33 GB and leaving limited RTX 3090
headroom. Preserve this as a long-run risk; do not respond by changing chunk
size, resolution, Reflection count, model math, or training settings in this
pilot decision.

Alternatives: Resume the aborted pre-repair health directory; extend directly
into a long run; classify cached reserved growth as a proven live-tensor leak;
or hide the allocator risk because no OOM occurred.

Why: The pilot meets the stated short-run health conditions: no crash/OOM or
NaN/Inf, final checkpoint/PLY save succeeds, ks does not collapse, reflection
contribution does not vanish, candidate density does not grow by an order of
magnitude, and performance remains in the post-repair regime. The reserved
headroom is nevertheless too material to ignore when deciding any later long
run.

Paper fidelity: This is a Stage B operational-health classification and memory
observation. It changes no representation, rendering equation, loss, data, or
training schedule.

Impact: Preserve
`output/stage_b_tihubird_reflection_dr_c03_health100_candidate_reduce_g15003_15103_retry1/`
as verified short-pilot evidence. Its final checkpoint SHA-256 is
`6f43ff1335f99e03f8ee08e4575ad4c91b29189cbe23a67942b23557adb87354`.
Do not continue training from it without separate authorization. Stage B remains
unaccepted: the next independent gate is a real 111/111 manual soft-mask set and
verified `L_spec`; Stage C and Stage D remain forbidden.

Required ablation: None for this health classification.

## B-013 — DR geometry may seed review proposals, never mask supervision

Date: 2026-07-01

Question: How may the existing DiffusionRenderer artifacts reduce manual
glass-enclosure annotation work without silently becoming `L_spec` supervision?

Chosen implementation: Add a fail-closed, offline audit/proposal tool that is
independent of the training mask loader. It validates the 111 ordered source
stems and hashes, every real DR RGB/normal/depth/basecolor/diffuse_albedo path,
format and size, the original generation validation, and all nine padding
records. Padding slots 111--119 are recorded but excluded. A missing required
artifact, source/hash/stem mismatch, unexpected image mode or size, invalid
normal, or failed generation provenance aborts the run; nothing is silently
resized to repair an audit failure.

DR normal is decoded as `rgb / 127.5 - 1`, unit-normalized, and used only through
dot-product continuity and boundary magnitude. Its axis convention remains
unconfirmed, and no component is assigned a world/camera semantic meaning. DR
depth remains a per-frame relative RGB visualization: it is never treated as
COLMAP-scale depth or compared numerically across views. The stored DR RGB is
used in the native 704x384 domain; source-sized review products are produced
only after the exact 3827x2152 / 704x384 relation has passed audit. The original
generation validation reports exact RGB alignment. Replaying PIL bilinear under
the current Pillow build differs by at most two uint8 levels, which is recorded
rather than mistaken for a stem or geometry mismatch.

For only the nine fixed representative views, generate a connected convex
enclosure proposal from multiple cues: RGB boundary/highlight evidence, DR
normal continuity and discontinuity, within-frame DR depth regions/boundaries,
optional DR basecolor evidence, centrality, connected-component filtering,
hole/small-fragment suppression, and a feathered signed-distance edge. No one
normal/depth threshold defines glass. `diffuse_albedo` is audited and exposed as
auxiliary provenance but is not required by this first proposal score.

Outputs live under `proposal_soft/<stem>/` and are automatic, read-only review
evidence. The review contract defines future `review_queue/` as human working
copies and future `reviewed_soft/` as human-confirmed 3827x2152 single-channel
uint8 masks whose stem matches RGB, exterior is zero, interior glass coverage
is 255, and only edges may transition softly. Bird/background visible through
the enclosure are not separate mask targets. This run deliberately creates no
`reviewed_soft`, no formal training manifest, and no accepted mask directory.

Alternatives: Threshold normal smoothness alone; threshold relative depth alone;
use Stage B D G-buffer geometry as the primary evidence; generate all 111
proposals before checking quality; or point `--specular_masks` at automatic
drafts.

Why: The nine-view contact sheet shows DR depth gives a strong coherent
enclosure block while DR normal and RGB supply complementary side/edge evidence.
The proposals visually cover the projected glass enclosure in all nine views,
but texture behind glass conservatively triggers bird-inclusion risk on every
view. That is useful triage evidence, not proof of annotation correctness.

Paper fidelity: This is an offline annotation-assistance tool. It does not
change Stage B rendering, losses, checkpoints, training inputs, field counts,
or full-image RGB semantics.

Impact: Stop for user review of the nine proposals. Do not expand to 111, enable
`lambda_spec`, train, or enter Stage C/D until separately authorized. Stage B
remains unaccepted.

Required ablation: Human review of the nine representative views before any
batch proposal expansion.

## B-014 — Freeze the approved proposal method for the 111-view review package

Date: 2026-07-01

Question: How should the accepted nine-view automatic proposal method be
expanded to every real TiHuBird view without per-frame tuning or accidentally
creating training supervision?

Chosen implementation: Freeze the proposal method and parameters at commit
`a2ee73212246674b97993ac7affa0e139e44dd5b`. Do not change the native candidate
score, thresholds, morphology, connected-component/hull selection, feathering,
or uncertainty definition. Run the same function once for each audited real
stem 000000--000110 in order. The review-packaging implementation is separate
from `utils/dr_mask_proposal.py`; it may validate, hash, rank, crop, and compose
contact sheets but never changes proposal pixels.

The native 704x384 soft signed-distance field is lifted to 3827x2152 with the
existing `cv2.INTER_LINEAR` scalar mapping after the manifest geometry has
passed strict audit. The hard preview is thresholded after lifting. Original
source RGB is used only for the overlay and human-review crops. No online/local
learned model, training, target-color shortcut, or per-frame RGB edge snapping
is added in the expansion.

After generation, fail closed unless all 111 exact stems contain every expected
file; every proposal parses as source-sized single-channel uint8, contains both
0 exterior and 255 high-confidence interior, and has hard area in the fixed
engineering safety interval `[0.02, 0.60]`; every metadata stem/size agrees;
normal/depth provenance and hashes remain valid; and real/padding raw-slot sets
are disjoint. This interval detects catastrophic empty/full or gross-region
selection only and does not delete statistical outliers.

Build ten chronological overlay sheets and a second ten-page queue sorted by a
review-only score: background/edge-missing risks weight 5, low confidence 4,
reflection 2, bird 1, plus ten times the uncertainty fraction. For each frame,
make RGB/overlay/uncertainty triptychs around the proposal top, bottom, the more
uncertain side, and the strongest RGB highlight. Crop selection is navigation
only. Robust median/MAD outliers and serious risk flags populate an anomaly
list, but trigger no proposal rewrite or removal.

Verified result: all 111 real frames pass; padding slots 111--119 contribute
zero proposals. Area min/mean/p50/p95/max is
`0.13856/0.22100/0.21222/0.31571/0.35572`. Uncertainty fraction above 140/255
is `0.001783/0.003679/0.003812/0.004875/0.005206` for the same statistics.
Five automatic anomalies carry `background_may_be_included`: 000012, 000013,
000039, 000040, and 000041. Fifteen frames carry the reflection-risk flag; the
bird-behind-glass texture flag remains active on all 111.

Alternatives: Tune thresholds per image; edge-snap each result differently;
skip flagged frames; copy neighboring proposals; treat a high-risk proposal as
failed supervision; or directly install the automatic directory as masks.

Why: All ten chronological sheets show a smooth, view-consistent enclosure
outline. The five automatic background-risk frames still align visually and do
not reveal a systematic method failure. Risk sorting therefore improves human
review order without compromising method consistency.

Paper fidelity: This is offline annotation preparation only. It changes no
renderer, field, BRDF, loss, optimizer, checkpoint, schedule, or RGB training
semantics.

Impact: Preserve
`output/stage_b_tihubird_dr_glass_proposal_111_v1/` as automatic review input.
It contains no `reviewed_soft`, formal training mask manifest, or accepted
supervision. Stop for human review; do not enable `lambda_spec`, train, or enter
Stage C/D.

Required ablation: Human review/correction of all 111 proposals, beginning with
the five background-risk frames and then reflection/high-uncertainty frames.

## B-015 — High-risk review packs are read-only views of frozen proposals

Date: 2026-07-01

Question: How should the 30 highest-priority automatic proposals be presented
for detailed human review without modifying masks or creating supervision?

Chosen implementation: Preserve the exact user-specified order: five
background-risk stems, fifteen reflection-risk stems, then ten
highest-uncertainty stems. For each stem, build one 2748x2308 review image with
the source RGB, proposal soft mask, proposal overlay, boundary-only RGB overlay,
uncertainty overlay/grayscale, area, bbox, uncertainty fraction, and active risk
labels.

Boundary-only display thresholds the existing soft proposal at 128 and draws a
magenta contour on a copy of the decoded source RGB. It is visualization only.
For top edge, bottom edge/yellow plate, left edge, right edge, lower base/black
support, and the previously recorded strongest-reflection location, crop a
fixed 900x506 window directly from the original 3827x2152 pixel domain. Store
both raw RGB and boundary-overlay crops. Paste the boundary versions into the
large pack one-to-one; never downsample then enlarge a crop. Overview panels may
be reduced once with Lanczos to fit, but are not proposal inputs.

Create a single 30-view contact sheet in the same group order. Create JSON, CSV,
and Markdown checklists with blank/null fields for `outer_boundary_ok`,
`base_included`, `background_included`, `edge_missing`, `reflection_leak`, and
`needs_manual_edit`, plus notes. These are human-fillable review records, not a
mask manifest.

Before and after writing the independent output directory, hash the relative
path and content SHA-256 of all 1,110 files under the source `proposal_soft`
tree. Both aggregate hashes are
`428139079931e8091409be70c1c4442c457e0869b10fea894fd2986d0f6b6d1e`.
The high-risk package contains 30 large packs, 180 raw crops, and 180 boundary
crops. Visual inspection sampled one pack from each risk group and confirmed
the required elements and crop coverage.

Alternatives: Edit flagged proposal pixels while preparing review materials;
generate masks in `reviewed_soft`; crop from downsampled DR images; enlarge
small crops; or omit high-risk frames from the queue.

Why: Direct source-pixel crops expose the exact glass/base/background boundary
needed for human judgment while the tree hash proves the automatic proposals
remain untouched.

Paper fidelity: This is offline human-review presentation only. It changes no
mask, renderer, loss, checkpoint, training input, or Stage B model behavior.

Impact: The review pack is
`output/stage_b_tihubird_dr_glass_high_risk_review_pack_v1/`. Stop for human
checklist completion. Do not create `reviewed_soft`, enable `lambda_spec`, run
training, or enter Stage C/D.

Required ablation: None. Human review is the next gate.

## B-016 — Failed top-boundary proposals receive subtractive local candidates

Date: 2026-07-01

Question: How should human-rejected frames 000039--000041 be repaired without
changing the frozen proposal method, copying neighboring masks, or creating
accepted supervision?

Chosen implementation: Record the initial human review exactly: 000039,
000040, and 000041 fail and require local repair; the other 27 high-risk-pack
views temporarily pass. Keep the complete v1 proposal tree read-only. Generate
three independent single-channel uint8 files under `repair_candidates/`, with
no `reviewed` or `final` naming.

The failure is localized to the top: each v1 convex hull reaches row zero and
includes dinosaur/ceiling background, while side and bottom boundaries remain
useful. Detect long near-horizontal candidates from full-resolution RGB Canny/
Hough evidence. Score each candidate using green glass-edge color, native DR
normal/depth boundary support, visible span, and continuity with the long upper
edges of unchanged 000038 and 000042 proposals. Neighbor masks are not copied
or image-coordinate-interpolated. Extend only the selected visible line across
the current frame's v1 bbox and multiply v1 by a 16-pixel soft lower-half-plane
gate. This operation is strictly subtractive: every repair value is at most its
v1 value and side/bottom pixels are not regenerated.

Verified result: 000039 area changes `0.210159 -> 0.183947`, bbox top `0 -> 215`,
and hard changed ratio is `0.026212`; 000040 changes
`0.215424 -> 0.186966`, bbox top `0 -> 191`, and `0.028458`; 000041 changes
`0.229978 -> 0.199907`, bbox top `0 -> 190`, and `0.030071`. Removed hard-pixel
counts are 215,873 / 234,372 / 247,654, while added counts are zero. The source
1,110-file proposal tree has identical before/after aggregate SHA-256
`428139079931e8091409be70c1c4442c457e0869b10fea894fd2986d0f6b6d1e`.

Each source-resolution review page contains RGB, v1/repair overlays, boundary
difference, direct top/bottom/left/right crops, statistics, and a 000037--000043
continuity strip. A three-row comparison provides
RGB/v1/repair/difference columns. The visible top-line segments span about
55.9%, 63.1%, and 41.1% of their bboxes. The remaining line is extrapolated
through dinosaur/reflection occlusion and therefore still requires human
confirmation or manual drawing.

Alternatives: Change global proposal thresholds; rerun all 111; copy 000038 or
000042; interpolate neighbor masks; add missing pixels; overwrite v1; or install
the candidates as `reviewed_soft`.

Why: The subtraction removes the exact visually rejected background region and
preserves previously acceptable side/base evidence. Explicit occlusion risk
keeps automatic geometry from masquerading as human truth.

Paper fidelity: Offline annotation assistance only; no rendering, training,
loss, checkpoint, schedule, or field changes.

Impact: Stop for human review of the three candidates. They are not formal
masks and do not authorize `lambda_spec`, training, or Stage C/D.

Required ablation: None. Human confirmation/manual top-edge correction is the
next gate.

## B-017 — Versioned formal soft masks require a self-hashed manifest

Date: 2026-07-01

Question: How are the user-accepted TiHuBird masks archived and admitted to
`L_spec` without allowing proposal or repair working directories to become
training inputs?

Chosen implementation: Create the new read-only directory
`data/TiHuBird/specular_masks_reviewed_v1/` exactly once. Files 000000--000110
are source-resolution mode-L uint8 PNGs. The 108 unchanged frames are byte
copies of frozen proposal v1; 000039--000041 are byte copies of the accepted
repair candidates. The directory and files are chmod 0555/0444 after complete
validation; no source proposal, repair, checkpoint, or prior run is modified.

`manifest.json` is the only accepted `--specular_masks` input. It fixes the
role, ordered 111 stems, RGB and mask hashes, known source class and source
hash, size/mode/dtype, soft area, nonzero bbox, human `accepted` status,
aggregate mask hash, canonical manifest-payload hash, and explicit exclusion of
DR padding slots 111--119. The loader rehashes every RGB and mask, rejects any
missing/extra file or field mismatch, and admits only `proposal_v1` or
`repair_candidate_v1`. A raw proposal/repair directory cannot satisfy this
contract.

Soft masks are decoded as `[0,1]` float32 and resized from 3827x2152 to the
camera training resolution with continuous `cv2.INTER_LINEAR`; no thresholding
or silent geometry repair occurs. `L_spec = mean(M * relu(k0 - ks))` is added
beside the unchanged whole-image RGB reconstruction loss. Tests prove its ks
gradient is nonzero only where mask support is positive, exactly zero where
the mask is zero, and points toward increasing under-threshold ks under gradient
descent. A diagnostics switch is fail-closed to at most three resumed
iterations and records this support check without adding its probe gradient to
model parameters.

Verified archive hashes before the smoke are aggregate mask SHA-256
`54dbb7661efbb2a334d86cef1cfec1d15ff812e71856c0d88930754013abc2e6`,
canonical manifest-payload SHA-256
`026ad1fa28fb7c2a30656fd37976a4315585e056a317e59f7c44502d69831e4f`,
and manifest-file SHA-256
`056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551`.

Alternatives: Accept a flat mask directory; promote proposal metadata; resize
with nearest-neighbor; threshold soft edges; crop RGB reconstruction to mask
support; or infer missing frames.

Why: The strict formal role and complete hash closure separate immutable human
acceptance from automatic annotation evidence and make accidental proposal
loading fail closed. Continuous interpolation preserves the intended soft edge,
while independent loss terms preserve full-scene reconstruction.

Paper fidelity: This implements the Stage B specular-mask constraint only. It
does not add Transmittance, mesh, inside/outside, two-hit, or any Stage C/D
representation.

Impact: The formal archive and unit tests authorize only the separately bounded
three-step `lambda_spec=0.2` smoke from the original 15,003/3 checkpoint. They
do not authorize 100 steps, long training, or Stage B acceptance.

Required ablation: Inspect the three-step transparent-mask overlay, inside/
outside ks statistics, and L_spec-only gradient evidence before considering a
100-step lambda-spec pilot.

Verified smoke result: The first launch attempt exposed only GPU 0 and stopped
at 0 optimization steps while restoring a checkpoint containing two CUDA RNG
states; its `v1` directory is preserved and is not a resume source. The fresh
`v2` retry restored the same original checkpoint with its original two-GPU
visibility and ran exactly global 15004--15006 / R-local 4--6. No training
formula or parameter changed.

All three L_spec values are finite and positive (`0.164784`, `0.156079`,
`0.130580`). Every valid inside-mask surface sample has a nonzero negative
L_spec-only derivative apart from floating reporting roundoff fractions, every
outside-mask derivative is exactly zero, and gradient descent therefore pushes
under-threshold ks upward. This probe uses `autograd.grad(..., retain_graph=True)`
and is not accumulated into model parameters. Candidate p50/p95/p99 stays
`113/194/253`, `106/215/267`, `121/219/267`; exact intersections stay
`26/55/73`, `25/64/78`, `28/65/75`. Warm raytrace wall time is about 218/208 ms.
Peak allocator evidence is 13,538,393,088 allocated and 15,101,591,552 reserved
bytes. D/R remain 346,118/4,096.

The final checkpoint is `rtgs_stage_b` at 15006/6, SHA-256
`f46c00375699d3b4b7c018a4277b4ba3e93abc66f60ddc7f282a0caf979b93fb`;
recursive inspection covers 63 tensors / 19,912,887 elements with no NaN/Inf.
The fixed-view 000101 transparent mask and overlay have matching 478x269 debug
geometry and the mask hash matches the formal manifest. This is a passed
technical smoke, but a 100-step lambda-spec pilot remains unauthorized until
the user inspects the overlay and statistics.

## B-018 — Controlled 100-step `lambda_spec` pilot remains evidence-only

Date: 2026-07-02

Question: After the user accepts the three-step formal-mask smoke overlay, does
the first bounded 100-step real-scene `lambda_spec=0.2` pilot remain numerically
and operationally stable enough for human Stage B evaluation, without
authorizing long training or any Stage C/D work?

Chosen implementation: Run exactly one fresh pilot from the original read-only
Stage B diagnostic checkpoint
`output/stage_b_tihubird_reflection_dr_c03_resume_diag_15003_retry1/chkpnt15003.pth`
after rechecking SHA-256
`ad92c7d75312cc5df60b7a1dd5d762d8e7d65b8de0f90264d742f969baf7e144`.
The 15,006 smoke checkpoint, the `lambda_spec=0` health pilot, aborted runs, and
profile directories are not resume sources. The only training-semantics change
relative to the earlier health pilot is loading the formal reviewed-v1 manifest
and setting `--lambda_spec 0.2`; RGB reconstruction remains full-image, no rays
or losses are cropped by the mask, and R count, chunk size, BVH, BRDF,
optimizers, schedulers, densification, resolution, and restored RNG state remain
unchanged.

Output path:

```text
output/stage_b_tihubird_reflection_dr_c03_lspec_pilot100_g15003_15103_v1/
```

The run completes exactly global 15004--15103 and R-local 4--103, saves the
final 15103 Stage B checkpoint and independent D/R PLYs, and writes fixed-view
debug directories at 15025, 15050, 15075, 15100, and 15103. D/R counts remain
346,118/4,096; no topology-changing densify/prune event occurs.

Verified scalar evidence: TensorBoard scalar intervals for the 95 non-debug
ordinary steps have p50/p95/mean `0.931/1.089/0.939 s`. Intervals following the
four debug nodes are 1.714, 1.845, 1.843, and 1.915 s. Per-sampled-view
`L_spec` is finite for all 100 steps but not monotonic because the selected
camera and mask support vary. At 15004 / 15025 / 15050 / 15075 / 15100 / 15103,
`L_spec` is 0.164784 / 0.165822 / 0.261634 / 0.173927 / 0.126985 / 0.277022
and total loss is 0.120062 / 0.110359 / 0.106250 / 0.081835 / 0.064169 /
0.119105.

Verified fixed-view material evidence: using the fixed debug view and the
formal mask support, quantized debug-map ks inside the mask rises from the
original 15003 mean 0.095482 to 0.096744 / 0.097544 / 0.098727 / 0.100213 /
0.100522 at 15025 / 15050 / 15075 / 15100 / 15103. Outside-mask fixed-view mean
changes from 0.087836 to 0.089029 / 0.088253 / 0.088522 / 0.088176 / 0.088187.
This is interpreted only as total-loss drift monitoring: the smoke already
verified that the L_spec-only gradient has zero outside-mask support, while the
complete training gradient outside the mask may still move ks through RGB,
normal, perceptual, and other active Stage B losses.

Verified ray/debug evidence: fixed-view candidate p50/p95/p99 is
`126/227/285`, `134/243/310`, `141/256/329`, `147/266/343`, and `147/266/344`.
Exact-intersection p50/p95/p99 is `33/72/86`, `35/77/96`, `36/80/103`,
`38/83/109`, and `38/84/109`. This is gradual growth rather than an
order-of-magnitude acceleration failure. Reflection contribution remains finite
and effectively nonzero, with mean/p99 from `1.087e-4/5.623e-4` to
`1.923e-4/1.171e-3`. Fixed-view raytrace wall time rises from 216.5 ms to
252.1 ms; traversal is 73.3--85.0 ms, and intersection/composite is
140.6--164.8 ms. All raw debug non-finite counts are zero.

Final checkpoint evidence: `chkpnt15103.pth` is `rtgs_stage_b` at global 15103
/ R-local 103, records `lambda_spec=0.2` and the reviewed-v1 formal manifest
hashes, has SHA-256
`c5e40e1a9025a3e191314759e8214e2eb11cba9e04ee2319c6f0c522e4cd6bca`, and
recursive inspection covers 63 tensors / 19,912,887 elements with no NaN/Inf.

Evidence gap: the non-smoke Stage B training loop does not persist
`torch.cuda.max_memory_allocated()` or `torch.cuda.max_memory_reserved()` for
ordinary steps. After process exit, exact whole-step allocator peaks and trend
cannot be reconstructed. Debug raytrace-local peak allocation is recorded in
the fixed-view metadata, but it is not a full training-step allocator peak and
must not be reported as if it were. This gap does not indicate a model failure,
but it prevents claiming the requested whole-step CUDA allocator evidence for
this run.

Alternatives: Resume from the 15006 smoke checkpoint; continue from the
`lambda_spec=0` health output; extend to a long run; lower resolution; change R
count/chunk size; crop RGB loss by the mask; gate reflection rays by the mask;
or enter Stage C/D.

Why: The pilot isolates the formal specular constraint under the same Stage B
rendering semantics while keeping the original C03-initialized D/R state and
all optimizer/RNG continuity. It verifies short-run numeric stability and
mask-directed ks movement before any acceptance discussion.

Paper fidelity: This remains Stage B reflection plus the formal specular-mask
constraint only. It does not add Transmittance, transparent mesh, inside/outside
classification, two-hit geometry, depth violation losses, or any Stage C/D
artifact.

Impact: Classification is
`L_SPEC_PILOT_COMPLETED_AWAITING_USER_EVALUATION_WITH_MEMORY_TELEMETRY_GAP`.
Stage B is still unaccepted. No long training, new pilot, `lambda_spec`
extension, matched StableNormal-vs-C03 run, Stage C, or Stage D is authorized by
this evidence.

Required ablation: Human inspection of the 15025/15050/15075/15100/15103 fixed
debug masks, overlays, ks maps, reflection contribution, scalar L_spec behavior,
and the memory telemetry caveat before deciding whether Stage B is ready for
acceptance review or needs an additional instrumented pilot.

## B-019 — Bounded, semantics-preserving Tier-1 telemetry

Date: 2026-07-02

Question: How can a future matched 10k-versus-15k Reflection handoff viability
pilot preserve auditable per-step topology, loss, mask/material, timing, and
allocator evidence without changing the Stage B training path or pretending
that unavailable raytrace measurements exist?

Chosen implementation: Add a default-off JSONL mode controlled only by
`--stage_b_telemetry_jsonl`, a required positive
`--stage_b_telemetry_max_steps`, and a nonempty caller-supplied
`--stage_b_telemetry_phase_tag`. Validate the planned remaining iteration count
against the bound before the loop, reject a nonempty destination, enforce an
exact versioned schema on every line, and fail closed on overflow, missing,
extra, non-finite, or inconsistently nullable fields.

The normal render continues to request ray diagnostics only from the existing
`specular_smoke_diagnostics` flag. Telemetry neither enables that flag nor adds
a renderer, raytrace, candidate/exact pass, backward pass, or CUDA
synchronization. Ordinary steps therefore record candidate/exact percentiles
and raytrace-forward/candidate-backward timing as null and name them in
`unavailable_fields`. If the separately requested existing smoke mode has
already produced candidate/exact and raytrace-forward statistics, those values
may be reused. No existing boundary provides candidate-backward time.

Reuse the existing loss scalars, D/R counts, D densify/prune call and count
change, real R topology version, sampled camera, global/R-local iterations, and
forward `surface_ks`, valid-surface, and formal-mask tensors. Phase A with
`lambda_spec=0` does not validate or load a formal mask and emits null
mask/material summaries. Phase B transfers the already-rendered tensors to CPU
and records mask support plus inside/outside min/mean/p50/p95/p99. These values
describe the pre-optimizer forward state used by total loss; obtaining a true
post-update map would require a forbidden rerender. The record therefore does
not claim a post-update map or zero total-loss gradient outside the mask.

Measure `whole_step_wall_ms` with CPU `perf_counter` from loop entry through
normal debug/save/checkpoint/optimizer/densification work, excluding telemetry
statistics and JSON serialization and adding no synchronize. Reset only
PyTorch allocator peak counters at an enabled telemetry step's loop entry.
Record current allocated/reserved separately from maximum allocated/reserved.
When the existing debug raytrace resets peak counters internally, preserve the
pre-debug maximum and combine it with the post-reset maximum; record both that
scope and the unobservable transient interval before the internal reset.
These are PyTorch allocator bytes, never `nvidia-smi` process or device-total
memory. A smoke diagnostic's internal training-ray reset may also truncate the
scope and is labeled accordingly.

Add `tools/compare_stage_b_pilots.py` as an offline CPU/PIL/numpy tool. It
selects the latest existing debug directory from two completed runs, requires
matching fixed-view ground truth and shapes, hashes input trees before and
after, and creates shared-scale ks, reflection contribution, and final-residual
views plus signed deltas. All scales are written to comparison metadata. It
does not import training, renderer, raytracer, Torch, or CUDA code. Existing
physical maps are 8-bit PNGs; if sub-code-value reflection has quantized away,
the affected spatial comparison is unavailable and cannot be reconstructed
from scalar metadata.

Alternatives: Enable the existing smoke diagnostics for every pilot step; add a
second traversal or post-update render; use per-image p99 display scales; infer
spatial reflection from metadata summaries; use external device memory as a
PyTorch allocator peak; silently append to a prior run; or leave the earlier
whole-step memory evidence gap unresolved.

Why: The future Tier-1 comparison needs a strict common evidence contract, but
its treatment variable must remain the D handoff checkpoint rather than an
instrumented rendering/training variant. Explicit nulls preserve that boundary
more honestly than extra GPU work. A bounded writer prevents an observability
flag from accidentally becoming an unbounded production logger.

Paper fidelity: This changes only optional observability and offline display.
It changes no field representation, renderer, reflection ray, BVH, candidate
selection, exact intersection, BRDF, composition, loss, optimizer, scheduler,
densification/pruning rule, checkpoint meaning, R initialization, model update,
or RNG consumption.

Impact: Seven new regression tests bring the full repository suite to 102
passing tests. No training, bootstrap, render, Tier-1 pilot, checkpoint, debug
artifact, Stage C, or Stage D work was performed. Stage B remains unaccepted,
and previous pilots retain their historical memory evidence gap.

Required ablation: After separate user authorization, run only the matched
bounded 10k/15k Phase A and Phase B Tier-1 protocol with identical R seed/count,
mask policy, resolution, chunks, ray settings, and phase lengths. Treat it as
handoff viability evidence only; it cannot establish the long-horizon causal
optimality of early versus late R intervention.

## B-020 — Full-state, manually gated Tier-2 continuation contract

Date: 2026-07-02

Question: How can a user execute a 3k-versus-7k Reflection-onset experiment in
finite supervised segments without segment boundaries, fresh-R creation, or
Codex's inability to monitor a live terminal changing the causal treatment?

Chosen implementation: Add the default-off `--operator_gate_continuation`
contract. D-only and Stage B checkpoints written under this mode are version 2
and contain model, optimizer/densification state, global/R-local iteration,
Python/NumPy/CPU/CUDA RNG, remaining shuffled-camera indices, full relevant
data/schedule configuration, and an `optimizer_step_completed=true` marker.
Unlike legacy bounded commands, the operator mode performs the optimizer update
at the command endpoint before saving. This makes a resumed segment equivalent
to the same iteration inside an uninterrupted run; default legacy semantics are
unchanged.

Restore the same deterministically shuffled base camera list, then rebuild the
remaining deck from saved indices. Restore checkpoint RNG after process-local
model/helper construction so VGG/module construction cannot consume trajectory
RNG. Stage A full-state checkpoints are fail-closed requirements for fresh-R
Tier-2 starts, and Stage B version-2 checkpoints are required for every later
gate. Source hashes are checked before and after fresh-R loading. Reflection
initialization retains its explicit private `torch.Generator(seed)` and the
operator path verifies that global Python/NumPy/CPU/CUDA RNG is byte-identical
before and after R creation.

Require exact expected global and R-local source values on every command.
R-local 0 starts must use `lambda_spec=0` with no manifest; R-local 100 and later
must use the formal manifest and exactly `lambda_spec=0.2`. D bootstrap and
Stage B telemetry are bounded, refuse nonempty files and oversized planned
segments, and remain opt-in. The D schema records global/camera/loss, D count
and topology, CPU loop wall time, and current/scoped allocator values without R
fields because R does not exist before onset.

Provide `tools/tier2_operator.sh` as a print-by-default wrapper for exactly one
bootstrap, branch gate, checkpoint-protection, or read-only audit action. It
never chains later gates. Provide `tools/audit_tier2_gate.py` as a CPU-only
packet generator that reads checkpoint/telemetry/log/debug metadata, recursively
scans checkpoint tensors, checks bounded continuity, and reports topology,
loss/L_spec/mask/ks, allocator, candidate/exact/raytrace, anomalies, and missing
artifacts. Codex can issue GO/HOLD/BLOCKED only after the user returns that
packet; no live background monitoring is claimed.

Resolution preflight: the historical C03 D-only run records `resolution=2`.
Existing Stage B Tier-1 runs at `resolution=8` already reach roughly 18--21 GiB
PyTorch allocator peaks on RTX 3090. Running the long Reflection path at
resolution 2 is therefore not a responsible operator command, while switching
only at R onset would confound the sole treatment variable. The wrapper fixes
resolution 8 across the new shared bootstrap and both branches, preserving
internal A/B causality, but this is an explicit deviation from literal legacy
C03 configuration parity. It refuses all training actions unless the user sets
`RTGS_TIER2_ACK_RESOLUTION8=YES`; until that decision, classify the operator
pack as blocked rather than silently choosing a resolution.

Alternatives: Resume legacy Stage A checkpoints without RNG/camera state; skip
the endpoint update at every gate; restart the camera deck at each process;
initialize R from global RNG; run one 12,000-step command; claim Codex is
watching a background process; run Stage B at resolution 2 despite measured
headroom; or change resolution only on the early branch.

Why: Each alternative either breaks deterministic continuation, adds a second
treatment variable, weakens operator safety, or overstates supervision. A
private R generator plus complete trajectory state makes onset timing the only
intended branch difference. Explicit resolution acknowledgment keeps the
remaining specification conflict visible.

Paper fidelity: This changes no representation, renderer, ray, BVH, candidate
selection, exact intersection, BRDF, loss formula, optimizer, scheduler, or
densification rule. It changes only opt-in checkpoint/runtime observability and
whether a bounded endpoint update is preserved for later continuation.

Impact: GPU-enabled full repository tests pass 111/111; CPU-only tests pass 96
with 15 CUDA-only skips. No bootstrap, branch, render, real checkpoint, output
directory, Stage C, or Stage D action was run. Stage B remains unaccepted, and
Tier-2 operator execution is blocked pending explicit resolution approval.

Required ablation: After resolution approval and explicit Gate-0 authorization,
the user may run only the shared 0--7000 D bootstrap. Protect and audit the real
3k checkpoint before Branch A starts, audit the final 7k bootstrap before Branch
B starts, and require returned GO decisions at R-local 100/205/500/1000 and
every later global-1000 endpoint. Do not infer long-horizon onset causality from
any earlier gate.

## B-021 — C03-r8 Tier-2 experiment identity

Date: 2026-07-02

Question: Which single image resolution and experiment identity govern the new
shared D bootstrap and the 3k-versus-7k Reflection-onset branches?

Chosen implementation: Name the experiment exactly `C03-r8 Tier 2 onset study`
and fix `resolution=8` for the shared D-only bootstrap, Branch A, and Branch B.
This is an internally matched onset study, not a strict reproduction of the
historical C03 resolution-2 baseline; do not mix r2/r8 artifacts or compare
their absolute metrics as if only R onset differed.

The operator wrapper now uses `c03_r8` output/log names and includes the exact
experiment name plus resolution in every printed and executed command. Its
telemetry phase tags explicitly include `experiment=...;resolution=8`; cfg_args
records both CLI fields; and version-2 Stage A/Stage B checkpoint configs record
both fields. Operator-gated training fails before entering the loop if either
identity value differs. The earlier temporary `RTGS_TIER2_ACK_RESOLUTION8`
execution block is removed because the user has supplied the required approval.

All previously defined trajectory controls remain unchanged: one common D-only
bootstrap, private seeded fresh-R initialization, R-local 1--100 with no mask and
`lambda_spec=0`, R-local 101+ with the formal manifest and
`lambda_spec=0.2`, bounded manual gates, and returned-packet GO/HOLD/BLOCKED
review. This clarification changes no renderer, tracer, BVH, candidate/exact
logic, BRDF, loss, optimizer, scheduler, densification, checkpoint state
meaning, initialization, or RNG consumption order.

Alternatives: Keep the operator pack blocked; run the long path at resolution 2;
or switch resolution only when R starts.

Why: Resolution 8 is the approved dual-RTX-3090-feasible configuration, and
using it from bootstrap through both branches keeps R onset as the intended
between-branch treatment variable. Explicit identity in every audit surface
prevents accidental legacy-r2 mixing.

Paper fidelity: Observability and experiment naming only; training mathematics
and schedules are unchanged.

Impact: The operator pack is ready for the user to run only Gate 0 manually. No
bootstrap, branch, render, checkpoint, output, Stage C, or Stage D action was run
while making this decision.

Required ablation: The approved Tier-2 onset study itself; no comparison to the
legacy resolution-2 C03 absolute metrics is claimed.

## B-022 — One-shot Tier-2 orchestration authorization

Date: 2026-07-02

Question: How should the explicitly authorized C03-r8 3k-versus-7k long onset
study run without returning to the user at the former manual gates?

Chosen implementation: Add `run-all --execute` to the existing operator wrapper
and implement its coordinator in a CPU-only control module. Require the explicit
`RTGS_TIER2_ACK_RESOLUTION8=YES` environment acknowledgment, a clean committed
worktree, exactly two RTX 3090 devices at physical indices 0/1, no conflicting
GPU training process, and entirely new `oneshot_v1` paths. The previously
user-operated manually gated v1 bootstrap and partial branches are immutable
historical assets and are never used as source material for the one-shot.

Run one continuous D-only bootstrap on GPU 1. Once its real 3k full-state
checkpoint and telemetry pass a CPU-only finite/identity/continuity audit,
record its SHA-256, protect only that checkpoint, and start Branch A on GPU 0
while bootstrap continues. After the 7k bootstrap completes and passes audit,
hash all seven checkpoints, protect the full bootstrap tree, and start Branch B
on GPU 1. Each branch uses exactly two processes solely because the supervision
contract changes after R-local 100: the first 100 steps have no mask and
`lambda_spec=0`; local 101 through global 15000 uses the accepted manifest and
`lambda_spec=0.2`. No other human gate remains.

Save full-state checkpoints at local 100/200/500/1000, every later global 1000,
and global 15000. Existing debug interval 100 covers every required node without
adding renderer passes. The coordinator records process-group PID, GPU, times,
exit status, logs, last checkpoint, and telemetry progress. It monitors existing
telemetry for exact global/R-local continuity and nonfinite values and checks
physical GPU isolation. OOM, NaN/Inf, I/O failure, discontinuity, abnormal exit,
GPU mismatch, output collision, or shared-source mutation is a hard failure;
candidate/topology/material/allocator variation is retained for final audit and
does not stop training. No automatic retry or cleanup occurs.

`status` only reads the atomic coordinator state. `final-audit` runs with CUDA
hidden, loads checkpoints with `map_location="cpu"`, recursively checks finite
state and experiment identity, joins both telemetry phases, verifies required
debug/checkpoint nodes, checks source hashes/logs, and writes one JSON report.
The R-local-100 debug node intentionally has no mask/overlay because loading the
formal mask there would violate the immutable warmup contract; this absence is
recorded as N/A rather than fabricated.

Alternatives: Reuse the earlier manual v1 assets; keep stopping for human GO;
run both branches only after bootstrap 7k; load the formal mask during warmup;
or automatically retry failed commands.

Why: Fresh paths and one shared bootstrap preserve the onset comparison, while
overlapping Branch A with the remaining bootstrap uses both GPUs without
changing either trajectory. The only process boundary retained is demanded by
the fixed L_spec/mask phase policy.

Paper fidelity: Operator control and offline observability only. Renderer,
raytracer, BVH, candidate/exact intersection, BRDF, losses, optimizer,
scheduler, densification rules, initialization, and RNG consumption are
unchanged.

Impact: After one purpose-specific implementation commit and clean preflight,
Codex is authorized to launch the full one-shot and wait until completion or a
hard failure. Stage B remains unaccepted; Stage C/D remain forbidden.

Required ablation: This one C03-r8 3k-versus-7k long onset study. It does not by
itself establish that either onset is superior.

## B-023 — Allocator-lifecycle retry after `oneshot_v1` HARD_FAILED

Date: 2026-07-02

Question: What is the smallest evidence-backed repair for the two formal-branch
CUDA OOMs that preserves the C03-r8 3k-versus-7k onset treatment?

Observed evidence: `oneshot_v1` completed the shared D-only bootstrap and both
R-local 1--100 warmups. Branch A's last complete telemetry was global 3347 /
R-local 347; backward then failed on a 32 MiB request with 20.65 GiB allocated,
22.87 GiB reserved, and 25.38 MiB device-free. Its scoped allocated/reserved
maxima were 21.676/22.893 GiB. Branch B's last complete telemetry was global
7179 / R-local 179; raytrace forward failed on a 26 MiB request with 21.01 GiB
allocated, 22.34 GiB reserved, and 41.75 MiB device-free. Its scoped maxima
were 20.812/22.332 GiB. Both paths have finite saved checkpoints and zero
telemetry nonfinites. Candidate/exact p99 stayed in the same hundreds-scale
range rather than increasing by an order of magnitude.

The full telemetry also shows end-of-step allocated memory returning to roughly
1--2 GiB and no monotonic increase. Reserved memory instead grows near device
capacity and occasionally falls sharply after a successful high-pressure step,
which is consistent with PyTorch allocation retry releasing cached blocks.
Training keeps the prior step's `package` alive while evaluating the next
render RHS; debug and checkpoint payload locals can also retain bounded CUDA
references until their next replacement. These are bounded lifetime overlaps,
not an unbounded graph/telemetry leak. Debug retention can reduce Branch A's
margin but cannot be the common sole cause because Branch B failed before its
first formal debug node. D topology and sampled camera determine the real live
peak; allocator fragmentation then makes a small contiguous allocation fail.
The failed-camera identities are unavailable because neither failed step wrote
a complete telemetry line, so no single camera is asserted as causal.

Chosen implementation: Introduce the opt-in retry identity
`oneshot_v2_allocator_lifecycle_retry` and allocator policy
`release_ephemeral_cache_each_step_v1`. At every completed Stage B step, after
all loss, optimizer, densification, debug, checkpoint, and telemetry consumers
finish, explicitly delete step-owned forward/loss/debug/checkpoint references
and call `torch.cuda.empty_cache()` to return only unused cached blocks. Launch
both A and B with exactly
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`. The policy is default-off and
operator-only. It performs no renderer/raytrace/backward work, consumes no RNG,
and changes no tensor, optimizer, scheduler, topology, or checkpoint state.

At each branch's first formal step (R-local 101), after the safe-boundary release,
record current allocated/reserved, device free/total, and that step's scoped
allocated peak. Define allocator capacity as `current allocated + device free`,
required peak as the maximum of the current scoped peak and v1's largest
successful scoped peak, 23,274,475,520 bytes, and projected headroom as capacity
minus required peak. Require at least 1,073,741,824 bytes (1 GiB) or stop before
the long formal segment. The margin is over 32 times either failed 26/32 MiB
request and is anchored to a real successful v1 high-water mark; it is a
fail-closed engineering gate, not proof that later cameras cannot exceed it.

Reuse the immutable `oneshot_v1` shared D-only 3k and 7k full-state checkpoints,
whose approved SHA-256 values are respectively
`c8f83b17d53f49a3f283d25078e69cb4c8073b2b09354cc901ecc23eae772e6c`
and `59461b60ac721f4e724b48ced9ee319f98ede49490f38bce91651590bb760d89`.
This is valid because the repair is Stage-B-only and therefore cannot change
the completed D-only mathematics or RNG trajectory. Fresh R is recreated from
each shared source at local 0. Do not continue from v1 A3200 or B7100: A already
contains 100 formal steps while B contains none, and neither trajectory applied
the v2 memory policy from R-local 1 or records its identity/headroom contract.

Alternative retained but not implemented: If the allocator-only retry still
lacks measured headroom, use one fixed, identical memory-bounded ray chunk size
for both branches under a new retry decision. Chunking preserves candidate and
exact-intersection definitions but may change floating-point accumulation order
in backward; it therefore requires matched equivalence/gradient tests and must
be recorded as an additional common treatment. Resolution, R count, L_spec,
D optimization, and densification remain unchanged unless both prior options
are proven insufficient and a separately named experiment is approved.

Why: The selected repair directly targets both demonstrated mechanisms while
leaving the onset comparison intact. It does not attribute the OOM to generic
"insufficient VRAM," nor claim allocator configuration alone is guaranteed;
the local-101 measurement decides whether adequate real headroom exists.

Paper fidelity: Allocator lifetime and operator observability only. No renderer,
ray tracer, BVH, candidate selection, exact intersection, BRDF, loss, optimizer,
scheduler, densification threshold/timing, checkpoint state meaning, R
initialization, random-number consumption, data, mask schedule, or resolution
changes.

Impact: All v1 outputs/logs/checkpoints/telemetry/state/audit remain immutable.
V2 uses new branch, log, telemetry, state, packet, audit, and JIT paths and runs
the same memory policy on both GPUs. The full GPU-enabled suite passes 117/117;
the CPU-only suite passes 102 with 15 CUDA skips. No v2 bootstrap, branch,
render, or real-scene smoke was run while implementing this decision. Stage B
remains unaccepted; Stage C/D remain forbidden.

Required ablation: The user may run only the single committed `oneshot_v2`
workflow. If either headroom gate fails or either branch OOMs, retain the CPU-only
final audit and return for a new decision; do not auto-retry or silently change
chunk size.

## B-024 — Telemetry assignment correction and isolated `oneshot_v3`

Date: 2026-07-02

Question: How should the first-step `oneshot_v2` implementation failure be
corrected without reusing its partial outputs or changing the training study?

Observed evidence: Both v2 branches restored their approved D-only source,
created fresh R=4096, loaded all cameras, and completed the first training step.
Both then raised `TypeError: 'NoneType' object is not subscriptable` while
normalizing telemetry ks fields. No telemetry row or branch checkpoint was
written. This was a Python observability bug, not CUDA OOM, numerical failure,
or training instability. The CPU-only v2 final audit is preserved with SHA-256
`492464757a0776d6b463a80d0b89fe2e668549586ab5a94bb01394142c0bcb7a`.

Chosen implementation: Restore the smoke-only dictionary assignment to
`record`, assign the ordinary strict-schema dictionary to `telemetry_record`,
and add a regression assertion that exactly one ordinary telemetry assignment
exists after the smoke record assignment. Use the new identity
`oneshot_v3_allocator_lifecycle_retry` and new v3 branch, telemetry, log, state,
audit, packet, and JIT paths. Never resume or overwrite v2.

Why: The correction is two variable-name substitutions. A new run identity is
still required because v2 output directories and terminal audit are immutable
evidence and the operator correctly refuses nonempty telemetry/output paths.

Paper fidelity: No training mathematics, allocator policy, renderer, tracer,
loss, schedule, RNG, source checkpoint, mask, resolution, R initialization, or
R count changes.

Impact: The protected v1 D-only 3k/7k sources remain the only branch sources.
V2 remains a preserved implementation failure. Stage B remains unaccepted and
Stage C/D remain forbidden.

## B-025 — Memory-bounded v4 after allocator-only retry failure

Date: 2026-07-02

Question: How can Stage B remain fast on ordinary views while recovering from
the real single-step live-memory peaks demonstrated by `oneshot_v3`?

Observed evidence: V3 Branch B completed local 101 but the old gate stopped it
because a 0.67 GiB projected margin was below the cross-branch 1 GiB threshold;
there was no B OOM. V3 Branch A passed that gate, completed finite telemetry
through global 3764 / R-local 764, and saved a finite global-3500 checkpoint.
The next backward failed while requesting 16 MiB with 21.98 GiB allocated,
22.89 GiB reserved, and 1.38 MiB device-free. The preceding successful step
had already reached a 23,531,863,040-byte scoped allocated peak and performed
the completed-step cache policy. Therefore fragmentation across iterations is
not the sufficient explanation: a single training graph can consume almost the
entire device, and clearing cache after the previous step cannot bound it.

Code audit found that every ray chunk's autograd graph is retained until one
full-image backward. `RaytraceAux.contributing_weights` also remained attached
to that graph even though it is consumed only after backward by a `no_grad`
densification-statistics method. Reducing chunk size lowers padded candidate and
candidate-reduction temporaries, but only checkpoint/recompute supplies a true
bounded-graph fallback.

Chosen implementation: Define the new operator identity
`oneshot_v4_memory_bounded_retry` and policy
`adaptive_pressure_cache_and_ray_retry_v1`. Both branches start fresh R from
the same protected D-only 3k/7k sources and use identical rules:

1. Base ray chunk size is 2048 and aux contributing indices/weights are
   detached immediately; this does not change loss inputs or gradients.
2. A CUDA OOM before any optimizer, densification, or checkpoint action clears
   partial D/R/exposure gradients, releases unused cache, and recomputes the
   same selected camera without consuming RNG at chunk 1024, then 512.
3. If ordinary chunk 512 still fails, checkpointed 512 recomputes candidate
   selection and exact intersections during backward instead of retaining all
   chunk intermediates. Exhausting that final fallback remains a hard failure;
   no finite-memory implementation can promise completion if the non-ray graph
   alone exceeds device capacity.
4. Successful ordinary steps explicitly delete their ephemeral payloads but
   call `empty_cache()` only when measured device-free memory is below 2 GiB.
   Retries and the local-101 measurement force a release. The allocator uses
   `max_split_size_mb:128,garbage_collection_threshold:0.8`.
5. The local-101 record now measures the v4 path itself against an advisory
   256 MiB margin and no obsolete v1 cross-branch reference peak. It no longer
   terminates a successfully computed branch; actual allocation failures are
   handled by the adaptive retry ladder. Recovery events are emitted as
   structured `STAGE_B_MEMORY_RETRY` log records and included in the CPU-only
   final audit.

Retry safety: Camera selection, reflection-local increment, and learning-rate
updates occur once before the attempt sequence. An OOM occurs before any model,
optimizer, topology, RNG, telemetry, or checkpoint commit. Partial gradients
are cleared before recomputation. Candidate selection is deterministic for
unchanged parameters and acceleration state. The fallback therefore changes
only chunk grouping/recomputation and floating-point gradient accumulation
order, not the represented rays, candidates, exact-intersection definition,
BRDF, loss, schedule, or data.

Alternatives: Continue every-step `empty_cache`; lower resolution/R count;
disable L_spec; stop D optimization/densification; resume a failed branch; or
use checkpoint/recompute on every ordinary step.

Why: Per-step cache release measured about 46% slower yet still OOMed. The v4
fast path avoids that tax, while progressively stronger fallbacks spend extra
time only on views that need it. Resolution, counts, losses, and schedules stay
matched across onset branches.

Paper fidelity: Candidate and exact-intersection definitions are unchanged.
Chunk grouping may change floating-point accumulation order, so v4 is a new
matched internal experiment identity and is not mixed with v1/v2/v3 numerical
trajectories. CUDA tests require ordinary and checkpointed outputs plus D-ray
and all R-parameter gradients to agree within explicit tolerances.

Impact: V1/v2/v3 artifacts and audits remain immutable. V4 uses new output,
log, telemetry, packet, state, audit, and JIT paths. No v4 real-scene training
or render was run during implementation. The full GPU-enabled suite passes
120/120; CPU-only passes 104 with 16 CUDA skips. Stage B remains unaccepted;
Stage C/D remain forbidden.

## B-026 — Tier-2 v4 closeout and Stage C geometry source

Date: 2026-07-03

Question: Did the completed C03-r8 onset study satisfy Stage B closeout, and
which final Diffuse state should seed Stage C geometry work?

Observed evidence: Both isolated v4 branches completed with exit code zero at
global 15,000. Branch A is R-local 12,000 with D/R counts 258,593/3,088 and
checkpoint SHA-256
`050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84`.
Branch B is R-local 8,000 with D/R counts 264,303/3,044 and checkpoint SHA-256
`f79c0e3a0548aee3278b816d2d37edd9f4c89d4e5cc3de82b0c470fea8dc4d68`.
Both checkpoints have zero recursively scanned non-finite elements, every
telemetry row has `nonfinite_count=0`, all required debug/PLY nodes exist, and
both logs end in `Training complete.` Branch A has lower matched loss and
L_spec through the final window, sampled-train PSNR 28.562 versus 28.184,
fixed-view PSNR proxy 36.258 versus 35.941, candidate/exact p99 495/124 versus
529/153, and no adaptive retries. Branch B completed after 731 retry attempts
across 399 steps.

The first automatically written final audit incorrectly classified all Branch
B retry records as malformed because the logger appends a timestamp after each
valid JSON object. The repaired parser uses `JSONDecoder.raw_decode`, accepts
only the exact `[DD/DD HH:MM:SS]` suffix, and rejects all other trailing text.
The regenerated CPU-only report is healthy with no errors.

Chosen implementation: Formally close Stage B and select Branch A global
15,000 as the sole Stage C D-geometry source. Preserve Branch B as the matched
7k control. Stage C may read the selected D state for geometry audit, mesh
extraction, mask-hard camera-ray two-hit intersection, and versioned caches.
Reflection rays remain full valid-surface rays. `mask_soft` remains L_spec-only,
`mask_hard` gates two-hit rays, `mask_eroded` gates mesh/depth validation, and
RGB reconstruction remains full-frame.

Why: In this single matched resolution-8 experiment, 3k onset is both viable
and the stronger downstream geometry-source candidate; choosing it avoids
mixing branches in Stage C. A healthy audit and explicit source hash make the
transition reproducible.

Evidence boundary: There is no Reflection ground truth. Higher mask-interior
ks and lower RGB/L_spec do not prove semantic Reflection separation, and small
mask-exterior ks/contribution differences remain indirect proxies. This study
does not prove 3k is globally optimal, does not reproduce the old resolution-2
C03 baseline, and does not authorize Transmittance, second bounce, or Stage D.

Required next validation: Before extracting a fixed glass mesh, audit the
selected D alpha/depth/normal against all formal masks for cross-view coverage,
boundary agreement, gaps, floaters, and background adhesion. Continue to mesh
and two-hit preprocessing only on an explicit `STAGE_C_MESH_AUDIT_PASS`.

## C-001 — Branch-A D audit and measured outer-shell preprocessing

Date: 2026-07-03

Question: Can the selected Branch-A global-15,000 Diffuse state provide a fixed
glass mesh and reliable front/back camera intersections without introducing a
hand-authored enclosure?

Observed evidence: The D-only source-resolution audit covers all 111 formal
masks. Median/minimum eroded-mask finite alpha/depth/unit-normal coverage is
1.0/0.96384. Multi-view voxel support >=2 covers 0.44312 of occupied voxels and
the largest supported component covers 0.83541. The audit therefore records
`STAGE_C_MESH_AUDIT_PASS`, while its pages also show residual non-glass adhesion.

Chosen implementation: Fuse the `ks >= 0.9` audited D candidate's eroded-mask
depth into a deterministic CPU TSDF at maximum grid resolution 160, minimum
weight 2, four-voxel truncation, and 0.15 bounds margin. Close two voxel-scale
cracks, retain the largest connected negative-TSDF occupancy, fill only enclosed
cavities, and mesh that measured outer boundary. Do not fit a cuboid or other
primitive. Intersect only `mask_hard` camera rays with a CPU BVH; cache the first
and last distinct positive hit, hit count, validity, and back position. Bind
every NPZ to schema, mesh hash, and source-checkpoint hash and validate finite
arrays plus `t_far > t_near` on load.

Why: High-ks D selection is an indirect glass candidate supported by the actual
Stage-B state, whereas full-scene D depth plainly contains the museum background
and bird. Occupancy cleanup removes disconnected TSDF debris without inventing
an ideal box. First/last hits define the outer interval even when a concave mesh
has intermediate crossings.

Measured result: The v3 mesh is one watertight component with 118,003 vertices,
236,470 faces, and no boundary/non-manifold edges. All 111 caches reload and all
valid rays satisfy depth order. Hard-mask validity is 0.80313 mean/0.67551
minimum and eroded validity is 0.81843 mean. However, 0.44044 of valid rays have
more than two crossings and debug maps show large structured holes plus internal
far-surface structure.

Evidence boundary and decision: The source checkpoint passes the requested
mesh-extractability audit, and Stage C preprocessing is implemented, but the
result does not yet satisfy the geometric prerequisite for Stage D. Do not hide
the holes with stronger unvalidated smoothing, a visual-hull heuristic, or a
hand-fitted cuboid. Stage C remains current; no Transmittance field, T training,
second bounce, or Stage D work is authorized.

Paper fidelity: This is offline geometry preprocessing only. It does not change
the Reflection ray domain, renderer, ray tracer, BRDF, losses, D/R parameters,
or any Tier-2 artifact. `mask_hard` gates mesh rays, `mask_eroded` gates depth
validation/fusion, `mask_soft` remains L_spec-only, and RGB remains full-frame.

Required ablation: Before Stage D, establish and approve a geometry repair that
raises cross-view two-hit coverage and removes folded/multi-crossing structure
without replacing measured glass geometry by an assumed primitive.

## C-002 — DR-guided six-plane repair for the TiHuBird enclosure

Date: 2026-07-03

Question: Can the failed pure-TSDF shell be repaired with the existing
DiffusionRenderer depth/normal evidence without hand-authoring a glass box?

Observed evidence: The repository contains 111 mapped raw DR depth and normal
frames at 704x384. Raw depth is an 8-bit RGB decoder output rather than metric
camera depth. Against full-D metric depth on reliable pixels outside the formal
glass mask, its per-view monotonic relation is nevertheless strong: Spearman
has 0.89904 median. Robust per-view direct- or inverse-depth calibration has R²
0.38935 minimum / 0.73163 median. DR normals visibly encode the glass planes and
fit three orthogonal world axes with 0.98984 median axis alignment.

Chosen implementation: Decode the audited C03 `[-x,+y,+z]` camera-normal
mapping, transform normals into world space, and robustly fit three sign-invariant
orthogonal axes. Use the retained D-depth TSDF mesh only to initialize metric
plane bounds at its 2nd/98th percentiles. Optimize the six bounds against all
111 geometry-resolution hard masks, then validate the projected cuboid against
all 111 original-resolution formal masks. Separately calibrate every raw DR
depth image on opaque outside-mask D depth and compare it with the candidate
front hit; never interpret raw PNG intensity directly as metric depth.

Why: TiHuBird's target is visibly a planar six-face enclosure. A constrained
model eliminates the folds and internal sheets that violate this known scene
structure, while its orientation, scale, offsets, and validation are all tied
to recorded DR/D/mask measurements. This is materially different from manually
drawing a convenient cuboid.

Measured result: The v2 mesh is watertight with 8 vertices/12 triangles. At
source resolution, recall is 0.97200 mean / 0.91240 minimum and IoU is 0.95128
mean. The v4 caches reload strictly for all 111 views, have hard-mask validity
0.97038 mean / 0.90821 minimum, eroded validity 0.99117 mean, 100% valid
`t_far > t_near`, and exactly zero more-than-two intersections. Worst-view
overlays show that the remaining error is concentrated at mask disagreement;
notably `000053` contains a background protrusion that is not absorbed into the
mesh.

Evidence boundary: The method assumes this particular enclosure is cuboid. DR
depth remains a calibrated relative proxy; across views, the median of each
view's median cuboid-versus-DR relative front-depth residual is 0.12280 and the
worst is 0.42512. The fit must not be generalized to curved or unknown glass
without a new model and audit.

Paper fidelity: Offline Stage C geometry only. Reflection rays remain full
valid-D-surface rays. `mask_hard` gates mesh rays, `mask_eroded` validates
geometry, `mask_soft` remains L_spec-only, and RGB remains full-frame. No D/R
checkpoint, Tier-2 artifact, renderer, BRDF, optimizer, loss, T field, second
bounce, or Stage D behavior is changed.

Impact: Retain the failed TSDF v3/v2-cache evidence unchanged. The repaired
mesh/caches use new v2/v4 identities. Stage C now supplies the geometric
prerequisite for Stage D under the documented cuboid assumption, but Stage D is
not entered and T training remains unauthorized.

Required ablation: If Stage D is later authorized, preserve this fixed mesh and
cache as immutable inputs; do not refit them during T training. Report mask-
invalid pixels separately rather than silently expanding the cuboid.

## C-003 — Immutable TiHuBird geometry release and Stage D handoff

Date: 2026-07-03

Question: How is the accepted cuboid mesh/cache made a reproducible Stage D
input without putting binary caches in ordinary Git history?

Chosen implementation: Freeze `stage_c_geometry_release_v1` as a copied,
read-only asset tree under `output/` and commit its self-hashed manifest at
`geometry_releases/stage_c_geometry_release_v1.json`. The manifest binds the
3k-A/global-15,000 checkpoint, pre-release code commit, formal masks, DR
provenance/calibration, six-plane parameters, optimization settings, mesh,
metadata, all 111 camera/image/mask/cache identities, and every asset hash. Its
ordered path-and-hash aggregate is
`4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d`.

Why: Git LFS is not configured. Keeping large NPZ/PLY assets out of ordinary Git
while committing a complete self-hashed manifest gives reproducibility without
repository bloat. Copying into a dedicated release prevents Stage D from
depending on mutable candidate paths.

Validation: The CPU-only audit checks 111/111 hashes, source image/mask/camera
identity, finite t-near/t-far/back positions, strict depth order, coverage
metadata, asset permissions, and runtime-generation independence. It returns
`STAGE_C_GEOMETRY_RELEASE_VALID` with the same aggregate hash.

Impact: Stage C is accepted and the project advances to Stage D. Stage D loaders
must run this validation, read only release-relative mesh/cache files, and write
`geometry_release_id` plus aggregate hash into checkpoints. It cannot regenerate
or overwrite geometry.

Evidence boundary: The release is a data-constrained six-plane proxy for the
approximately cuboid TiHuBird enclosure only. It is not a general transparent-
object mesh extractor.

Required ablation: None for release identity. Any future geometry change is a
new release ID and cannot silently replace v1.

## D-001 — First Stage D D/R/T execution semantics

Date: 2026-07-03

Question: How should the first independently optimized T field and two-bounce
path consume the frozen Stage C geometry?

Chosen implementation: Initialize a fresh Transmittance surfel field with an
independent seeded generator inside the frozen mesh world AABB. Restore D/R and
their optimizers from the accepted 3k-A/global-15,000 Stage B checkpoint. For
each frozen-valid glass pixel, use the camera-through-glass direction; trace T
from `D_position + eps*d_trans`, then trace a Diffuse base-color adapter from
`back_position + eps*d_trans`. The adapter aliases D's actual geometry,
opacity, and base-color parameters, so second-bounce gradients update D without
copying or merging fields.

Compose `Ct = Cin + (1-Ain)*Cout` and
`At = Ain + (1-Ain)*Aout`. Use thin-shell
`wt = mask_hard*(1-F)` and add
`alpha*ks*wt*Ct` to the unchanged Stage B diffuse/reflection contributions.
Reflection rays remain global valid-D-surface rays.

Depth coordinates: the common raytracer returns distance relative to its ray
origin. Convert T depth to absolute camera-ray distance by adding the D first-
origin distance before applying `relu(Din-t_far)`. L_depth uses only
`mask_eroded & valid_two_hit`. The master-plan default enables L_depth at global
40,000; the explicitly bounded smoke may set its start to the first smoke step
so the loss path is exercised and recorded. This smoke override is metadata,
not a long-training schedule decision.

Memory policy: Use checkpointed 512-ray chunks for the first smoke across R, T,
and second-bounce D. This preserves ray/candidate/intersection definitions while
bounding retained autograd intermediates; as in Stage B v4 it may change only
floating-point accumulation order. The smoke is fail-closed to at most three
new steps.

Checkpoint contract: D/R/T model, optimizer, scheduler/topology state remain
independent. Every checkpoint stores the frozen release ID and aggregate hash,
source Stage B checkpoint hash, three local/global iterations, RNG, and camera
deck; resume refuses any geometry mismatch.

Paper fidelity: First/second bounce, alpha-over Ct, thin-shell `(1-F)` weight,
and L_depth follow the master plan. T random-AABB initialization, smoke depth-
start override, and chunk size are explicit engineering choices.

Impact: No Stage B checkpoint or Stage C asset is modified. No long run or Stage
E work is authorized. A real smoke must still prove finite optimization,
checkpoint/PLY/debug output, release immutability, and restore compatibility.

Required ablation: Longer Stage D work must return to the global-40,000 L_depth
schedule unless a separately approved experiment changes it. The smoke does not
establish branch semantic separation.

## D-002 — Canonical frozen-cache camera stems

Date: 2026-07-03

Question: Why did the second Stage D smoke invocation stop before its first
backward despite a valid 111-view geometry release?

Observed evidence: Scene camera identity was `000047.jpg`; release cache identity
was `000047`. Both refer to the same source image, but exact string lookup raised
`KeyError`. No optimizer update or checkpoint occurred.

Chosen implementation: Canonicalize the runtime camera name with
`Path(str(image_name)).stem` at the cache lookup and fixed-debug-view selection
boundaries. Add a CUDA renderer regression using `toy.jpg` with cache stem
`toy`.

Paper fidelity: Identity normalization only; no tensor, ray, loss, geometry,
schedule, or RNG behavior changes.

Impact: Preserve v1/v2 smoke evidence and use a new v3 output identity.

Required ablation: None.

## D-003 — Minimum real-scene smoke and actual Stage D resume

Date: 2026-07-03

Question: Does the implemented Stage D path execute end to end on TiHuBird,
including fresh T, two bounces, loss, optimizer state, checkpoint restore, PLY,
debug, telemetry, and frozen-release enforcement?

Observed evidence: `smoke_v3` completed two fresh steps to global 15,002, saved
a full checkpoint, then restored that checkpoint and completed one more step to
global 15,003 / R-local 12,003 / T-local 3. CPU-only audit recursively checked
86 tensors / 15,055,935 elements, all finite, and returned
`STAGE_D_SMOKE_PASSED_AWAITING_REVIEW`. Final checkpoint SHA-256 is
`f71d4644fcb2873ddc9d0ea058c87ce698c405d2f1b43e83b7fdd9df4349e560`.

Branch independence: D/R/T counts are 258,593/3,088/4,096 with separate
optimizer state and storage. Relative to their source/first saved state,
maximum xyz updates are `8.83e-4 / 7.44e-5 / 3.20e-4`; opacity and color also
change finitely and nonzero. This proves all three optimizers executed; it does
not prove semantic branch separation.

Path evidence: The fixed 000039 debug view has coherent frozen two-hit support,
faint but nonzero inside T color, and structured outside D color. Ct uses the
tested alpha-over formula. L_depth is finite/active for all three steps; the
fixed-view `Din <= t_far` fraction is 0.63439. The final image is visibly
over-bright over glass, expected from an untrained T branch and strong outside
contribution, so no quality acceptance is claimed.

Engineering health: All telemetry has `nonfinite_count=0`. Scoped peak
allocated/reserved memory is at most 3,483,877,888/4,395,630,592 bytes and step
wall time is 5.94--8.29 seconds. The geometry release revalidates after training
with unchanged aggregate
`4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d`.

Impact: The technical smoke passes and awaits user review. No long Stage D
training, semantic T claim, Stage D acceptance, or Stage E transition follows.

Required ablation: Before long training, decide how to warm up T and control
early outside/transmission energy; retain the global-40,000 production L_depth
schedule unless separately approved.

## D-004 — Formal T onset from the frozen global-15,000 geometry source

Date: 2026-07-03

Question: Which D/R state, geometry release, and loss schedule define the first
formal 5,000-step Stage D trajectory after the successful technical smoke?

Chosen implementation: Start one entirely new trajectory from the exact
`C03-r8 Tier 2 onset study` Branch-A Stage-B checkpoint at global 15,000 /
R-local 12,000, SHA-256
`050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84`.
Use the already frozen `stage_c_geometry_release_v1`, aggregate SHA-256
`4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d`.
This release's geometry source is that same global-15,000 state. The first T
update is therefore global 15,001; do not separately extend D/R to global
20,000 and do not rebuild the mesh or cache.

Create fresh T with the smoke-verified independent initialization
`random_bbox`, count 4,096, seed 20260703, inside the frozen mesh AABB. Restore
the complete D/R model, optimizer, topology, RNG, and camera-deck state from the
source. Continue through global 20,000 only, yielding R-local 17,000 and T-local
5,000. Use the verified checkpointed 512-ray chunks for R, first-bounce T, and
second-bounce D. Preserve full-frame RGB, global R rays, frozen valid two-hit
caches, first-bounce T, second-bounce D, alpha-over Ct, and all existing
rendering/loss/optimizer mathematics.

The source checkpoint actually records `lambda_spec=0.2`, `K0=0.9`, and the
reviewed-v1 mask manifest file SHA-256
`056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551`.
The formal run must validate and copy this contract into metadata rather than
infer it from historical prose.

This is not the smoke's early-L_depth exercise. For every update global
15,001--20,000, compute depth violation only as telemetry/debug evidence but
apply zero depth-loss weight. The production schedule remains
`stage_d_depth_start_iteration=40000`; at global 40,000 and only then,
`lambda_depth=0.2` becomes active. No part of this run authorizes continuation
past global 20,000 or entry into Stage E.

Alternatives: Resume `smoke_v3`; repeat short smoke/pilot gates; extend D/R to
20,000 before creating T; rebuild a geometry v2; activate L_depth at T onset;
or change chunks/formulas to manage early brightness.

Why: Stage C is already frozen against the selected global-15,000 geometry, and
the smoke proved the exact T/two-bounce execution path. A fresh 5,000-step
trajectory from that same state isolates whether T semantic structure starts to
form without mixing smoke updates, a new geometry fit, or early depth pressure.

Paper fidelity: This changes the project-specific T onset from the master
plan's generic global 20,001 to global 15,001 because the accepted geometry is
already frozen at global 15,000. It retains the master plan's global-40,000
L_depth activation and changes no image-formation formula.

Impact: A fail-closed one-command operator owns the unique output
`output/stage_d_tihubird_c03r8_formal_onset_g15000_g20000_v1`. It saves five
complete checkpoint/PLY/review nodes, per-step telemetry, nine fixed views,
contact sheets, a CPU-only recursive audit, and a semantic-review report. A
healthy run stops at `HOLD_FOR_SEMANTIC_REVIEW`; loss reduction alone never
establishes bird/background separation.

Required ablation: None in this Stage D onset run. Final ablations remain Stage
E work and are not authorized.

## D-005 — Restart formal onset with bounded checkpointed 2,048-ray fast path

Date: 2026-07-03

Question: How should the formal T-onset trajectory be restarted after the user
stopped v1 because its 512-ray checkpointed path projected an impractical run
time?

Observed evidence: v1 was interrupted by explicit user request after 18
complete updates at global 15,018. It wrote no checkpoint, PLY, or review node
and is never a resume source. Excluding the first JIT-warmup step, its median
step time was 6.340 seconds, projecting 8.81 hours for 5,000 steps. Step time
correlated 0.9700 with the number of valid T-domain pixels. Peak allocated and
reserved memory were only 3.75/4.46 GiB. This isolates small checkpointed chunk
count/recomputation as the dominant variable rather than capacity pressure.

Chosen implementation: Preserve v1 unchanged and start a new v2 output from the
original global-15,000 Stage-B checkpoint with fresh T using the exact D-004
initialization and schedule. Use checkpointed 2,048-ray chunks as the fast path
for R, first-bounce T, and second-bounce D. If CUDA OOM occurs before any
optimizer, topology, checkpoint, telemetry, or debug commit, clear all D/R/T
and exposure gradients, release unused cache, and recompute the same selected
camera at checkpointed 1,024 then 512. Exhausting 512 is a hard failure. Camera
selection, local/global counters, learning-rate updates, and RNG are committed
only once outside the attempt sequence.

At a completed-step boundary, delete all forward/loss/checkpoint references and
call `empty_cache()` only when device-free memory is below 2 GiB. Debug renders
remain at the smoke-verified 512-ray setting and do not alter training state.
Telemetry records the successful chunk, every OOM retry, peak allocation, and
whether pressure cache release occurred.

Alternatives: Continue v1 for 8.8 hours; resume its uncheckpointed step 18;
disable checkpointing; change resolution/model counts/losses; use data-parallel
training; or enlarge chunks without an OOM fallback.

Why: A 2,048-ray batch reduces Python/checkpoint/LBVH chunk launches by roughly
four while preserving every ray and the same candidate and exact-intersection
definitions. Checkpointing bounds retained graphs; the same-camera fallback
restores the verified 512 endpoint if a view is unusually expensive. Pressure-
only cache release reuses the Stage-B v4 evidence that unconditional per-step
release adds substantial latency and does not bound a single live graph.

Paper fidelity: Chunk grouping can change floating-point accumulation order but
does not change represented rays, candidates, intersections, BRDF/BTDF,
alpha-over, loss, optimizer, topology rule, data, mask, or geometry. This is a
new v2 runtime identity and is never mixed with v1 updates.

Impact: The sole restarted output is
`output/stage_d_tihubird_c03r8_formal_onset_g15000_g20000_v2`. It still stops at
global 20,000, keeps L_depth disabled until the future global-40,000 boundary,
and runs the same checkpoint/debug/audit nodes. V1 remains aborted evidence.

Required ablation: None. Runtime equivalence is covered by the existing
checkpointed ray output/gradient tests plus Stage-D same-camera retry tests.

## D-006 — Cached T-only warm-up followed by exact uncached joint fine-tuning

Date: 2026-07-03

Question: How should fresh T receive a useful onset trajectory without paying
for unchanged D rasterization, global R tracing, and second-bounce D tracing on
every early update?

Chosen implementation: Supersede the uncompleted all-joint v2 trajectory with
a new, non-resuming formal output. It starts from the same immutable Stage-B
global-15,000 checkpoint and `stage_c_geometry_release_v1`, creates the same
fresh T (`random_bbox`, 4,096, seed 20260703), and ends at global 20,000.

Phase A is global 15,001--18,000. D/R parameters, optimizers, learning-rate
schedulers, densification statistics, pruning state, topology, and R-local
iteration remain frozen. Per-view FP32 caches contain only D/R/geometry values
that are mathematically independent of T; they contain no GT/target RGB. T
ray tracing, alpha-over, full-frame RGB loss, backward, optimizer, scheduler,
densification, and topology remain live. Cache identity binds the source
checkpoint, geometry release, reviewed mask manifest, renderer/ray/BRDF
configuration, camera identity, and schema. Any mismatch fails closed.

Before Phase A, nine fixed views plus one deterministic random view compare the
frozen uncached path against the cached path for final RGB, Cin/Ain/Din, Ct/At,
every loss term, and T xyz/rotation/scale/opacity/color gradients. Maximum and
mean absolute tolerances are `2e-5` and `2e-6`. D/R full-state hashes before and
after preflight and Phase A must be identical.

Phase B is global 18,001--20,000. It makes D/R trainable, disables the static
cache completely, and invokes the existing exact D-raster → global-R →
first-bounce-T → second-bounce-D → alpha-over composition and joint backward.
R-local stays 12,000 through Phase A and reaches 14,000 after Phase B; Phase A
must never be described as full joint training.

Alternatives: Continue the slow all-joint v2 attempt; change resolution, ray
sampling, candidate/intersection definitions, BRDF/BTDF, alpha-over, masks, or
geometry; or train T against cropped/partial RGB.

Why: D/R and their derived rays/contributions are invariant while those fields
are frozen, so caching removes repeated work without approximating T or the
final renderer. The explicit exact Phase B restores all cross-branch gradients
before the onset endpoint.

Paper fidelity: No rendering, ray, candidate, intersection, BRDF/BTDF,
alpha-over, mask, resolution, or data definition changes. This is a training
schedule and execution-layout decision. `L_depth` remains disabled through
global 20,000 and retains its future global-40,000 activation with
`lambda_depth=0.2`.

Impact: The unique output is
`output/stage_d_tihubird_c03r8_cached_twarmup_then_joint_g15000_g20000_v1`.
The single operator performs preflight, cache build, parity, no-update
performance benchmark, both phases, CPU-only audit, and final report. It never
continues beyond global 20,000 or enters Stage E.

Required ablation: None in Stage D. The cached and uncached parity gate is an
equivalence test, not a quality ablation.

## D-007 — Preserve failed cached-T v1 and retry debug-only fix as v2

Date: 2026-07-03

Question: How should the cached-T operator recover after its first review node
failed despite successful cache parity and finite Phase-A updates?

Observed evidence: v1 built and validated all 111 caches, passed the nine fixed
plus one random-view parity gate with zero output/loss differences and T-gradient
differences below `7.3e-12`, and measured a 254.95 ms cached median versus
1,909.46 ms frozen-uncached median on view 000018 (7.49x). It completed 25 T
optimizer updates and wrote a finite global-15,025 checkpoint, but telemetry was
committed only through global 15,024 because review export follows checkpoint
save and precedes telemetry commit.

The failure was confined to debug statistics: `camera.specular_mask` is CHW
`[1,269,478]`, while transparent-region RGB L1 expanded it directly against
HWC `[269,478,3]`. No renderer, loss, gradient, optimizer, cache, geometry, or
training tensor failed.

Chosen implementation: Preserve v1 unchanged and forbid resume. Normalize only
the debug-statistics mask to HWC before RGB boolean expansion, with CPU tests
for CHW/HWC inputs. Retry from the original Stage-B global-15,000 source and
fresh T in a new `..._v2` output. The v2 operator fail-closed audits the exact
v1 telemetry, log, checkpoint hash, source/release identity, parity verdict,
and exception before launching.

Alternatives: Resume the global-15,025 checkpoint; overwrite v1; omit the first
review node; or change training/cache mathematics.

Why: The checkpoint was written before an incomplete review/telemetry commit,
so it is not an atomic operator resume boundary. A fresh v2 preserves evidence
and deterministic schedule semantics. The mask conversion repairs only output
observability.

Paper fidelity: Debug layout only. No image formation or training mathematics
change.

Impact: v1 remains `CACHED_T_WARMUP_BLOCKED`; the only retry output is
`output/stage_d_tihubird_c03r8_cached_twarmup_then_joint_g15000_g20000_v2`.

Required ablation: None.

## D-008 — Semantic-repair v3 spatial responsibility filters and anti-veil prior

Date: 2026-07-04

Question: How should Stage D falsify the three observed cached-v2 shortcuts:
R representing cuboid-internal content, second-bounce Cout returning internal
content, and T becoming a near-opaque black veil?

Observed evidence: The read-only v2 causal audit records final transparent-mask
Ain mean/p50/p95/p99 `0.9584/1/1/1`, saturation coverage `0.9517`, conditional
Cin/Ain luminance mean `0.00774`, and high-Ain/near-black fraction `0.9001`.
T pruning starts at T-local 200 (4,096 to 2,531) and finishes at 540 surfels.
Classifying the saved v2 world-space T positions gives only 34 inside, 27
interface, and 479 outside. V2 did not save class-filtered R or Cout traces, so
their inside/interface/outside source energies are explicitly an evidence gap;
no independent versioned bird ROI exists.

Chosen implementation: Introduce `rtgs_cuboid_space_v1`, shared by D/R/T. For
orthonormal cuboid axes and local fitted bounds, signed clearance is
`min(local-lower, upper-local)` over all six planes. With default margin
`m=0.05` and epsilon `e=1e-6`, inside means clearance `>m+e`, outside means
clearance `<-(m+e)`, and the remainder is the separately reported interface
band. The named policy is
`transparent_interface_margin_mode=exclude`: interface surfels are never
silently assigned to inside or outside. The margin is in the release's
orthonormal local/world length units, is checkpointed, cache-identity-bound,
and does not claim a physical glass thickness.

Only inside the reviewed transparent mask, formal R replaces the unfiltered R
trace with a trace whose candidates are current-position outside-class R
surfels. Outside the mask the original global-R result is unchanged. Formal
second-bounce Cout similarly accepts only outside-class D candidates. Candidate
generation, exact intersection, ray origins/directions, BRDF/BTDF, alpha-over,
and the immutable Stage-C cache are unchanged. Unfiltered, inside, interface,
outside, and final-filtered traces are retained separately with surfel,
candidate, hit, and energy statistics.

Fresh T stores an unconstrained cuboid-local latent `z` and renders positions
as `local=(lower+m+4e)+sigmoid(z)*(upper-lower-2(m+4e))`, then rotates to world
space. Thus finite or saturated latent values remain strictly in the
inside-safe class without projection or optimizer-state surgery. The
1,000-step pilot fixes T at 4,096 and disables both pruning and densification;
any count or spatial-class drift is a hard failure.

The versioned engineering prior `rtgs_stage_d_anti_veil_v1` uses luminance
`Y`, `Ccond=Cin/clamp(Ain,1e-6)`, soft gates
`b=sigmoid((Ygt-0.15)/0.05)`, `h=sigmoid((Ain-0.80)/0.05)`, and
`s=sigmoid((Ain-0.95)/0.05)`. Over valid-two-hit transparent-mask bright
support `M=mask*valid_two_hit` with `Z=sum(M*b)`:

```text
d = 0.02 * softplus((0.08 - Y(Ccond))/0.02)
L_black = sum(M * b * h * d) / clamp(Z, 1e-6)
q = sum(M * b * s) / clamp(Z, 1e-6)
L_sat = 0.02 * softplus((q - 0.35)/0.02)
L_anti = ramp(T-local) * (0.05 * L_black + 0.02 * L_sat)
ramp(k) = smoothstep(clamp(k/200, 0, 1))
```

The bright-GT gate avoids penalizing legitimately dark content, the conditional
color test targets black opacity rather than copying GT into T, and the 35%
soft saturation allowance permits localized high-Ain objects such as a bird
instead of globally forcing Ain down. This is an explicit project engineering
prior, not an RT-GS paper loss or branch-level GT label. Parity preflight
records nonzero T-opacity/T-color gradients where mathematically applicable and
requires no D/R gradient leakage.

Alternatives: Enable L_depth early; add an unsupported D-zero loss; derive a
bird ROI from T predictions; project T after optimizer updates; prune T as in
v2; or change mesh, rays, resolution, sampling, BRDF, or alpha-over.

Why: Each mechanism directly blocks one observed shortcut while preserving the
final renderer definitions outside the transparent semantic-decomposition path.
The pilot is deliberately falsification-oriented: absence of an independent
bird ROI forces at least `SEMANTIC_REPAIR_PILOT_HOLD` even when all technical
constraints pass.

Impact: The only new executable experiment is a fresh 1,000-step cached-T-only
pilot from the immutable Stage-B global-15,000 source at
`output/stage_d_tihubird_c03r8_semantic_repair_v3`. D/R and their entire state
remain frozen, R-local remains 12,000, L_depth remains off, and the operator
cannot enter joint Phase B, global 16,001, Stage E, or resume any v2 checkpoint.

Limitation: Spatial source filtering is necessary but not a semantic label. In
the v2 final checkpoint all 3,088 R surfel centers already classify outside, so
an outside R surfel could still memorize a view-dependent bird appearance;
likewise an outside D surfel may encode colors that visually resemble internal
content. The v3 pilot therefore exports cross-view contribution maps and cannot
claim responsibility separation from the filter or loss alone.

Required ablation: This pilot is a semantic falsification gate, not a final
ablation. Any later comparison or joint continuation requires a separate user
decision after nine-view review.

## D-009 — Cuboid-front transparent path and permanent support-safe ownership

Date: 2026-07-04

Question: How should Stage D remove the D-G-buffer path contamination and
branch non-identifiability that remained after the v3 center-space filters and
anti-veil prior?

Observed evidence: The user-executed v3 completed 1,000 T-only updates with D/R
state hashes unchanged, T fixed at 4,096, no topology events, and every T center
inside the cuboid. It nevertheless ended `SEMANTIC_REPAIR_PILOT_BLOCKED`:
transparent Ain mean/p50/p95/p99 was 0.9729/0.99997/1/1, saturation fraction
was 0.8744, and the high-Ain near-black conditional fraction was 0.9723. Its
first bounce still originated at the transparent D G-buffer position, D direct
and R remained formal contributors, and spatial ownership used centers rather
than finite surfel support. Therefore v3 is immutable failed evidence and is
not a resume source.

Chosen implementation: Add the versioned `cuboid_front_v1` path only on
`mask_hard & valid_two_hit`. Frozen camera direction is normalized from camera
center to frozen `back_position`; `front_position=camera_center+t_near*d`.
The nearest of the six fitted cuboid planes selects the entering-face normal in
cuboid-local coordinates; it is transformed by the orthonormal axes and flipped
when necessary so `dot(front_normal,-d)>0`. Non-finite fields, invalid depth
ordering, a front point farther than 5e-4 from a cuboid plane, or a missing D
material surface fail the run rather than falling back to D position/normal.

For valid transparent rays, R origin, incident direction, outgoing direction,
reflection direction, Fresnel normal, and microfacet normal all use this frozen
front contract. T starts at `front_position+eps*d`; its absolute depth is
`t_near+eps+relative_T_depth`, in the same coordinate system as `t_far`.
Second-bounce Cout retains `back_position+eps*d`. Outside the hard transparent
mask the existing D/R path and contribution tensors remain unchanged.

All D/R/T ownership now uses one cuboid-local finite 3-sigma tangent-support
partition: `strict_inside_safe`, `interface_margin`, `strict_outside_safe`, and
`crossing_or_ambiguous`. The pilot sets D direct and R contribution to `off` on
the entire hard transparent mask, admits only strict-outside-safe D support to
Cout, and constrains all 4,096 fixed T supports to strict-inside-safe through a
rotation/scale-aware sigmoid parameterization. The configurable modes are part
of the renderer/cache/checkpoint contract and are intended to survive later
joint fine-tuning; re-enabling a contribution requires a later explicit
decision and new evidence. The strong D/R-off policy is an ownership-isolation
diagnostic, not a claimed final physical renderer.

The sole matched ablation compares `random_strict_inside` with
`transferred_d_inside`. Transfer eligibility requires strict-inside-safe D
support, nonzero attributed contributions from at least three cached training
views, aggregate weight at least 0.01, and attributed absolute depths inside
`[t_near+0.05,t_far-0.05]`. It copies D xyz, quaternion, raw 2D scale, and raw
base color, but not roughness/f0/ks or optimizer/topology state. Opacity is
recalibrated as `clamp(0.25*sigmoid(D_raw),0.005,0.05)`; a deterministic
strict-inside random fill reaches 4,096 if eligible D is insufficient. This is
an initialization hypothesis, never a fact label; D is copied, not removed.

The new static cache schemas are `rtgs_stage_d_cuboid_front_cache_v4` and
`rtgs_stage_d_cuboid_front_cache_manifest_v4`. Their identity binds source,
release, masks, cuboid-front path, ownership modes, support rule, and renderer
configuration, making all legacy D-origin caches incompatible. A single
fresh-only operator runs parity, Arm A 500 steps, Arm B 500 steps, and CPU audit
at nodes 15,000/15,100/15,250/15,500. Both arms share the camera deck, losses,
seed, T count, and all training settings; only T initialization provenance and
the mechanically necessary Arm-B cache reuse flag/output path differ.

Alternatives: Continue v3 longer; enable L_depth early; retain center-only
filtering; use anti-veil as the primary repair; delete D surfels globally; or
derive a bird ROI from model/GT appearance. These do not remove the path and
ownership ambiguity or would create unsupported semantic supervision.

Impact and boundary: The only authorized output is
`output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4`. L_depth stays
disabled (future marker global 40,000). No joint fine-tune or Stage E follows.
Without an independent, versioned, human-authored bird ROI the audit must HOLD
or BLOCKED. Even PASS means only that T has testable handoff conditions under
the cuboid-front path and strong ownership isolation, not final D/R/T separation.

Required ablation: The two matched 500-step arms above; no conclusion may be
drawn from one arm alone or from RGB loss alone.

## D-010 — Preserve the zero-step v4 launch failure and retry the canonical output

Date: 2026-07-04

Question: How should the v4 operator recover after its first user launch failed
before cache construction or any optimizer update?

Observed evidence: Commit `b697c25` guarded support-safe T initialization on a
non-null release cuboid, but `training_stage_d` constructed that cuboid only for
`stage_d_semantic_repair_pilot`, omitting `stage_d_ownership_pilot`. Arm A
therefore stopped during source-checkpoint initialization with
`support-safe T initialization requires cuboid space`. The failed output has no
cache manifest, checkpoint, telemetry, debug node, or training update; source
and release before/after hashes remain exact. Its audit is correctly BLOCKED
but its many missing-file errors are downstream consequences, not additional
runtime failures.

Chosen implementation: Define one tested cuboid-requirement predicate shared by
semantic-repair and ownership modes, use it before Stage-D initialization, and
make ownership contract validation require `rtgs_cuboid_space_v1` metadata.
The canonical output name remains unchanged. On the next identical operator
invocation only, the wrapper may atomically rename this exact known b697c25
zero-step failure to
`stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4_failed_preflight_b697c25`
before creating a fresh canonical directory. Eligibility binds the commit,
exception, Arm-A exit, source/release hashes, and absence of cache/checkpoint/
telemetry. Any other existing output, or an existing archive, still fails
closed. Nothing is deleted, resumed, or overwritten.

Impact: This is a launch-gate correction only. Cuboid-front formulas, ownership
policy, A/B variables, schedule, cache schema, losses, nodes, and acceptance
boundary are unchanged. Codex does not launch the retry.

Required ablation: None; unit tests cover both ownership cuboid construction
eligibility and the exact zero-step archival guard.

## D-011 — Bound active T support scale and restart the matched v4 A/B

Date: 2026-07-04

Question: How should the ownership pilot preserve strict 3-sigma legality when
the transferred arm's scale optimizer grows a surfel beyond cuboid capacity?

Observed evidence: The user retry on commit `17d2663` built all 111 v4 caches,
passed parity, and completed Arm A through global 15,500. Arm B passed parity
and the 15,000/15,100 nodes, then stopped after telemetry global 15,114 when
`support_safe_local_bounds` found no feasible center for the updated raw scale.
The last saved 15,100 checkpoint has all 4,096 supports strict-inside and needs
no cap; the failure appeared during later scale growth. Source and release
hashes remain exact. Missing endpoint hashes/files and camera-deck mismatch in
the final audit are consequences of the interrupted Arm B, not evidence that
D/R actually changed.

Chosen implementation: Keep `_scaling` as the copied/optimized raw log-scale,
including exact D raw-scale transfer at initialization. For
`cuboid_inside_support_sigmoid_v2`, derive active positive 2D scale by computing
the local-axis 3-sigma radii for the current quaternion, comparing them with
the cuboid half-extents after interface/epsilon padding, and applying one common
factor in `(0,1]` to both tangent scales. Feasible values are exactly unchanged;
only a would-be infeasible support is uniformly reduced, preserving anisotropy.
An additional 4-epsilon capacity reserve keeps the center interval strictly
positive. The operation is tensor-native and differentiable almost everywhere;
raw optimizer state is neither projected nor rewritten.

Checkpoint identity records `cuboid_support_uniform_cap_v1`. Telemetry/debug
record capped count, minimum factor, and raw/active maximum scale. The CPU audit
reconstructs active scale with the same rule before checking all 4,096 supports.
This bound enforces representational legality; it does not assert semantic
correctness or change D/R/Cout ownership.

Alternatives: Allow T support to cross the cuboid; prune the large surfel;
project raw scale after every update; resume the interrupted Arm B; or reuse
Arm A from a different code commit. These violate the fixed-support, fresh-state,
or matched-code contracts.

Impact: The exact `17d2663` failure is preserved under suffix
`_failed_scale_17d2663`. The canonical v4 directory is rebuilt from global
15,000 and both arms rerun under one new commit; no cache, checkpoint, optimizer,
or Arm-A result from the interrupted attempt is reused. All renderer, loss,
schedule, A/B-variable, and verdict contracts remain unchanged.

Required ablation: The already required matched A/B; synthetic tests additionally
force extreme raw scale and verify finite strict-inside active support without
changing copied feasible raw scales.
