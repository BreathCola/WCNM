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
