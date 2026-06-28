# RT-GS Engineering Decisions

This log records implementation choices required where the paper or master plan
does not fully specify behavior. New entries must use the decision template from
`RTGS_MASTER_PLAN.md`.

No engineering decisions outside Stage A have been made.

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

Required ablation: Compare Stage A training with `lambda_mono=0` and the same
training with these priors before accepting their quality contribution.

## A-010 — Matched monocular-normal comparison protocol

Date: 2026-06-28

Question: How should the effect of the completed StableNormal prior set be
measured without confounding it with resolution, initialization, schedules, or
debug/checkpoint differences?

Chosen implementation: Run two sequential 30,000-iteration, native-resolution
truck trainings under the repository's deterministic seed. Both runs load the
same complete camera-space prior set and use identical Stage A settings,
including `lambda_norm=0.04`, `lambda_perc=0.01`, debug output every 1,000 steps,
test/save milestones at 7,000/15,000/30,000, and checkpoints at
10,000/20,000/30,000. The only loss-setting difference is `lambda_mono=0` versus
`lambda_mono=0.01`; output directories and log paths necessarily differ.

Alternatives: Compare against the earlier resolution-divisor-8 run; omit loading
priors in the zero-weight run; change random seeds; or enable different
checkpoint/debug schedules.

Why: Loading the same data in both runs preserves memory and data-path behavior,
while deterministic initialization and identical schedules isolate the
monocular-normal coefficient.

Paper fidelity: This is a Stage A engineering evaluation protocol. It does not
change the loss formula, model, renderer, or training implementation.

Impact: The comparison requires two long runs and approximately doubles the
target-quality compute and output storage. It remains unrun pending user review
of the prior contact sheet.

Required ablation: The two runs defined above are the required ablation.
