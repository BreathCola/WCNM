# Stage B — Differentiable Ray Tracing and Reflection

## Goal

Add an independent Reflection 2D surfel field, a shared differentiable Gaussian
ray tracer, reflection-ray generation, full microfacet reflection shading, and
the minimum valid specular-mask constraint.

Stage B solves reflection only. It does not solve or claim:

```text
Transmittance Gaussian
bird/background separation
transparent mesh extraction
two-hit geometry or cache
mesh-guided second bounce
full D + R + T reconstruction
```

Stage C and Stage D have not started.

## Authorized entry state

- Stage A is formally closed by explicit user authorization on 2026-06-30.
- Current branch: `feature/stage-b-reflection-dr-c03`.
- Frozen Stage A StableNormal D-only baseline:

  ```text
  baseline/stage-a-stablenormal
  output/stage_a_tihubird_30k_mono001_r2_retry1/
  ```

- First Stage B hypothesis uses this read-only Diffuse initialization candidate:

  ```text
  output/stage_a_tihubird_drnormal_c03_15k/chkpnt15000.pth
  output/stage_a_tihubird_drnormal_c03_15k/point_cloud/iteration_15000/
  ```

- C03 15k is not a proven replacement for the StableNormal D-only baseline.
  A matched Stage B comparison may be considered later, but no second training
  is authorized for the first experiment.
- No Reflection model instance, ray tracer, BRDF implementation, mask pipeline,
  or Stage B training exists at entry.

## B-0 — Read-only implementation audit

- [ ] Read `AGENTS.md`, `RTGS_MASTER_PLAN.md`, `docs/STATUS.md`,
  `docs/DECISIONS.md`, and this file.
- [ ] Audit `train.py`, `render.py`, `scene/diffuse_surfel_model.py`,
  `gaussian_renderer/`, checkpoint/load/resume logic, and the current
  position/normal/roughness/f0/ks maps.
- [ ] Specify the independent Reflection model, optimizer, densification, and
  checkpoint contracts without implementing them.
- [ ] Specify how D resumes from C03 15k while R starts independently.
- [ ] Specify the minimum correct differentiable tracer path, ray convention,
  epsilon, miss/background behavior, and microfacet inputs/outputs.
- [ ] Specify the minimum honest specular constraint possible before a final
  transparent mask exists.
- [ ] Define B-1 file scope, tests, debug maps, smoke acceptance, command, and
  operator-run prerequisites.
- [ ] Do not modify rendering code or run training during B-0.

## B-1 — Minimal implementation sequence

The concrete sequence and file list are determined by the completed B-0 audit.
Implementation must remain incremental and tested; no later-stage dummy outputs
may be introduced.

### Reflection model contract

- [ ] Add `scene/reflection_surfel_model.py` with independent `xyz`, `rotation`,
  `scaling_2d`, `opacity_raw`, and `color_raw` tensors.
- [ ] Give R its own optimizer, scheduler, densification/pruning state,
  serialization namespace, and exported PLY.
- [ ] Never merge R tensors or optimizer state into D arrays.

### Differentiable ray tracer contract

- [ ] Provide the shared `raytrace(model, origins, directions, ...)` contract
  returning color, alpha, expected depth, and hit mask.
- [ ] Support gradients required by the master plan, chunked rays, and an
  acceleration structure whose state is synchronized after parameter changes.
- [ ] Keep any PyTorch brute-force tracer test-only and small-scene-only.

### Reflection ray and shading contract

- [ ] Generate reflection rays from Diffuse position and face-forward normal
  using the master-plan direction convention.
- [ ] Offset every origin by scene-scaled epsilon and define explicit
  hit/miss/background behavior.
- [ ] Implement GGX/Trowbridge-Reitz D, Schlick F, Smith GGX G, and the complete
  non-split-sum reflection weight.
- [ ] Keep Stage B composition limited to
  `C = (1 - ks) * Cd + ks * wr * Cr`.

### Specular constraint contract

- [ ] Load, validate, and visualize only a real mask source accepted for the
  Stage B smoke.
- [ ] Never fabricate a transparent mask or claim a temporary proxy is the
  final Stage C mask.
- [ ] Apply `L_spec` only where the accepted mask is valid and keep mask use out
  of full-image RGB cropping.

## Debug outputs required before Stage B acceptance

```text
ground_truth.png
final.png
diffuse_color.png
diffuse_depth.png
alpha.png
normal.png
roughness.png
f0.png
ks.png
reflection_color.png
reflection_alpha.png
reflection_depth.png
reflection_hit_mask.png
transparent_mask.png       # only after an accepted real mask exists
overlay.png                 # only after an accepted real mask exists
diffuse_contribution.png
reflection_contribution.png
```

No inside/outside/transmittance, near/far, two-hit, mesh, or depth-violation
dummy outputs are permitted in Stage B.

## Tests required before Stage B acceptance

- [ ] Reflection model activation, shapes, independent storage, optimizer, PLY,
  densification/pruning, and checkpoint round-trip tests.
- [ ] Reflection direction, face-forward normal, epsilon, hit/miss, and
  background tests.
- [ ] Raytrace output-contract, chunking equivalence, finite output, gradient,
  and acceleration-state synchronization tests.
- [ ] GGX finite-value, limiting-angle, Fresnel monotonicity, and full
  composition tests.
- [ ] Real-mask loading/validation and `L_spec` masking tests before enabling
  the specular constraint.
- [ ] Existing Stage A regression suite.

## Acceptance checklist

- [ ] Independent Reflection Gaussian exists.
- [ ] Ray tracer runs on R and satisfies its differentiability contract.
- [ ] Reflection-ray direction and scene-scaled epsilon are correct.
- [ ] Reflection color/alpha/depth/hit maps are real and visualized.
- [ ] D and R optimizers, densification, checkpoints, and exports are separate.
- [ ] Full microfacet BRDF runs without split-sum approximation.
- [ ] The accepted transparent-region mask raises ks through the documented
  specular constraint.
- [ ] Outputs and gradients contain no NaN or Inf.
- [ ] Stage A functionality and the original 3DGS baseline do not regress.
- [ ] The first Stage B smoke is run by the user from reviewed prerequisites and
  produces the required debug evidence.
- [ ] `docs/STATUS.md`, `docs/DECISIONS.md`, and this checklist reflect verified
  reality before any Stage C transition.
- [ ] A rollback commit exists.

The project remains in Stage B until every applicable item above is verified.
Passing B-0 alone does not authorize Stage C or Stage D.
