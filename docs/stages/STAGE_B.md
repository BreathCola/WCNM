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
  or Stage B training existed at entry. B-1 implementation now exists, but no
  TiHuBird Stage B optimization step has completed. The first user smoke attempt
  stopped at 0/2 steps on the CUDA JIT non-ASCII-path issue recorded in B-007.

## B-0 — Read-only implementation audit

- [x] Read `AGENTS.md`, `RTGS_MASTER_PLAN.md`, `docs/STATUS.md`,
  `docs/DECISIONS.md`, and this file.
- [x] Audit `train.py`, `render.py`, `scene/diffuse_surfel_model.py`,
  `gaussian_renderer/`, checkpoint/load/resume logic, and the current
  position/normal/roughness/f0/ks maps.
- [x] Specify the independent Reflection model, optimizer, densification, and
  checkpoint contracts without implementing them.
- [x] Specify how D resumes from C03 15k while R starts independently.
- [x] Specify the minimum correct differentiable tracer path, ray convention,
  epsilon, miss/background behavior, and microfacet inputs/outputs.
- [x] Specify the minimum honest specular constraint possible before a final
  transparent mask exists.
- [x] Define B-1 file scope, tests, debug maps, smoke acceptance, command, and
  operator-run prerequisites.
- [x] Do not modify rendering code or run training during B-0.

User approved B-0 before B-1 implementation began.

## B-1 — Minimal implementation sequence

The concrete sequence and file list are determined by the completed B-0 audit.
Implementation must remain incremental and tested; no later-stage dummy outputs
may be introduced.

### Reflection model contract

- [x] Add `scene/reflection_surfel_model.py` with independent `xyz`, `rotation`,
  `scaling_2d`, `opacity_raw`, and `color_raw` tensors.
- [x] Give R its own optimizer, scheduler, densification/pruning state,
  serialization namespace, and exported PLY.
- [x] Never merge R tensors or optimizer state into D arrays.

### Differentiable ray tracer contract

- [x] Provide the shared `raytrace(model, origins, directions, ...)` contract
  returning color, alpha, expected depth, and hit mask.
- [x] Support gradients required by the master plan, chunked rays, and an
  acceleration structure whose state is synchronized after parameter changes.
- [x] Keep any PyTorch brute-force tracer test-only and small-scene-only.

### Reflection ray and shading contract

- [x] Generate reflection rays from Diffuse position and face-forward normal
  using the master-plan direction convention.
- [x] Offset every origin by scene-scaled epsilon and define explicit
  hit/miss/background behavior.
- [x] Implement GGX/Trowbridge-Reitz D, Schlick F, Smith GGX G, and the complete
  non-split-sum reflection weight.
- [x] Keep Stage B composition limited to
  `C = (1 - ks) * Cd + ks * wr * Cr`.

### Specular constraint contract

- [ ] Load, validate, and visualize only a real mask source accepted for the
  Stage B smoke.
- [x] Never fabricate a transparent mask or claim a temporary proxy is the
  final Stage C mask.
- [x] Apply `L_spec` only where the accepted mask is valid and keep mask use out
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

- [x] Reflection model activation, shapes, independent storage, optimizer, PLY,
  densification/pruning, and checkpoint round-trip tests.
- [x] Reflection direction, face-forward normal, epsilon, hit/miss, and
  background tests.
- [x] Raytrace output-contract, chunking equivalence, finite output, gradient,
  and acceleration-state synchronization tests.
- [x] GGX finite-value, limiting-angle, Fresnel monotonicity, and full
  composition tests.
- [ ] Real-mask loading/validation and `L_spec` masking tests before enabling
  the specular constraint.
- [x] Existing Stage A regression suite: full repository result is 67 passed.

## Acceptance checklist

- [x] Independent Reflection Gaussian exists and passed synthetic tests.
- [x] Ray tracer runs on R and satisfies its tested differentiability contract.
- [x] Reflection-ray direction and scene-scaled epsilon are correct in tests.
- [ ] Reflection color/alpha/depth/hit maps are real and visualized.
- [x] D and R optimizers, densification, checkpoints, and exports are separate.
- [x] Full microfacet BRDF runs without split-sum approximation in tests.
- [ ] The accepted transparent-region mask raises ks through the documented
  specular constraint.
- [ ] Outputs and gradients contain no NaN or Inf.
- [x] Stage A test suite and original code paths do not regress in the 67-test run.
- [ ] The first Stage B smoke is run by the user from reviewed prerequisites and
  produces the required debug evidence.
- [ ] `docs/STATUS.md`, `docs/DECISIONS.md`, and this checklist reflect verified
  reality before any Stage C transition.
- [x] Separate B-1a, B-1b, and B-1c rollback commits exist.

The project remains in Stage B until every applicable item above is verified.
Passing B-0 alone does not authorize Stage C or Stage D.

## B-1 implementation commits and current stop point

```text
7ad0b8881b9b2a00e9f766b68be7ff051c74281e  B-002 through B-005 decisions
bb5eb072e757a32735ad378b38a3d4b489ac1a57  B-1a
30dd8be5d05d6da9081b0a1f4b80a555234b74f3  B-1b
953334009a8db3d63cc68179842687ba306186c2  B-1c
c6b918442eb1891eeba1c5914a8db0b85763ec4b  checkpoint schedule metadata fix
f1e90e780eb4777ddeeece70bc393e0b21b080db  on-disk checkpoint resume test
36b7fdfdd98ec7f3d04d256cf76830189a6a10a9  ASCII-only CUDA JIT staging fix
```

Implemented and tested does not mean accepted. The first TiHuBird Stage B smoke
attempt was user-run but aborted before step one and produced no checkpoint or
D/R PLY save. Its retry and review of real debug outputs remain user-only.
