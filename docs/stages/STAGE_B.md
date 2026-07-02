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
  long Stage B training has run. The first user smoke attempt stopped at 0/2
  steps on the CUDA JIT non-ASCII-path issue recorded in B-007; the preserved
  `smoke_2_retry1` completed 2/2 steps and was reviewed before B-1d began.

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

- [x] Load, validate, and visualize only a real mask source accepted for the
  Stage B smoke.
- [x] Never fabricate a transparent mask or claim a temporary proxy is the
  final Stage C mask.
- [x] Apply `L_spec` only where the accepted mask is valid and keep mask use out
  of full-image RGB cropping.

### B-1d debug-only observability

- [x] Reuse LBVH offsets for exact per-ray candidate counts without a second
  traversal or any production fallback.
- [x] Report exact plane/ellipse intersection counts separately from final alpha
  hits.
- [x] Collect CUDA phase timing and peak-memory data only when diagnostics are
  explicitly requested by fixed debug/offline renders.
- [x] Preserve physical-scale images and add recorded p99/log display-only
  companions for sub-8-bit reflection contribution and unbounded microfacet D.
- [x] Record raw finite/count/min/mean/p50/p95/p99/max statistics and display
  scales without reading target-image colors.
- [x] Verify these diagnostics in a user-operated one-step TiHuBird resume.

The first diagnostic resume attempt stopped before iteration 15,003 because
checkpoint `map_location="cuda"` moved the CPU RNG ByteTensor onto CUDA. B-009
defines the corrected device contract; the real checkpoint now passes read-only
RNG restoration. That failed attempt is preserved separately from the corrected
retry below.

The corrected retry completed at global 15,003 / R local 3. Candidate
p50/p95/p99 is `111/205/256`, exact-intersection p50/p95/p99 is `28/64/76`,
raytrace wall time is about 152 ms for 128,582 rays, and incremental peak memory
is about 319 MiB. D/R checkpoint state is finite. Microfacet D lies in
`1.270–1.276`, so its nearly white map is a real low-variance result. The LBVH is
not a full-field brute-force traversal; R count and initial scale remain fixed.

### Bounded Tier-1 pilot observability

- [x] Add a default-off, explicitly named Stage B JSONL mode with a required
  positive maximum-step bound and caller-provided phase tag.
- [x] Keep it independent of `lambda_spec` and smoke diagnostics; never enable
  candidate/exact diagnostics, traversal timing, renderer work, backward, or a
  second CUDA traversal merely to populate telemetry.
- [x] Record D/R counts and deltas, real R topology version/event, D densify-call
  or count event, loss, camera/global/local iteration, CPU loop wall boundary,
  current allocator state, and correctly scoped allocator maxima.
- [x] Leave unavailable candidate/exact/forward/backward timing fields explicit
  and null on ordinary Tier-1 steps.
- [x] Keep Phase A mask-free. In Phase B, summarize the existing forward ks map
  and formal mask on CPU; label it as the pre-optimizer state consumed by total
  loss rather than claiming a post-update rerender or zero outside total gradient.
- [x] Add a CPU-only, read-only two-run comparison that uses one shared display
  scale per quantity and fails honestly when physical 8-bit maps lack spatial
  information.
- [x] Verify default-off behavior, loss/parameter/RNG invariance, Phase A/B mask
  policy, strict schema/bounds, allocator labels, and postprocess imports and
  input immutability.
- [x] User ran and Codex read-only audited matched 7k/10k/15k Phase A/B Tier-1
  pilots. Results are 7k/10k `CONDITIONAL PASS`, 15k `HEALTHY`; they do not
  answer long-horizon causal quality.

This instrumentation does not alter renderer, tracer, BVH, candidate/exact
logic, BRDF, losses, optimizer/scheduler, densification thresholds/timing,
checkpoint semantics, R initialization, or RNG consumption. Resetting PyTorch
peak allocator counters changes observability counters only. On ordinary steps,
the scoped maximum spans the complete loop boundary; on debug steps, maxima
from before debug begins and after the debug path's existing reset are combined,
while the transient interval before that internal reset is labeled unavailable.
No result from an earlier run is retroactively upgraded by this code.

### Tier-2 manually gated operator pack

- [x] Add opt-in version-2 D-only and Stage B continuation checkpoints with
  model/optimizer/densification, global/R-local, Python/NumPy/CPU/CUDA RNG,
  remaining camera deck, configuration, and completed-endpoint-update state.
- [x] Preserve the endpoint optimizer update only in explicit operator mode so
  a bounded resume follows the uninterrupted trajectory; retain legacy defaults.
- [x] Require full-state sources and exact expected global/R-local starts at
  every gate.
- [x] Verify fresh R uses its private seed/generator without changing restored
  global RNG or the shared checkpoint bytes.
- [x] Enforce no mask/L_spec before local 101 and the formal manifest with
  `lambda_spec=0.2` from local 101 onward.
- [x] Add bounded D-only telemetry without any R model/optimizer/renderer field.
- [x] Add a CPU-only, read-only gate packet tool covering checkpoint hash/finite
  scan, telemetry continuity, topology, losses/material, allocator, endpoint
  debug metadata, log anomalies, and missing artifacts.
- [x] Add a print-by-default shell wrapper for bootstrap, checkpoint protection,
  Branch A/B gates, global-1000 segments, and one-command audits. It never
  advances a later gate automatically.
- [x] Pass the GPU-enabled full 117-test repository suite and CPU-only 102-pass /
  15-skip suite without running training.
- [x] Resolve the Tier-2 image-resolution contract: the user approved the
  internally matched `C03-r8 Tier 2 onset study` at resolution 8 across the
  shared bootstrap and both branches. It is not a strict legacy C03-r2 replay,
  and operator mode rejects a mismatched experiment name or resolution.
- [x] Preserve the user-operated manual v1 bootstrap/partial branches as
  historical assets; do not resume, overwrite, or use them as one-shot inputs.

### Tier-2 one-shot onset study

- [x] User explicitly authorizes a fresh `oneshot_v1` C03-r8 run with one shared
  D-only 0--7000 bootstrap, Branch A onset at 3000, and Branch B onset at 7000.
- [x] Add `run-all`, `status`, and CPU-only `final-audit` orchestration without
  changing training mathematics, schedules, mask policy, or RNG semantics.
- [x] Fail closed on a dirty worktree, wrong experiment/resolution, unavailable
  RTX 3090 pair, conflicting training process, preexisting one-shot output,
  telemetry discontinuity/nonfinite state, abnormal exit, or source mutation.
- [x] Preserve complete checkpoints at bootstrap 1k--7k and branch local
  100/200/500/1000, every later global 1000, and global 15000; retain debug every
  100 global steps so all required nodes are covered.
- [x] Execute `oneshot_v1` through a hard failure and generate its CPU-only final
  report. The shared bootstrap and both warmups completed; both formal branches
  OOMed, so this is a HARD_FAILED experiment, not acceptance evidence.
- [x] Diagnose v1 without rerunning training: no nonfinite/checkpoint corruption
  or monotonic graph retention; camera/topology-dependent live peaks combine
  with severe allocator cache fragmentation in both branches.
- [x] Add an independent, fail-closed `oneshot_v2` allocator-lifecycle retry
  identity. Reuse only the immutable shared D-only 3k/7k sources; never resume
  failed v1 A/B checkpoints.
- [x] Apply the same `max_split_size_mb:128`, completed-step ephemeral-reference
  release, and safe-boundary `empty_cache` policy to both branches without
  changing renderer, ray/tracing definitions, loss, optimizer, scheduler,
  densification, RNG, resolution, R initialization, or R count.
- [x] Require a local-101 per-branch allocator headroom record and fail closed if
  projected capacity does not exceed the greater of the observed step peak and
  the 23,274,475,520-byte v1 reference by at least 1 GiB.
- [ ] User executes the committed v2 one-shot through both branches or a hard
  failure. The coordinator must automatically generate the CPU-only v2 final
  audit in either case. No Stage C/D work is allowed.

Gate decisions after the user returns one generated packet:

- `GO`: packet complete and continuous; checkpoint/runtime/RNG finite; no OOM or
  anomaly; counts/topology match the scheduled gate; candidate/exact, allocator,
  timing, material drift, and fixed debug remain controlled. This authorizes
  only the next named finite segment.
- `HOLD`: core state is finite but review evidence is incomplete, allocator or
  traversal trends need comparison, topology is elevated but explainable, or
  fixed-view human inspection is pending. No next command is authorized.
- `BLOCKED`: missing/corrupt checkpoint or telemetry, nonfinite/OOM, global or
  R-local discontinuity, source/config/GPU isolation failure, unexplained
  topology, approximately 2x candidate/exact escalation, or visible collapse.

Gate 0 additionally requires a D-only version-2 checkpoint with no R namespace.
Gate 1 requires R=4096/version 0 and null mask/L_spec fields. Gate 2 is the first
local-200 topology crossing and must explain the R count/version change before
either branch advances. Gates 3/4 stop at local 500/1000. Every later command is
exactly one global-1000 segment and requires a new returned packet.

### Formal-mask L_spec smoke

- [x] Archive the accepted 111 masks in an immutable versioned directory and
  validate exact sources, RGB/mask hashes, ordered stems, and padding exclusion.
- [x] Resume only the original 15,003/3 checkpoint and run exactly three added
  iterations with `lambda_spec=0.2` and the formal manifest.
- [x] Record finite positive L_spec and total loss at each step, nonzero
  inside-mask L_spec-only ks gradient, exactly zero outside-mask gradient, and
  the upward ks direction implied by gradient descent.
- [x] Preserve full-image RGB reconstruction, D/R counts, BVH behavior,
  checkpoint namespaces, and complete reflection debug outputs.
- [x] Save and recursively inspect the 15,006/6 checkpoint and independent D/R
  PLYs with no NaN/Inf.
- [x] User inspects `transparent_mask.png`, `overlay.png`, and the inside/outside
  ks statistics before any controlled 100-step lambda-spec pilot is authorized.

The clean smoke output is
`output/stage_b_tihubird_reflection_dr_c03_lspec_smoke_g15003_15006_v2/`.
The three positive L_spec values are 0.164784 / 0.156079 / 0.130580. Outside
mask L_spec gradient is exactly zero on all sampled valid D surfaces; inside
gradient is nonzero and points toward increasing under-threshold ks. Candidate/
exact distributions remain in the prior diagnostic range, warm raytrace wall is
about 218/208 ms, and peak allocated/reserved is 13.54/15.10 GB. D/R remain
346,118/4,096. The final finite checkpoint hash is
`f46c00375699d3b4b7c018a4277b4ba3e93abc66f60ddc7f282a0caf979b93fb`.
This is technical smoke evidence only; Stage B remains unaccepted.

### Controlled 100-step formal-mask L_spec pilot

- [x] Resume only the original read-only 15,003/3 checkpoint; do not resume from
  the 15,006 smoke checkpoint, health pilot, aborted runs, or profile outputs.
- [x] Keep all Stage B model, renderer, BVH, BRDF, optimizer, scheduler,
  densification, R count, chunk size, resolution, and RNG semantics unchanged;
  only add the formal `specular_masks_reviewed_v1/manifest.json` and
  `--lambda_spec 0.2`.
- [x] Run exactly global 15,004--15,103 / R-local 4--103 and stop.
- [x] Preserve debug nodes at 15,025 / 15,050 / 15,075 / 15,100 plus final
  15,103 debug; save only the 15,103 checkpoint and independent D/R PLYs.
- [x] Verify no crash/OOM, no non-finite checkpoint/debug tensors, no D/R count
  change, no candidate/exact-intersection explosion, and nonzero reflection
  contribution.
- [x] Distinguish the already verified L_spec-only zero outside-mask gradient
  from ordinary total-loss ks drift outside the mask.
- [x] Record that exact whole-step CUDA allocated/reserved peaks were not
  persisted by the non-smoke training path and therefore remain an evidence gap.
- [ ] User evaluates the fixed-view transparent masks, overlays, ks maps,
  reflection contribution, L_spec behavior, and memory telemetry caveat before
  any Stage B acceptance decision.

The pilot output is
`output/stage_b_tihubird_reflection_dr_c03_lspec_pilot100_g15003_15103_v1/`.
It completed all 100 added iterations with D/R counts fixed at 346,118/4,096.
The 95 ordinary TensorBoard scalar intervals have p50/p95/mean
`0.931/1.089/0.939 s`. Per-view L_spec is finite but not monotonic across
random sampled cameras: 0.164784 / 0.165822 / 0.261634 / 0.173927 / 0.126985 /
0.277022 at 15,004 / 15,025 / 15,050 / 15,075 / 15,100 / 15,103.

Fixed-view mask-inside debug-map ks mean rises from the original 15,003 mean
0.095482 to 0.100522 at 15,103, while fixed-view outside-mask mean changes only
from 0.087836 to 0.088187. Candidate p50/p95/p99 reaches 147/266/344 and exact
intersections reach 38/84/109 at 15,103, still far below full-field traversal.
Reflection contribution remains nearly fully nonzero with final mean/p99
`1.923e-4/1.171e-3`; all raw debug non-finite counts are zero.

Final checkpoint
`output/stage_b_tihubird_reflection_dr_c03_lspec_pilot100_g15003_15103_v1/chkpnt15103.pth`
has SHA-256
`c5e40e1a9025a3e191314759e8214e2eb11cba9e04ee2319c6f0c522e4cd6bca` and
recursive inspection found no NaN/Inf in 63 tensors / 19,912,887 elements. This
is evidence for human Stage B evaluation, not Stage B acceptance or long-run
authorization.

### Controlled 100-step health pilot

- [x] Resume exactly from global 15,003 / R local 3 without resetting any state.
- [x] Keep `lambda_spec=0` and load/create no mask.
- [x] Keep all model, renderer, BVH, densification, and training settings fixed.
- [x] Use the existing `--debug_interval 25`; debug at global 15,025 / 15,050 /
  15,075 / 15,100 is accepted and exact +25 alignment is not required.
- [x] Save checkpoint and independent D/R PLY only at 15,103.
- [x] Stop at global 15,103 / R local 103 and classify health without treating
  a flat 100-step PSNR as automatic failure.

The first attempt reached global 15,019 / R local 19 before receiving SIGINT
because ordinary steps took roughly 50--60 seconds. It is
`PILOT_ABORTED_FOR_PROFILE`, not `PILOT_HEALTHY` or `PILOT_BLOCKED`; preserve its
partial output and never resume from it. The original global 15,003 / R local 3
checkpoint remains the sole retry source.

The post-repair retry completed all 100 steps in the fresh
`health100_candidate_reduce_g15003_15103_retry1` directory. The 95 ordinary
iteration wall intervals have p50/p95/mean `0.939/1.092/0.944 s`; training
raytrace forward averages 228.7 ms and the candidate-backward envelope averages
643.0 ms. All losses and debug raw statistics are finite. D/R remain
346,118/4,096 with no densify/prune count change. Candidate and exact-count
quantiles grow gradually but remain far below a full-field scan. Reflection
contribution stays nonzero, and final fixed-view ks remains centered near 0.097
with no pixel below 0.01 or above 0.9.

Allocator peak allocated/reserved are 18,094,459,392 / 20,333,985,792 bytes.
Live allocated memory is not monotonic, while reserved memory grows in caching
steps to 20.33 GB. Therefore classify this run as a healthy short pilot with a
material long-run memory-headroom risk. It does not authorize a long run or
constitute Stage B acceptance.

### Candidate-gradient performance repair

- [x] Preserve candidate IDs/order, exact intersection, depth sort, alpha
  composite, BRDF, loss, chunk size, model state, and D ray gradients.
- [x] Gather one compact 13-channel raw R table per candidate instead of five
  independent activated parameter views.
- [x] Sort candidate IDs once per chunk and reduce all 13 gradient channels by
  Reflection surfel ID without per-candidate global atomics.
- [x] Match legacy forward values, full raytrace outputs, Stage B loss, R
  gradients, and ray origin/direction gradients in CUDA tests.
- [x] Pass the complete Stage A/Stage B suite: 73 tests.
- [x] Re-profile the same two steps from the original read-only 15,003/3
  checkpoint and verify the long candidate indexing backward kernels disappear.

For global 15,004, the matched Nsight interval fell from 50.461 s to 1.572 s.
The R backward envelope fell from 49.524 s to 0.547 s. All 160 long candidate
`indexing_backward_kernel` calls (49.356 s) disappeared; nine unrelated short
calls remain and total 0.430 ms. The replacement's 32 grouped reduction kernels
total 253.612 ms. Scalar loss is exactly identical at both profiled iterations.

The new external GPU sampler observed 15,067 MiB device-total peak versus an
earlier separately observed 10,627 MiB total. Because the observations were not
collected by one identical sampler, treat this as a material memory-risk signal,
not a precise allocation delta. It must be watched in the fresh health pilot;
no model or training setting is changed pre-emptively.

### DR-assisted glass-mask proposal gate

- [x] Strictly audit all 111 source RGB and real DR RGB/normal/depth/basecolor/
  diffuse_albedo artifacts, including paths, hashes, modes, dimensions, and
  source/stem ordering.
- [x] Record and exclude the final nine padded DR slots; never map them to
  TiHuBird views.
- [x] Keep DR normal axis semantics unconfirmed and DR depth within-frame
  relative; do not substitute the Stage B D G-buffer as primary geometry.
- [x] Generate automatic proposals for only the nine fixed representative
  stems using combined RGB, DR normal, DR depth, connectivity, and optional
  basecolor cues.
- [x] Export source-resolution proposal soft/hard-preview/overlay/uncertainty,
  cue boundaries, metadata, and a five-column contact sheet.
- [x] Keep proposals outside the accepted specular-mask contract. The training
  loader rejects their nested/non-complete layout, and no `reviewed_soft`,
  formal mask manifest, or `lambda_spec` run exists.
- [x] User reviews the nine-view contact sheet and authorizes the frozen method
  for a 111-view automatic-draft pass.
- [x] Generate and strictly validate 111/111 automatic proposals with zero
  padding proposals, per-frame hashes, dimensions, area/bbox/uncertainty/risk
  records, ten chronological and ten priority contact sheets, four boundary
  crops per frame, global distributions, and a non-destructive anomaly list.
- [x] Inspect every chronological contact sheet and find no systematic proposal
  failure requiring a method rollback.
- [x] Build the 30-view high-risk review package in background-risk,
  reflection-risk, then uncertainty order, with one large pack per view, six
  direct source-pixel crop types, one contact sheet, and blank human checklists.
- [x] Verify the complete 1,110-file proposal tree has identical aggregate
  SHA-256 before and after review-pack generation.
- [ ] Human fills the high-risk checklist; no automatic item is promoted to
  `reviewed_soft` by this packaging step.
- [x] Record initial review: 000039/000040/000041 fail locally; the other 27
  high-risk-pack views temporarily pass but are not formally accepted masks.
- [x] Generate three strictly subtractive top-boundary repair candidates using
  current-frame RGB/DR cues plus 000038/000042 continuity, without copying or
  interpolating neighbor masks and without regenerating the other 108 views.
- [x] Verify zero added hard pixels and identical source proposal-tree hashes;
  export source-resolution pages, direct boundary crops, continuity evidence,
  and the three-frame RGB/v1/repair/difference comparison.
- [x] Human confirms or redraws the partially occluded top boundary in all
  three repair candidates.
- [x] Humans revise/confirm all 111 source-resolution masks before any formal
  manifest or separately authorized `L_spec` validation.
- [x] Archive exactly 111 accepted mode-L uint8 masks with byte-copy source
  proof, complete RGB/mask hashes, aggregate/payload hashes, and zero padding
  contamination in `specular_masks_reviewed_v1/`.
- [x] Admit only the formal manifest at the training entry; reject proposal,
  repair, missing, extra, corrupt, wrong-mode, wrong-size, or unknown-source
  inputs fail closed.
- [x] Preserve soft edges with recorded `opencv.INTER_LINEAR` resizing and prove
  in tests that RGB reconstruction remains whole-image while L_spec ks gradient
  has support only inside the mask.

The strict audit passed with 111 real records at source 3827x2152 and native DR
704x384, plus padding slots 111--119 excluded. Normal maps are RGB uint8 decoded
and normalized for continuity only; no axis convention is assumed. Depth maps
are relative per-frame RGB visualizations and carry no COLMAP scale. The nine
review proposals are at
`output/stage_b_tihubird_dr_glass_proposal_9views_v2/`. They are annotation
inputs, not real masks or supervision, and Stage B remains unaccepted.

The accepted frozen method was expanded to all real views at
`output/stage_b_tihubird_dr_glass_proposal_111_v1/`. All 111 source-sized
single-channel uint8 proposals and retained review artifacts pass the strict
review manifest; padding slots 111--119 contribute zero proposals. The next
gate is human review/correction of every queue item, not `L_spec` or training.

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
reflection_contribution_vis.png  # display-only p99 scale; raw image preserved
microfacet_D_log.png             # display-only log1p/p99 scale
ray_candidate_count.png          # display-only p99 scale; raw stats in JSON
ray_exact_intersection_count.png # display-only p99 scale; raw stats in JSON
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
- [x] Real-mask loading/validation and `L_spec` masking tests before enabling
  the specular constraint.
- [x] Candidate/exact-intersection chunking, timing/memory, physical-map
  preservation, and companion-scale tests.
- [x] On-disk CUDA-map-location RNG restoration with CPU/CUDA state equality.
- [x] Existing Stage A regression suite plus DR proposal/review/repair,
  formal-mask/L_spec, bounded telemetry, and Tier-2 lifecycle contracts: full
  repository result is 117 passed; CPU-only is 102 passed / 15 skipped.

## Acceptance checklist

- [x] Independent Reflection Gaussian exists and passed synthetic tests.
- [x] Ray tracer runs on R and satisfies its tested differentiability contract.
- [x] Reflection-ray direction and scene-scaled epsilon are correct in tests.
- [x] Reflection color/alpha/depth/hit maps are real and visualized in the
  successful user smoke.
- [x] D and R optimizers, densification, checkpoints, and exports are separate.
- [x] Full microfacet BRDF runs without split-sum approximation in tests.
- [x] The accepted transparent-region mask raises ks through the documented
  specular constraint.
- [x] Reviewed smoke checkpoint tensors and tested gradients contain no NaN or Inf.
- [x] Stage A test suite and original code paths do not regress in the full
  84-test run.
- [x] The first successful Stage B smoke was run by the user and produced real
  checkpoint, D/R PLY, and debug evidence.
- [x] B-1d candidate density, exact intersections, timing/memory, and display
  companions are verified by the user-operated resume smoke.
- [x] The post-repair 100-step health pilot completed with finite state, stable
  counts/contribution, preserved performance, final checkpoint, and complete
  diagnostics; its reserved-memory risk is explicitly recorded.
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
8cd03e861f89487472144d5c9d4fcd7d62915e19  B-008 diagnostics decision
9d69a0d2a8ac402744ee638c64822b8d4e601da9  B-1d observability implementation
47bc8422a7c024ad7946a053f8a21e74d9af2851  B-009 RNG device contract
9156b1eed7ecab41b873ef796d27783289723f6d  CUDA checkpoint RNG restore fix
c87bb66934ddfcf86173c77dc9dcd724837ca8ae  grouped candidate-gradient reduction
0e673522c07b457c538385228acb90c513dac115  reviewed-mask archive and loader
e50eae723548ad9963b0bf236fa4da0d408e2da9  three-step L_spec smoke evidence
```

Implemented and tested does not mean accepted. The successful user smoke proved
the structural Stage B path, and B-1d diagnostics, the candidate-gradient
performance repair, and the 100-step health pilot have real evidence. The user
has since accepted the 111-view reviewed-v1 masks, and their formal manifest/
loader tests pass. Real-scene L_spec smoke and the single authorized 100-step
L_spec pilot have now run, but the pilot is evidence for human evaluation only
and carries an explicit whole-step memory telemetry gap. Stage B remains
unaccepted, and Stage C and Stage D remain forbidden.
