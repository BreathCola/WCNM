# Stage D — Transmittance Gaussian and Full RT-GS

## Entry contract

Stage C is accepted as immutable geometry release
`stage_c_geometry_release_v1`, manifest
`geometry_releases/stage_c_geometry_release_v1.json`, aggregate SHA-256
`4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d`.

Stage D must run the CPU-safe release validator before constructing T. It may
only read release-relative caches. It must never regenerate, refit, chmod, or
write the mesh/cache. Every Stage D checkpoint stores the release ID and
aggregate hash and refuses a mismatch on resume.

The geometry is a TiHuBird-specific data-constrained six-plane enclosure proxy,
not a general solution for curved or arbitrary transparent objects.

## Fixed branch domains

- Reflection rays remain unchanged on all valid Diffuse surfaces.
- T first-bounce and D second-bounce rays run only on `mask_hard &
  valid_two_hit` pixels from the frozen cache.
- `mask_eroded & valid_two_hit` is the L_depth domain.
- `mask_soft` remains the L_spec domain.
- RGB reconstruction remains full-frame.

## Implementation order

1. Add an independent Transmittance surfel model, optimizer, scheduler,
   densification state, PLY namespace, checkpoint namespace, and fresh seeded
   initialization.
2. Read frozen per-view `t_near`, `t_far`, validity, and `back_position`; never
   query or rebuild the mesh at training time.
3. First bounce: trace T from the D surface along camera transmission direction
   to produce `Cin/Ain/Din`.
4. Second bounce: trace D from frozen `back_position + eps*d_trans` to produce
   `Cout/Aout/Dout`.
5. Compose `Ct = Cin + (1-Ain)*Cout` and
   `At = Ain + (1-Ain)*Aout`.
6. Add `L_depth = mean(mask_eroded * valid * relu(Din-t_far))` under the explicit
   schedule/weight recorded in checkpoint metadata.
7. Compose full D/R/T output without changing Reflection behavior and export
   all branch debug maps.

## Minimum real-scene smoke gate

The authorized smoke starts from the frozen 3k-A/global-15,000 D/R checkpoint,
validates the unique geometry release, creates fresh T, and runs only the
minimum steps needed to prove one optimizer update plus save/restore. It must:

- finish without OOM, NaN, Inf, cache mismatch, or geometry mutation;
- save independent D/R/T state, optimizer/densification namespaces, PLYs,
  release identity/hash, telemetry, and a restorable checkpoint;
- show finite first/second bounce and exact alpha-over Ct;
- report `Din <= t_far`, valid-two-hit coverage, D/R/T counts, losses, step
  time, and allocator current/scoped peaks;
- output final/diffuse/reflection/transmittance contributions, inside/outside
  color-alpha-depth, transmittance color-alpha, depth violation, and frozen
  near/far/valid maps.

The smoke was not itself a long training authorization and does not enter Stage
E. The later formal-onset authorization is recorded below.

## Acceptance boundary

Stage D acceptance still requires longer evidence that Cin represents inside
content, Cout represents exterior content, T does not absorb R/background,
L_depth constrains Din, reconstruction remains stable, and novel views do not
flicker. A passing minimum smoke only returns
`STAGE_D_SMOKE_PASSED_AWAITING_REVIEW`.

## Minimum smoke evidence

`output/stage_d_tihubird_c03r8_smoke_v3/` completed two fresh steps and one
actual restored step. Final checkpoint is global 15,003 / R-local 12,003 /
T-local 3 with SHA-256
`f71d4644fcb2873ddc9d0ea058c87ce698c405d2f1b43e83b7fdd9df4349e560`.

- D/R/T counts: 258,593 / 3,088 / 4,096;
- three independent optimizer/state namespaces updated finitely;
- all three telemetry rows have zero non-finite count;
- scoped peak allocated/reserved: 3.48/4.40 GB;
- fixed-view `Din <= t_far`: 0.63439;
- geometry release aggregate matches before/after;
- all required debug files, three PLYs, telemetry, checkpoints, and CPU-only
  final audit are present.

Verdict: `STAGE_D_SMOKE_PASSED_AWAITING_REVIEW`. Inside color remains faint and
the combined glass region is over-bright after only three updates. This is an
engineering-path pass, not Stage D semantic/quality acceptance, long-training
authorization, or Stage E entry.

## Authorized formal T-onset trajectory

The user has now authorized exactly one new formal 5,000-step trajectory. It
must start from the frozen Branch-A Stage-B global-15,000 checkpoint (SHA-256
`050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84`),
not from any Stage D smoke. It uses `stage_c_geometry_release_v1` unchanged,
creates fresh T as `random_bbox`, count 4,096, seed 20260703, and updates global
15,001 through 20,000 only. The initial v1 at
`output/stage_d_tihubird_c03r8_formal_onset_g15000_g20000_v1` was stopped by
the user after global 15,018 because the safe 512-ray path projected 8.8 hours.
It has no checkpoint and is excluded from resume. The authorized restart is a
fresh trajectory at
`output/stage_d_tihubird_c03r8_formal_onset_g15000_g20000_v2`.

V2 uses checkpointed 2,048-ray training chunks with same-camera pre-commit OOM
fallbacks at 1,024 and 512. It releases allocator cache only below 2 GiB device
free. Debug remains at 512. Candidate/intersection definitions and all training
math remain unchanged.

This formal run keeps `L_depth` disabled throughout. Metadata must record
global 40,000 as its future activation with `lambda_depth=0.2`. Checkpoint, PLY,
telemetry, nine-view debug, and contact-sheet nodes are exactly 15,100, 15,500,
16,000, 17,500, and 20,000. A CPU-only final audit must stop at
`HOLD_FOR_SEMANTIC_REVIEW` or `BLOCKED`; it cannot enter Stage E or continue
beyond global 20,000.

## Authorized cached T warm-up then exact joint trajectory

The user stopped the uncompleted all-joint v2 path and replaced it with one new
trajectory at
`output/stage_d_tihubird_c03r8_cached_twarmup_then_joint_g15000_g20000_v1`.
It still starts exclusively from the original Stage-B global-15,000 checkpoint
and uses the immutable Stage-C v1 release and fresh-T initialization above.

- Phase A, global 15,001--18,000: D/R and all D/R training state are frozen;
  R-local remains 12,000. A fail-closed per-view FP32 cache replaces only
  D/R/geometry computations independent of T. Full-frame RGB and exact T
  gradients remain active.
- Phase B, global 18,001--20,000: the cache is disabled and the complete exact
  D/R/T renderer and joint backward resume. R-local reaches 14,000 and T-local
  reaches 5,000.
- Cached/uncached parity is required on nine fixed plus one deterministic
  random view for outputs, losses, and all T parameter gradients. D/R
  full-state hashes must match before and after Phase A.
- Complete checkpoint/PLY/nine-view nodes are 15,025, 15,100, 15,500, 16,000,
  17,500, 18,000, 19,000, and 20,000.
- `L_depth` is disabled for both phases; global 40,000 with
  `lambda_depth=0.2` remains only a future schedule marker.

The operator may return only `CACHED_T_WARMUP_AND_JOINT_ONSET_PASS`,
`HOLD_FOR_SEMANTIC_REVIEW`, or `CACHED_T_WARMUP_BLOCKED`. It must not infer
bird/background separation from falling loss and must not continue into Stage E.

### Cached trajectory v1 failure and v2 retry

The user-launched v1 completed 25 Phase-A updates but failed inside debug export
at the first 15,025 node because a CHW mask was expanded against HWC RGB. Its
cache/parity/training evidence remains preserved, but its checkpoint precedes
an incomplete review/telemetry commit and is not resumable. The debug-only mask
layout fix retries from the original Stage-B source with fresh T at the unique
v2 output. V2 keeps every Phase A/B schedule, cache identity, parity tolerance,
node, renderer, loss, and audit requirement above unchanged.

## Authorized semantic-repair v3 falsification pilot

The completed cached-v2 engineering path is a semantic failure and is now
read-only evidence. It must never be resumed. The only authorized next run is
fresh from the original Stage-B global-15,000 source and immutable Stage-C v1
release at `output/stage_d_tihubird_c03r8_semantic_repair_v3`.

- Run only global 15,001--16,000. D/R parameters, optimizers, schedulers,
  densification bookkeeping, topology, and R-local iteration are frozen. Use
  the fail-closed FP32 static cache and full-frame RGB; do not enter an exact
  joint phase.
- Use shared `rtgs_cuboid_space_v1` classification with margin 0.05 and
  `transparent_interface_margin_mode=exclude`. Interface is independently
  counted and may not be folded into another class.
- Within the transparent mask, formal R and formal second-bounce Cout accept
  only current-position outside-class surfels. Outside-mask R remains the
  original global path. Export unfiltered/inside/interface/outside/final maps
  and surfel/candidate/hit/energy statistics.
- Fresh T uses `cuboid_inside_sigmoid_v1`, count 4,096, seed 20260703. Pruning
  and densification are disabled; every step must assert all T surfels remain
  strictly inside-safe and count remains 4,096.
- Apply the D-008 anti-veil prior with defaults and T-local 0--200 smoothstep
  ramp recorded in checkpoint/run/cache metadata. It is an engineering prior,
  not an RT-GS paper claim. It may not copy GT as T supervision or leak
  gradients into frozen D/R.
- Before optimization, compare uncached frozen-D/R and cached-T paths on nine
  fixed plus one deterministic random view for final/Cin/Ain/Din/Cout/Aout/
  Dout/Ct/At, every loss, and all T parameter gradients. Max/mean absolute
  tolerances are `2e-5/2e-6`; any mismatch blocks training.
- Save full checkpoint, PLY, nine-view maps, and semantic contact sheet at
  15,025, 15,100, 15,500, and 16,000. CPU audit checks source/release/cache
  identity, recursive finite state, frozen D/R hashes, telemetry continuity,
  fixed T topology/space, outside-only formal composition, hard metrics, and
  all required files.

Verdict is only `SEMANTIC_REPAIR_PILOT_PASS`,
`SEMANTIC_REPAIR_PILOT_HOLD`, or `SEMANTIC_REPAIR_PILOT_BLOCKED`. With no
independent versioned bird ROI, a technically valid run remains at least HOLD
for human nine-view review. The operator cannot run global 16,001, joint
fine-tuning, Stage E, or infer semantic success from lower loss.

Outside-only center classification does not prove semantic content: outside R
or D surfels can still memorize colors resembling internal content. Nine-view
structure review remains mandatory and the filters alone are never acceptance
evidence.
