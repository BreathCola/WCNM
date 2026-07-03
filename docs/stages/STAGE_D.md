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

The smoke is not a long training authorization and does not enter Stage E.

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
