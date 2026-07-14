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

## Authorized cuboid-front ownership v4 A/B pilot

The user-executed v3 is blocked evidence, not a resume source. Despite fixed
4,096 inside-center T and center-filtered R/Cout, it retained the D-derived
transparent interface path and ended with Ain saturation/black-veil collapse.
The next and only authorized execution is the fresh v4 A/B pilot at
`output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4`.

For `mask_hard & valid_two_hit`, `cuboid_front_v1` computes
`front_position=camera_center+t_near*d`, where `d` is the normalized frozen
camera-to-back-position direction. The normal is the selected cuboid entering
face normal, transformed from cuboid-local axes and face-forwarded so
`dot(n,-d)>0`. R origin/d_cam/wo/reflection direction/Fresnel normal and T
origin/direction use that contract; T absolute depth is
`t_near+eps+relative_depth`. Cout retains the frozen back-position origin.
Missing/non-finite fields, invalid near/far ordering, off-plane front points,
or a missing required material surface fail closed. Pixels outside the hard
transparent mask retain the existing D/R formulas and paths exactly.

Ownership is based on complete cuboid-local finite 3-sigma surfel support, not
centers. Every D/R/T support is classified as strict-inside-safe, interface,
strict-outside-safe, or crossing/ambiguous, with separate surfel, candidate,
hit, alpha-energy, and map outputs. The pilot requires:

- transparent D direct contribution `off`;
- transparent R contribution `off` (while retaining tested
  `support_safe_outside` as a future configurable mode);
- Cout from strict-outside-safe D support only;
- exactly 4,096 strict-inside-safe T surfels with no pruning/densification;
- full-frame RGB loss and L_depth disabled throughout;
- D/R parameters, optimizer/scheduler/topology state, and local iteration
  frozen for all 500 updates in each arm.

These path/support/ownership controls are versioned renderer and cache
contracts intended to remain available for later joint fine-tuning. The strong
D/R-off setting isolates ownership and is not asserted to be the final physical
rendering policy. Changing or disabling a gate requires a new decision and
evidence; it must not happen automatically at a warm-up boundary.

Arm A is `random_strict_inside`; Arm B is `transferred_d_inside`. The latter
copies only D surfels with strict-inside-safe 3-sigma support and multi-view
attributed hits whose absolute depths lie safely between frozen near/far. It
copies xyz, rotation, raw 2D scale, and raw base color; opacity is scaled by
0.25 and clamped to 0.005--0.05; material fields and all training state are
fresh. Deterministic support-safe random fill reaches 4,096. This is a matched
initialization ablation, not a semantic label or proof that D content is true.

Both arms start from the same immutable Stage-B global-15,000 checkpoint and
Stage-C release, share the same random camera sequence and all training/loss
settings, and stop at global 15,500. Required nodes are 15,000, 15,100, 15,250,
and 15,500 for the fixed nine views. A new v4 static-cache schema binds the
cuboid-front and ownership identities and rejects every legacy D-origin cache.

The operator performs preflight, immutable hashes, fresh cache/parity, Arm A,
Arm B using the same validated cache, and a CPU-only audit. Verdicts are only
`CUBOID_PATH_OWNERSHIP_PILOT_PASS`, `CUBOID_PATH_OWNERSHIP_PILOT_HOLD`, or
`CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED`. No independent human-authored bird ROI
currently exists, so RGB improvement cannot be called semantic success and the
run must HOLD or BLOCKED. Even PASS means only that T has demonstrated handoff
conditions under strong ownership isolation; it cannot authorize joint
fine-tuning, final separation claims, or Stage E.

### V4 zero-step launch correction

The first v4 invocation from commit `b697c25` failed before cache construction
and before any optimizer update because the ownership flag did not activate
release-cuboid construction. This is a launch-gate defect, not pilot evidence.
The corrected implementation requires the cuboid for both semantic-repair and
ownership modes and validates its schema before the run contract proceeds.

The next operator invocation preserves the exact known zero-step directory by
atomically renaming it with suffix `_failed_preflight_b697c25`, then recreates
the canonical v4 output. This exception is allowed only when source/release
hashes match and cache, checkpoint, and telemetry are all absent. No partial
run may be resumed or overwritten; every other pre-existing output remains a
hard refusal. All v4 renderer, A/B, schedule, and verdict requirements above
remain unchanged.

### V4 transferred-scale correction

The `17d2663` retry completed cache/parity and Arm A, but Arm B stopped after
global 15,114 when an optimizer update made raw T scale too large for a feasible
strict-inside 3-sigma support. The strict failure is preserved evidence; it is
not resumed. The corrected T representation retains raw log-scale storage and
exact feasible D raw-scale transfer, while active scale uses a current-rotation
cuboid-capacity factor shared by both tangent axes. The factor is one for every
feasible support and only contracts a would-be infeasible support, so anisotropy
is retained and raw optimizer state is not projected.

Checkpoint, telemetry, debug, and audit identify this rule as
`cuboid_support_uniform_cap_v1` and report its activation. The existing v4
strict-support and fixed-count requirements remain mandatory. For matched-code
integrity, the next retry archives the entire `17d2663` output with suffix
`_failed_scale_17d2663`, rebuilds the cache, and reruns both arms fresh from
global 15,000. Reusing completed Arm A or resuming partial Arm B is forbidden.

## Authorized ownership transferred-T long continuation

The completed `7c37088` A/B is technically healthy but remains HOLD because no
independent bird ROI exists. It is sufficient to select transferred Arm B for
one bounded T-only continuation: fixed-nine transparent L1 and intervention
evidence are consistently better than random Arm A, T-off worsens every view,
and black-veil coverage remains approximately zero. It is not sufficient to
claim bird semantics, restore D/R, or enter joint training.

The only source is Arm-B global-15,500 checkpoint SHA-256
`eff2135e19b48dd68f3661cbf23825d79deb0beaf29338887c0d5e8146127321`.
Run global 15,501--20,000 / T-local 501--5,000 only. D/R and all their state
remain frozen at R-local 12,000; transparent D direct and R stay off; Cout stays
strict-outside-safe; T remains exactly 4,096 strict-inside-safe supports with no
topology changes; full-frame RGB remains active; L_depth remains disabled.

The runtime uses a 100-step rolling guard from T-local 1,000: saturation <0.50,
high-Ain/near-black <0.10, capped-support fraction <0.10, and minimum scale-cap
factor >0.02. Required global nodes are 15,500, 16,000, 17,500, and 20,000 with
checkpoint, three PLY namespaces, fixed-nine debug, raw statistics, ownership
interventions, and CPU audit. Any contract/hash/cache/frozen-state/finite/OOM/
support/count/guard failure blocks.

The unique output is
`output/stage_d_tihubird_c03r8_cuboid_path_ownership_tlong_g15500_g20000_v1`.
Verdict is only `OWNERSHIP_T_LONG_HOLD` or `OWNERSHIP_T_LONG_BLOCKED`. HOLD does
not authorize joint fine-tuning, final glass physics, Stage D acceptance, or
Stage E.

## Authorized global-16,000 projected-scale recovery

The v1 long run is preserved blocked evidence. Its only recovery source is the
complete global-16,000 / R-local-12,000 / T-local-1,000 checkpoint with SHA-256
`52d1368dfb2a729240265e27f7696b0d230522af3fae795049c70932a673158e`.
It may be migrated to `cuboid_support_projected_cap_v2`: raw T scale is replaced
by the numerically identical active scale for capped rows, and only those rows'
scaling Adam moments are cleared. Fixed-input active geometry, raytrace/Ct, and
loss parity plus save/load legality are mandatory.

The implementation agent is authorized to run exactly 50 real steps,
16,001--16,050, in a new output. D/R remain fully frozen, T remains 4,096 with
no topology updates, cuboid-front and all ownership gates remain unchanged,
and L_depth remains off. Required complete nodes are 16,000, 16,001, 16,010,
16,025, and 16,050. Verdict is only
`TSCALE_RECOVERY_PREFLIGHT_PASS` or `TSCALE_RECOVERY_PREFLIGHT_BLOCKED`.

PASS authorizes only a user-operated T-only continuation from the new 16,050
checkpoint to 20,000. That continuation must keep the projected-scale repair,
write checkpoint/PLY/fixed-nine/hash audits at least every 250 steps, stop on
any guard, and never enter joint tuning or Stage E.

The first v1 recovery implementation stopped at its zero-update real parity
gate: tiny log/exp geometry round-off changed finite-support candidates and was
not acceptable as a lossless migration. V1 is preserved BLOCKED evidence. The
v2 retry checkpoints the exact active scale as non-optimizer state with a
straight-through bounded derivative, retains raw log-scale projection and
scale-only Adam surgery. The real v2 zero-update retry proved active geometry
was exactly equal but rendering still changed because LBVH AABBs consumed
active scale while exact candidate intersection decoded raw `_scaling`. V2 is
preserved BLOCKED evidence. V3 makes T candidate intersection consume the same
active scale used by its AABB and forward contract; R remains unchanged. The
fresh v3 output is required before any continuation.

The v3 recovery preflight subsequently passed. Migration parity is exactly
zero for active scale, decoded position, all audited outputs, and all losses.
All required 16,000/16,001/16,010/16,025/16,050 checkpoints and audits exist;
D/R hashes are unchanged; T remains 4,096 and entirely strict-inside-safe. At
16,050, raw/active maximum scale difference is `3.07e-8`, the minimum factor is
`0.999999875`, and the fixed-nine transparent L1 improved from `0.126823` to
`0.118399` without Cin or T-contribution collapse. Only the guarded user-run
16,050--20,000 T-only continuation is now authorized.

## Authorized D-015 zero-update semantic-renderer ablation

The global-20,000 continuation is retained as numerical scale-repair success
but visual/semantic HOLD. It is a read-only render source and is permanently
forbidden as a training resume point. D-015 runs fixed-nine, zero-update Arms
0--3 for both that complete D/R/T state and the original Branch-A global-15,000
D/R source with one captured fresh 4,096-T initialization reused across arms.
No new checkpoint may be written.

Arm 0 reproduces D-off/R-off/strict-outside Cout. Arm 1 restores R only for
`mask_hard & valid_two_hit` using cuboid-front geometry and complete-support
strict-outside-safe R; invalid pixels remain Arm 0. Arm 2 replaces only
hard-mask invalid pixels with an exact invocation/reuse of the original Stage-B
legacy D/R compositor and labels every fallback pixel. Arm 3 retains the strict
Cout result and separately admits interface/crossing D candidates only when
their individual exact hit lies beyond the frozen back face. Positive cuboid
clearance is inside, zero is on-plane, and negative is outside; accepted hits
require both positive outward exit-plane distance and clearance below the
negative tolerance. Strict-inside candidates are always rejected.

All formal metrics are computed from float tensors. Required products include
raw/conditional/alpha/weight D, R, Cin, and Cout; formal/unfiltered R energy;
final luminance; multi-label black-hole attribution; fallback maps; strict and
handoff Cout maps/differences; accepted/rejected handoff diagnostics; Arm
contact sheets; outside-mask bitwise parity; source/release/cache immutability;
and zero-update proof. PNG is visual evidence only. Transparent D direct stays
off and no Arm 4 is implemented. Verdict is only `AWAITING_USER_REVIEW`,
`HOLD`, or `BLOCKED`; no training or Stage E action follows automatically.

Implementation note: D-015 uses a dedicated zero-update operator rather than
the ordinary Stage D training loop. It may construct models for read-only
forward rendering, but it must not call backward, optimizer/scheduler steps, or
write checkpoint/PLY/resume artifacts.

## Authorized D-016 Grounded-SAM2 internal-object T ownership

D-016 replaces further D-015 Arm debugging. D-015 outputs, code, and logs are
preserved as read-only evidence; they are not resume sources and are not the
training source for this task.

The goal is to create `bird`, `internal_base`, and
`internal_object_union = bird | internal_base` masks for the 111 TiHuBird source
images using the local Grounded-SAM2 environment. V3 retires `bird_support` and
does not allow `support_mount` as a final semantic role. `internal_base`
contains the yellow rectangular base board inside the glass, the white platform
under the bird, and only reviewed fixtures connected to those base structures;
it excludes glass, outside ground, independent rails/poles, labels,
reflections, and bird. These masks separate two responsibilities:

- the reviewed glass mask continues to define where T first bounce and Cout
  second bounce run: `mask_hard & valid_two_hit`;
- the reviewed internal-object mask defines where T is encouraged to form bird
  and support occupancy inside the glass.

Grounded-SAM2 outputs first enter a proposal directory only. Proposal outputs
must include raw masks, glass-clipped processed masks, overlays, contact sheets,
per-view metadata, `manifest.json`, `validation.json`, and `review_queue.csv`.
They are review evidence, not formal supervision. A separate promotion tool may
create `data/TiHuBird/internal_object_masks_reviewed_v3/` only after explicit
user approval of all 111 stems. Stage D loaders must reject proposal directories
and loose PNG sets.

The fixed-nine D-016 v3 proposal probe must not OR all prompts, boxes, and SAM
masks together. It must emit independent candidates for `bird`,
`yellow_base_board`, `white_platform`, and optional `connected_fixture`.
`connected_fixture` is not a final role and may be empty. The generator must
reject candidates that fill glass, leak outside glass, touch too many glass
boundaries, fragment the bird, or resemble independent side poles/rails.
`internal_base` is formed only from selected yellow-board, white-platform, and
reviewed connected-fixture helper candidates.

The reviewed-anchor 111-view proposal package is still proposal evidence only.
It records explicit user acceptance of fixed-nine v3 anchors and uses those
accepted masks as SAM2 video anchors with object IDs `1 = bird`,
`2 = yellow_base_board`, and `3 = white_platform`. Each anchor interval is
propagated both forward and backward, the two candidate sets are saved
separately, and the selected helper masks must not be an unconditional
forward/backward union. Anchor masks must remain byte-identical to the accepted
fixed-nine masks. The package must include raw selected helper masks,
glass-clipped processed `bird`, `internal_base`, `internal_object_union`, review
CSV/JSON, contact sheets, review pages, temporal metrics, and preview-only
`Mpos/Mignore/Mneg` domains. Review guards may flag low temporal IoU,
forward/backward disagreement, raw outside-glass leakage, centroid or area
jumps, and distant significant components. They may not promote masks.

The current reviewed-anchor artifact is
`output/stage_d_tihubird_internal_object_mask_proposal_111_v3/`. It contains
111/111 processed masks for the required roles, 9 accepted anchors, 69
auto-candidate-ready frames, 33 review-required frames, and 0 manual-edit
required frames. This status does not create formal reviewed masks and cannot be
used by training until a separate all-111 human-reviewed manifest is promoted.

The user has now explicitly accepted all 111 frames from that proposal. D-016c
promoted the reviewed masks to the local formal release
`data/TiHuBird/internal_object_masks_reviewed_v3/` with role
`stage_d_internal_object_masks_reviewed`, `human_status = accepted`, semantic
version `tihubird_bird_and_internal_base_v3`, and stems 000000--000110. The
formal `bird`, `internal_base`, and `internal_object_union` PNGs are byte copies
of the proposal `processed/` PNGs; `glass_hard/` is included as a byte-copied
validation domain; `internal_ignore/` is empty. Accepted-with-warning frames are
000012, 000048, 000049, 000050, 000051, 000052, and 000053.

Formal release identity:

- aggregate mask SHA-256:
  `c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052`;
- manifest file SHA-256:
  `ae5be91083c9859a81917c383358b030bc3621cd3d8ac52faea8a5becf40350e`;
- manifest canonical payload SHA-256:
  `34e244f45758d2e6fac010b3510a63b418c5a006678c84478b3dae9701491592`;
- human review record SHA-256:
  `a1980f4604da3d7b935e149c8b55ac53be87a0361a1b4528325e4130e9d304ca`;
- source proposal processed payload SHA-256 before/after promotion:
  `521b6b9e3f1010342edee3e3b2e9e377fa46b270753f9196ced090581c52f8ae`.

The formal loader must continue to reject the proposal directory, fixed-nine
proposal artifacts, loose PNGs, v1/v2 manifests, non-accepted status, missing or
extra stems, hash mismatches, `union != bird | internal_base`, and
`union` outside `glass_hard`. This promotion does not authorize or run the
D-016 pilot, optimizer updates, checkpoints, PLY export, Stage C changes, joint
training, Stage D acceptance, or Stage E. `data/` remains Git-ignored, so the
formal release is local immutable data rather than a Git-tracked asset.

The fixed preferred prompt roles are:

```text
Bird: the physical taxidermy bird specimen inside the glass display case
Internal base: the yellow rectangular display board and white platform inside the glass case
```

The fixed fallback prompt lists are recorded in the proposal manifest. Prompt
changes may not be tuned from RT-GS render results.

The D-016 training integration is default-off. When explicitly enabled with a
formal reviewed manifest, transferred-D initialization adds semantic eligibility:
strict support-safe, current depth/visibility gates, enough valid object-mask
projections, sufficient support ratio, and low boundary-band ownership. Failed
semantic quota is filled only by the existing deterministic strict-inside random
fill; thresholds are not relaxed to reach 4,096. The first object loss supervises
only `Ain`:

```text
Mpos = erode(Mobj) & mask_hard & valid_two_hit
Mignore = (dilate(Mobj) - erode(Mobj) | reviewed internal_ignore) & mask_hard & valid_two_hit
Mneg = outside(dilate(Mobj)) & mask_hard & valid_two_hit & !reviewed_internal_ignore
L_object = lambda_pos * mean(Mpos * relu(alpha_floor - Ain))
         + lambda_neg * mean(Mneg * Ain)
```

`Mignore` has no alpha penalty and prevents uncertain internal details,
boundary errors, labels, highlights, or optional reviewed ignore regions from
becoming forced negatives. This loss must not use target RGB as `Cin`
supervision, crop final RGB, change the T/Cout ray domain, disable Cout, or
allow gradients into frozen D/R. All metrics are computed from float tensors;
PNGs are visual evidence only.

The bounded D-016 pilot is an independent Stage D operator. It defaults to a
plan-only dry run and requires `--execute` to launch training. It starts fresh
from Branch-A global-15,000, reads Stage-C v1 immutably, freezes D/R and all D/R
training state, updates only T for the bounded schedule, keeps fixed T count and
support safety, keeps full-frame RGB, keeps Cout, refuses existing output,
refuses proposal masks, and stops with review/audit evidence only. It does not
authorize joint tuning, global 20,000 continuation, Stage D acceptance, or Stage
E.

### D-016d plan-only and audit infrastructure

The D-016 pilot operator's default plan-only path has been repaired to use the
actual dict returned by `validate_geometry_release()`. It validates the source
checkpoint, immutable Stage-C release, formal reviewed glass masks, and formal
reviewed internal-object masks, refuses an existing pilot output, then prints a
JSON plan and returns without calling `train.py` unless `--execute` is present.
The plan records the exact source/release/mask identities, 500-update
15,001--15,500 bounded schedule, required nodes, T count 4,096, seed 20260703,
D/R frozen status, object-loss parameters, ray-domain statement, planned
products, and future command. The intended output remains
`output/stage_d_tihubird_c03r8_internal_object_townership_pilot_v1` and must
not exist before execution.

The D-016 final CPU audit is now
`tools/audit_stage_d_internal_object_townership.py`. It is read-only with
respect to checkpoint/PLY/debug artifacts and fail-closed: it checks operator
plan identity, Stage-C identity, formal internal-object release identity,
metadata, exact telemetry range 15,001--15,500, finite losses/state, T-only
optimizer updates, unchanged D/R parameter and optimizer hashes, finite nonzero
T changes, fixed T count, no densification/pruning/topology changes, required
checkpoint/PLY/debug nodes, and split internal-object metrics. It writes only
its audit JSON and Markdown summary into a completed pilot output when invoked.
Allowed verdicts are `D016_PILOT_PASS_AWAITING_USER_REVIEW`,
`D016_PILOT_HOLD`, and `D016_PILOT_BLOCKED`; none authorize Stage D acceptance,
additional training, or Stage E.

D-016 telemetry now includes float-tensor split metrics for `bird`,
`internal_base`, `union`, and `Mneg`: pixel counts; Ain mean, median, p05, p50,
p95, max, nonzero ratio, and above-alpha-floor ratio; Cin RGB/energy/nonzero
statistics; Mneg Cout energy; and union-outside Cout retention. These metrics
are diagnostics only. The loss still uses the union domain and does not double
count bird/base overlap.

The implementation and tests for this infrastructure were added without running
`--execute`, without starting training, without optimizer updates, without
creating the pilot output, without checkpoint or PLY creation, without modifying
Stage C, and without entering Stage E. The real pilot remains a user-launched
action.

### D-016d launch-gate correction

The first user-launched wrapper for the D-016 pilot reached `train.py` and then
stopped at `_validate_args()` before any optimizer update because
`transferred_d_inside` was not included in the allowed T initialization set for
`stage_d_internal_object_pilot`. The execute log records the failure, but the
formal output directory was not created and there are no pilot checkpoints,
PLYs, telemetry, or CPU audit products.

The launch gate now includes `stage_d_internal_object_pilot` in the
`transferred_d_inside` allowance while preserving the D-016-specific
requirements for formal internal-object masks, `cuboid_front_v1`, and
`support_safe_outside`. This is a launch validation repair only; it does not
change the 500-update schedule, source checkpoint, Stage-C release, mask
release, renderer, losses, ray domain, or authorization boundary. The fixed
pilot still requires a separate user launch.

### D-016e posthoc review materialization

The completed user-run D-016 pilot produced real training evidence only for the
bounded 15,001--15,500 interval. Therefore 15,000 must not be represented as a
real pilot checkpoint. It is now a posthoc deterministic replay node:

- root pilot checkpoints and PLYs are required only at 15,100 / 15,250 /
  15,500;
- 15,000 review evidence must live under `posthoc_review/` and must be labeled
  `deterministic_zero_update_replay`;
- the replay must use the same source checkpoint, Stage-C release, formal
  reviewed internal-object masks, `transferred_d_inside` seed/config, and exact
  transferred-D internal-object filter recorded by the run;
- no optimizer update, scheduler advance, densification, pruning, checkpoint
  resume, or training step may occur during materialization.

The CPU audit reads the filter only from
`actual_transmittance_initialization.selection.internal_object_filter`. This is
the runtime metadata path written by the D-016 training code. The older direct
path under `actual_transmittance_initialization` is rejected.

`tools/materialize_stage_d_internal_object_review.py` is the post-run tool for
the separate user step. It refuses existing `posthoc_review/`, verifies the
real 15,100 / 15,250 / 15,500 checkpoints before heavy work, replays the
initial state twice to prove determinism, renders fixed-nine review artifacts
under `posthoc_review/debug/`, emits derived PLY/replay metadata under
`posthoc_review/`, and records immutable before/after hashes for the original
pilot files, operator plan, source checkpoint, Stage-C manifest, and formal
internal-object manifest.

This D-016e infrastructure change does not rerun the pilot, does not run the
materializer on the real output, does not run the final posthoc CPU audit, does
not alter Stage C or formal masks, and does not authorize Stage E.
