# RT-GS Status

Current stage: Stage D — Transmittance Gaussian and Full RT-GS

Current implementation branch: `feature/stage-d-transmittance`

The current authorized task is D-015, a zero-update semantic-renderer repair
ablation only. The completed global-20,000 ownership result is preserved as
scale-repair success but visual/semantic HOLD; it is a read-only reproduction
source and may never be resumed for training. A second group starts from the
immutable Branch-A global-15,000 D/R source with one deterministic fresh T
snapshot reused across Arms 0--3. The ablation tests cuboid-front safe R,
exact legacy D/R fallback only on two-hit-invalid hard-mask pixels, and an
experimental exact-hit back-face Cout handoff. Transparent D direct remains
off. Final metrics must come from float tensors. No optimizer update,
checkpoint output, short pilot, joint training, Stage E action, or separation
claim is authorized. The unique planned output is
`output/stage_d_tihubird_c03r8_semantic_renderer_repair_zero_step_ablation_v1`.

The user-executed ownership T-long v1 is now preserved
`OWNERSHIP_T_LONG_BLOCKED` evidence. It completed through global 16,774 and
stopped on the intended minimum scale-factor guard: active scale and all 4,096
supports remained legal, but raw maximum scale escaped to 18.47 while the
minimum active/raw factor fell to 0.019987. Its global-16,000 checkpoint is the
last formal recovery point and has SHA-256
`52d1368dfb2a729240265e27f7696b0d230522af3fae795049c70932a673158e`.

The current task implements `cuboid_support_projected_cap_v2`. A CPU audit of
that exact checkpoint projects 24 affected raw scales to their active values
with maximum active-scale/world-position differences `2.98e-8/8.94e-8`, leaves
all 4,096 complete supports strict-inside, and changes only the scaling
parameter plus affected scaling-Adam rows. CPU and synthetic CUDA tests cover
forward/raytrace parity, optimizer isolation, support legality, and checkpoint
round-trip. The complete GPU-enabled repository suite passes 192/192. One
same-tick static-cache `touch` check failed once, then passed its isolated rerun
and the complete rerun; this is recorded as a timestamp test flake, not a cache
or scale-repair failure. After commit, the implementation agent is authorized
to run only the new global 16,001--16,050 real recovery preflight. No 16,051+ training,
joint tuning, or Stage E action is authorized until that audit passes and the
user manually launches the returned continuation command.

The first committed recovery implementation (`2c4db50`) was correctly blocked
before any optimizer update by the real fixed-view migration parity gate. A
float32 `log(active)` round trip changed scale by at most `5.96e-8`, but finite
support candidate boundaries amplified that into final-RGB mean/max differences
`0.00943/0.70810`. Its v1 preflight directory is preserved. The corrected v2
stores the exact active scale as checkpointed non-optimizer state and uses a
straight-through bounded derivative, while retaining projected raw log-scale
and scale-only Adam surgery. The v2 retry also stopped before any update:
active scale and decoded world position were exactly equal, yet final RGB had
the same large difference. The root cause was a split raytracer contract: LBVH
bounds used `get_scaling`, while exact candidate intersection hard-coded
`exp(_scaling)`. V3 makes T candidate intersection encode the actual active
forward scale; R retains its original path. Both blocked attempts remain
preserved, and the fresh retry is uniquely suffixed `_v3`.

The committed v3 repair (`62280ac`) passed its real recovery gate. The exact
global-16,000 source migrated with zero fixed-view difference in active scale,
decoded position, every audited raytrace output, and every audited loss. The
50 authorized updates completed through global 16,050 with 4,096/4,096 T
supports strict-inside-safe, zero capped supports at the final checkpoint,
raw/active maximum difference `3.07e-8`, and unchanged D/R hashes. Fixed-nine
means from 16,000 to 16,050 changed as follows: Ain `0.25543 -> 0.29633`, Cin
energy `0.11896 -> 0.13450`, T contribution `0.28242 -> 0.28690`, and
transparent L1 `0.12682 -> 0.11840`. The CPU verdict is
`TSCALE_RECOVERY_PREFLIGHT_PASS`. This authorizes only the user-operated v3
T-only continuation from global 16,050; it is not Stage D acceptance and does
not authorize joint tuning or Stage E.

Stage B was formally accepted and closed by explicit user authorization on
2026-07-03. The matched `C03-r8 Tier 2 onset study` v4 run completed both
branches at global 15,000: Branch A started Reflection at global 3,000 and
finished at R-local 12,000, while Branch B started at global 7,000 and finished
at R-local 8,000. Both final checkpoints, telemetry streams, logs, point clouds,
and debug nodes are complete and recursively finite. The regenerated CPU-only
final audit is healthy at
`output/tier2_c03_r8_oneshot_v4_final_audit.json`. Its earlier false
`PARTIALLY_COMPLETED` classification came only from parsing valid structured
retry JSON together with the logger's trailing timestamp; the parser now
accepts only the documented timestamp suffix and remains fail-closed for any
other trailing text.

Branch A global 15,000 is the selected Stage C geometry source. Its checkpoint
SHA-256 is
`050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84`.
This choice is specific to the matched resolution-8 study: A had lower matched
loss/L_spec, slightly better fixed-view and sampled-train reconstruction, and
no memory retries. Branch B is retained as the complete 7k onset control. The
result does not prove that 3k is globally optimal, that Reflection separation
is complete, or that mask-exterior Reflection/ks is causally correct. There is
no Reflection ground truth and the Stage B fixed debug view is not a complete
cross-view quality evaluation.

Stage C's D-only extractability audit passed and the implementation now produces
a fixed watertight mesh plus 111 strict version/hash-validated mask-hard two-hit
caches. The accepted source, audit, mesh, and cache details are recorded in
`docs/stages/STAGE_C.md`. The retained v3 mesh has zero boundary and non-manifold
edges. All valid cached rays satisfy `t_far > t_near`.

The first pure-TSDF v3 mesh remains a diagnostic failure: mean hard-mask
two-hit coverage was 0.80313 and 0.44044 of valid rays had more than two mesh
crossings. It has not been overwritten. The independent DR-guided repair uses
C03 normals for orthogonal axes, the D/TSDF result only for metric initialization,
the existing raw DR depth after per-view opaque-region metric calibration as an
independent front-depth check, and all 111 source-resolution formal masks to fit
the six measured enclosure planes. It does not hand-place a box.

The repaired v2 cuboid mesh and v4 caches pass the Stage C geometry gate. At
source resolution the projected mesh has mean/minimum mask recall
0.97200/0.91240 and mean IoU 0.95128. The 111 strict cache reloads have mean
hard-mask validity 0.97038, minimum 0.90821, mean eroded validity 0.99117,
`t_far > t_near` on every valid ray, and zero rays with more than two distinct
crossings. Source-resolution worst-view overlays and fixed-view near/far/valid
maps show a coherent front/back cuboid without the former holes or folds.

This establishes the Stage D geometric prerequisite under an explicit
TiHuBird-glass-is-a-six-plane-enclosure assumption. Reflection rays remain
enabled on every valid D surface and RGB remains full-frame; the Stage C
geometry work itself created no Transmittance Gaussian, T training, or second
bounce.

Stage C is formally accepted and frozen as `stage_c_geometry_release_v1`.
The Git-tracked manifest is
`geometry_releases/stage_c_geometry_release_v1.json`; immutable assets are at
`output/stage_c_geometry_release_v1/`; aggregate SHA-256 is
`4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d`.
The independent CPU-only audit returns `STAGE_C_GEOMETRY_RELEASE_VALID` for
111/111 caches and records that no runtime mesh/cache generation is required.
The release directory and its mesh/cache source directories have no write bits.

Stage D is current. Its implementation and minimum real-scene smoke are
complete. It may only read this exact release, must store its release ID and
aggregate in every checkpoint, and may not refit, regenerate, or modify
geometry. The user has now authorized exactly one formal 5,000-step T-onset run;
Stage E and continuation beyond global 20,000 remain unauthorized.

Stage D implementation now includes an independent Transmittance surfel model,
optimizer/scheduler/topology/checkpoint/PLY namespace; validated frozen-cache
loading; first-bounce T and second-bounce D ray tracing; alpha-over Ct/At;
thin-shell `(1-F)` transmission weighting; L_depth; full D/R/T composition;
debug maps; and bounded smoke telemetry. It is awaiting the authorized minimum
real-scene smoke and is not yet accepted.

The first real invocation (`smoke_v1`) stopped at 0 steps on an operator path
that repeated the scene prefix for the formal-mask manifest. The corrected
`smoke_v2` passed release/mask/camera loading but stopped before its first
backward because Scene camera names include `.jpg` while release keys are bare
stems. Both attempts are preserved; neither performed an optimizer update or
wrote a checkpoint. Stage D now canonicalizes camera identity with
`Path(image_name).stem` before frozen-cache lookup.

The new `smoke_v3` completed global 15,001--15,002 from the frozen Stage B
source, then actually restored `chkpnt15002.pth` and completed global 15,003 / R
local 12,003 / T local 3. The CPU-only final audit returns
`STAGE_D_SMOKE_PASSED_AWAITING_REVIEW`. Final D/R/T counts are
258,593/3,088/4,096; all three parameter namespaces changed by finite nonzero
amounts and retain separate optimizer state. The final checkpoint SHA-256 is
`f71d4644fcb2873ddc9d0ea058c87ce698c405d2f1b43e83b7fdd9df4349e560`.

All three telemetry rows have zero non-finite count. Maximum observed scoped
allocated/reserved memory is 3,483,877,888 / 4,395,630,592 bytes and training
steps take 5.94--8.29 seconds. L_depth is finite and active at every smoke step;
fixed debug view 000039 has `Din <= t_far` on 0.63439 of valid pixels. Inside T
color is finite/nonzero but faint after only three updates; outside D color is
structured from the frozen back-position query. Final glass is over-bright, so
this is a technical smoke pass, not semantic T separation or Stage D acceptance.
That smoke verdict did not itself authorize long training. The subsequent
formal authorization is limited to a new global 15,001--20,000 trajectory from
the original Stage-B global-15,000 source; Stage E is still unauthorized.

The committed formal-onset operator is fail-closed to that source hash, the
frozen release aggregate, fresh-T `random_bbox/4096/20260703`, global-20,000
endpoint, and global-40,000 depth start. The current v2 policy uses checkpointed
2,048-ray chunks with same-camera 1,024/512 OOM fallback and a unique v2 output.
It records every training step and automatically writes complete
checkpoint/PLY/nine-view review nodes at 15,100/15,500/16,000/17,500/20,000,
then runs a CUDA-hidden recursive final audit. Updated v2 regression results are
149/149 with CUDA enabled and 132 passed / 17 skipped with CUDA hidden.

The first operator invocation performed no training and created no output: its
GPU preflight conservatively classified the persistent 260 MiB
`gnome-remote-desktop-daemon` C+G context as a conflicting compute job. The
preflight now permits only that exact display-service path up to 512 MiB and
records it; every other compute process remains a hard conflict.

The next invocation also stopped before its first update: the selected source
checkpoint contains one CUDA RNG state, while the operator had exposed two
devices. Its exact zero-step metadata/log footprint is preserved under the
formal v1 output and is never a resume source. The operator now audits this
cardinality before launch, exposes only GPU 0, and may reuse the required v1
root only after moving that exact allowlisted footprint into a named archive;
any checkpoint, telemetry, debug, PLY, unexpected file, hash change, or
different failure text blocks reuse.

The subsequently launched v1 formal trajectory was explicitly paused by the
user after 18 complete finite updates (global 15,001--15,018) because its
steady 512-ray checkpointed path projected 8.81 hours. It has no checkpoint,
PLY, or review node and is excluded from resume. Median steady step time was
6.340 seconds; time/T-valid-pixel correlation was 0.9700; peak allocated/
reserved memory was only 3.75/4.46 GiB. The authorized v2 restart therefore
returns to the original global-15,000 source and fresh T, uses checkpointed
2,048 with same-camera pre-commit OOM fallbacks at 1,024/512, and releases cache
only under 2 GiB device-free pressure. No rendering/loss/data formula changes.

The user subsequently stopped v2 after global 15,065 because its roughly
3.19-second steady steps still projected about 4.4 hours. It has no authorized
resume checkpoint and remains preserved as aborted evidence. A no-update
profile localized the remaining time to repeated frozen D/R work, candidate
gradient reduction, checkpoint recomputation, and many small CUDA launches;
unused memory capacity was not the limiting resource.

The currently authorized replacement is the new cached-T-warm-up then exact-
joint operator. It always restarts from the original Stage-B global-15,000
source with the same fresh T and immutable Stage-C release. Phase A is global
15,001--18,000: D/R model, optimizer, scheduler, densification, topology, and
R-local state are frozen at R-local 12,000 while a source/release/mask/renderer/
camera-bound FP32 cache removes repeated D/R computations. Phase B is global
18,001--20,000: the cache is disabled and exact D/R/T joint training resumes,
ending at R-local 14,000 and T-local 5,000. This R-local meaning differs from
the abandoned all-joint plan, and Phase A is not full joint training.

The implementation includes a fail-closed 10-view output/loss/T-gradient
parity gate, D/R full-state hashes, a no-update performance benchmark, eight
complete checkpoint/PLY/nine-view nodes, continuous phase-aware telemetry, and
a CPU-only final audit. `L_depth` remains off through global 20,000 with future
activation recorded at global 40,000.

The user-launched cached-T v1 passed its 111-view cache build and ten-view
parity gate. Output and loss differences were exactly zero; the largest T
gradient difference was below `7.3e-12`. View 000018 measured 254.95 ms cached
T-only versus 1,909.46 ms frozen-uncached median, a 7.49x speedup. Training then
completed 25 finite T-only updates but failed while exporting the first global-
15,025 review node: the CHW `[1,269,478]` mask was expanded directly against an
HWC `[269,478,3]` RGB tensor. Telemetry is complete only through global 15,024,
although a finite 15,025 checkpoint was written immediately before the debug
failure. V1 is preserved, classified `CACHED_T_WARMUP_BLOCKED`, and is never a
resume source.

The debug-only CHW-to-HWC normalization is covered by CPU tests. The authorized
retry uses a new v2 output, restarts from the original global-15,000 Stage-B
checkpoint with fresh T, and fail-closed audits the exact v1 failure artifacts
before launch. No v2 cache build, benchmark, render, or GPU training has been
started by the implementation agent.

Historical Stage B record follows. Stage A is formally closed. Stage B-0 was approved by the user and
B-1a/B-1b/B-1c implementation is present with synthetic/CUDA tests passing.
The first user-operated TiHuBird smoke restored D and initialized R, then
aborted before its first optimization step on an ASCII-locale CUDA JIT path
error. The user-operated `smoke_2_retry1` then completed 2/2 steps from the
fixed code and produced a valid Stage B checkpoint, independent D/R PLYs, and
real debug maps. After the first diagnostic resume stopped on the RNG
map-location bug, `resume_diag_15003_retry1` completed and verified B-1d at
global 15,003 / R local 3. The first controlled health pilot was interrupted
with SIGINT after global 15,019 / R local 19 and classified
`PILOT_ABORTED_FOR_PROFILE`: ordinary steps took roughly 50--60 seconds, so its
partial output is preserved and must never be used as a resume source. Nsight
then isolated 160 long candidate advanced-index backward kernels as 97.8% of a
step. Commit `c87bb66` replaces only that R-parameter gather backward with a
grouped custom CUDA reduction. A matched two-step profile from the original
15,003/3 checkpoint reduced global 15,004 from 50.461 s to 1.572 s with exact
15004/15005 scalar losses. The fresh controlled 100-step retry then completed
from global 15,003 / R local 3 to global 15,103 / R local 103 without OOM,
non-finite values, count changes, or performance collapse. It is accepted as a
health pilot, not as Stage B acceptance or authorization for long training. The
formal reviewed-v1 mask archive, fail-closed loader, and three-step
`lambda_spec=0.2` smoke are complete, and the user accepted the smoke
transparent-mask/overlay alignment. The single authorized 100-step
`lambda_spec=0.2` pilot then ran from the original read-only 15,003/3 checkpoint
to 15,103/103. It completed without crash/OOM, non-finite debug stats, count
changes, or candidate/exact-intersection explosion, and fixed-view mask-inside
ks rose relative to the original 15,003 fixed view. Its non-smoke training path
did not persist whole-step CUDA allocated/reserved peaks, so exact allocator
trend remains an evidence gap. Stage B is neither complete nor accepted. The
frozen StableNormal baseline and C03 initialization artifacts remain unchanged.
At that historical Stage-B record point, Stage C and Stage D had not started.
The current-stage record above supersedes that historical sentence.

The bounded Tier-1 observability path was subsequently used by the user for
matched 7k, 10k, and 15k fresh-R handoff pilots through R-local 175. Read-only
audits classify 7k and 10k as `CONDITIONAL PASS` because D still performs one
expected endpoint densification and long-horizon evidence is absent; 15k is a
`HEALTHY` control. Candidate/exact counts, raytrace time, allocator evidence,
L_spec behavior, and fixed-view structure remain in the same engineering range.
These results establish 7k only as the earliest conditionally viable checkpoint
tested so far, not as an optimum or Stage B acceptance.

An opt-in Tier-2 operator pack now implements version-2 full-state D-only and
Stage B continuation checkpoints, endpoint optimizer continuity, Python/NumPy/
CPU/CUDA RNG plus remaining-camera-deck restoration, bounded D telemetry, and a
CPU-only read-only gate packet generator. The user approved one internally
matched `resolution=8` experiment named
`C03-r8 Tier 2 onset study`; it is not a literal reproduction of the historical
C03 resolution-2 baseline and its absolute metrics must not be compared as if it
were one. The shared D bootstrap and both R-onset branches are fail-closed to
this exact experiment identity and resolution. A prior user-operated manually
gated v1 bootstrap reached 7k, Branch A reached 3,205/205, and Branch B reached
7,100/100 before a later attempted command collided with an existing telemetry
file. Those manual assets remain preserved and excluded.

The separate `oneshot_v1` execution at commit
`2c8191a32790a2789c8c6b87d6b622647d21abea` completed its fresh shared D-only
bootstrap and both R-local 1--100 warmups, then HARD_FAILED without retry in
both formal branches. Branch A's last telemetry is global 3347 / R-local 347
and its OOM occurred during backward while requesting 32 MiB; Branch B's last
telemetry is global 7179 / R-local 179 and its OOM occurred in raytrace forward
while requesting 26 MiB. All saved checkpoints are recursively finite and all
telemetry has `nonfinite_count=0`. The immutable final audit is
`output/tier2_c03_r8_oneshot_final_audit_v1.json`, SHA-256
`72887af7b8d93ee64f59747c745030970e96aa379e7dc09f980bd749af8df88a`.

The v1 evidence shows real camera/topology-dependent transient allocated peaks
up to 23,274,475,520 bytes plus allocator cache fragmentation: successful steps
periodically drop reserved memory only after allocator retry, while ordinary
end-of-step allocated memory remains roughly 1--2 GiB and no monotonic graph
retention is present. The `oneshot_v2_allocator_lifecycle_retry`
therefore changes only allocator lifetime: explicitly release completed-step
ephemeral references, call `torch.cuda.empty_cache()` at that safe boundary,
and set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` identically on A/B. A
formal local-101 headroom record fails closed unless projected capacity exceeds
the larger of the current step peak and the v1 23,274,475,520-byte reference by
at least 1 GiB. The protected v1 shared D-only 3k/7k checkpoints are reusable
because this policy is Stage-B-only and changes neither D bootstrap mathematics
nor RNG. Failed v1 A/B checkpoints are never retry resume sources.

The user-launched `oneshot_v2` then HARD_FAILED after the first warmup training
step in both branches because the observability refactor assigned the ordinary
telemetry dictionary to `record` but read `telemetry_record`. The exact failure
was `TypeError: 'NoneType' object is not subscriptable`; it was not OOM and
produced no usable branch checkpoint or telemetry row. The v2 final audit is
preserved at `output/tier2_c03_r8_oneshot_v2_final_audit.json`, SHA-256
`492464757a0776d6b463a80d0b89fe2e668549586ab5a94bb01394142c0bcb7a`.
The corrected run identity is `oneshot_v3_allocator_lifecycle_retry`, with
entirely new v3 output/log/state/audit/JIT paths. V1 and v2 branch outputs are
never reused. V3 then also HARD_FAILED: Branch B was conservatively stopped at
global 7101 / R-local 101 by the old cross-branch headroom gate, while Branch A
continued to global 3764 / R-local 764 and then OOMed in backward on a 16 MiB
request with 21.98 GiB allocated and only 1.38 MiB device-free. Its last full
checkpoint is global 3500 / R-local 500; all completed telemetry is finite. The
v3 final audit is preserved with SHA-256
`776ab1102ad74e90ea35985e45ef29b9e73fecb99c20d86b71d53845858c9a4f`.

V3 proves that completed-step `empty_cache()` cannot bound a single step's live
autograd peak and adds about 46% ordinary-step latency. The subsequently executed v4 policy
uses base ray chunks of 2048, detaches no-grad R densification auxiliaries,
reclaims cache only below 2 GiB device-free, and automatically retries the same
camera before any optimizer/topology update at 1024, 512, then checkpointed 512.
The checkpointed CUDA path recomputes ray candidates/intersections during
backward and has matching tested outputs and D-ray/R-parameter gradients. V4
used entirely new output/log/state/audit/JIT paths and never resumed v1/v2/v3
branches. Both branches completed; the formal closeout above supersedes this
historical pre-run status.

Stage B branch point: `772c0c0e1c9fec012a10795101e874e2bc065c44`

Stage B implementation commits:

```text
7ad0b8881b9b2a00e9f766b68be7ff051c74281e  rendering/path decisions
bb5eb072e757a32735ad378b38a3d4b489ac1a57  B-1a model/math/checkpoint foundation
30dd8be5d05d6da9081b0a1f4b80a555234b74f3  B-1b fail-closed CUDA LBVH
953334009a8db3d63cc68179842687ba306186c2  B-1c train/render/debug integration
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

Stage A code evidence commit: `829dd82dc74f4c5dce640201448626df24afa25a`

## Repository and scene

- The original 3DGS baseline remains available through `--model_type 3dgs`.
- RT-GS uses `--model_type surfel` and the pinned 2D surfel rasterizer.
- Diffuse and Reflection now have separate Stage B models, optimizers,
  schedulers, densification state, checkpoint namespaces, PLY exports, and debug
  outputs. No Transmittance model or Stage C/D path exists.
- `data/TiHuBird` is the real acceptance scene. Truck artifacts are historical
  engineering smoke evidence only.
- TiHuBird has 111 final undistorted PINHOLE images at 3827x2152, 111/111 COLMAP
  registrations, and 84,989 sparse points.

## Frozen StableNormal Stage A baseline

- The frozen rollback branch is `baseline/stage-a-stablenormal` at
  `3e9f62ae8b34eaaf77ce67ba86a077041aaa5f04`.
- The accepted D-only output is
  `output/stage_a_tihubird_30k_mono001_r2_retry1/` with
  `lambda_mono=0.01`. Its 30,000-step checkpoint and point cloud remain present
  and must not be overwritten, moved, or modified.
- Generated all 111 TiHuBird StableNormal priors in
  `data/TiHuBird/normal_priors/` and passed strict validation: 111 images, 111
  priors, 111 manifest entries, with empty missing, unexpected, duplicate, and
  abnormal lists. Evidence:
  `output/stage_a_tihubird_priors_validate_111.log`.
- The matched `lambda_mono=0` and `lambda_mono=0.01` retry1 runs completed to
  30,000 iterations. The first `lambda_mono=0.01` attempt stopped around
  iteration 230 without a usable checkpoint and remains excluded.
- Gate 9 matched metrics and the global comparison are present at
  `output/stage_a_tihubird_30k_matched_metrics.txt` and
  `output/stage_a_tihubird_30k_matched_comparison.png`; Gate 9.5 six-view and
  crop evidence remains under `output/stage_a_tihubird_gate9p5/`.

## Matched StableNormal 30,000-step metrics

| Iteration | `lambda_mono=0` L1 | `lambda_mono=0` PSNR | `lambda_mono=0.01` retry1 L1 | `lambda_mono=0.01` retry1 PSNR |
|---:|---:|---:|---:|---:|
| 7,000 | 0.0233018197119236 | 25.840939331054688 | 0.022160319611430168 | 26.222691726684573 |
| 15,000 | 0.022149086557328702 | 26.27469940185547 | 0.020849463716149333 | 26.722418594360352 |
| 30,000 | 0.01844301298260689 | 27.603187561035156 | 0.017842570878565313 | 27.78533058166504 |

## DiffusionRenderer C03 exploratory result

- DiffusionRenderer priors were evaluated independently on
  `feature/stage-a-diffrender-priors`; they never replaced or modified the
  StableNormal baseline.
- C03 completed 15,000 D-only iterations at
  `output/stage_a_tihubird_drnormal_c03_15k/`. The log
  `output/stage_a_tihubird_drnormal_c03_15k.log` records saves at 7,000 and
  15,000, checkpoint saves at 7,000/10,000/15,000, and `Training complete.`
- Its 15,000-step train metrics are L1 `0.021616848371922973` and PSNR
  `26.401429367065433`. The matched StableNormal retry1 15,000-step metrics are
  L1 `0.020849463716149333` and PSNR `26.722418594360352`.
- C03 trained stably and did not show the front glass surface swallowing the
  bird subject. The RGB/normal/depth audit nevertheless did not establish a
  clear D-only advantage over the frozen StableNormal baseline. Therefore C03
  will not be extended to 30,000 D-only iterations.
- The comparison image is
  `output/stage_a_tihubird_drnormal_c03_15k/comparison_vs_stablenormal_15k.png`.
- The C03 15,000-step checkpoint is a readable `rtgs_stage_a` checkpoint at
  iteration 15,000 with 346,118 finite Diffuse surfels. Its usable initialization
  artifacts are:

  ```text
  output/stage_a_tihubird_drnormal_c03_15k/chkpnt15000.pth
  output/stage_a_tihubird_drnormal_c03_15k/point_cloud/iteration_15000/
  ```

- These C03 artifacts are read-only Stage B initialization candidates. C03 is
  not a proven replacement for the StableNormal D-only configuration.

## Tests and verified behavior

- `MAX_JOBS=4 conda run --no-capture-output -n RT-GS python -m pytest -q
  tests` → 88 passed on 2026-07-01. This includes the complete Stage A
  regression; Stage B model/ray/BRDF/checkpoint/render/debug/mask/JIT staging
  and candidate-gather equivalence tests; and fifteen DR proposal/review/repair
  audit, output, fail-closed, read-only, and training-isolation contracts.
- The CUDA extension compiled successfully for PyTorch 2.0.1 + CUDA 11.8 on an
  RTX 3090. CUDA/oracle consistency, chunking, refit/rebuild, and nonzero
  finite-difference gradients for xyz/rotation/scaling/opacity/color/ray
  origin/ray direction passed.
- Original 3DGS and surfel training/render paths, checkpoint resume, PLY reload,
  densification, material gradients, and Stage A debug exports have passed their
  documented checks.
- Synthetic Stage B rendering produced real reflection hit/color/alpha/depth,
  D/F/G/fr/wr, and diffuse/reflection contribution tensors and debug files in
  pytest temporary directories only.
- The user-operated `smoke_2` attempt created only initialization/config/event
  artifacts and stopped at 0/2 steps before any checkpoint or D/R PLY save. Its
  attached log identifies `UnicodeEncodeError` while PyTorch writes the CUDA JIT
  Ninja file from the non-ASCII repository path. Codex did not run this smoke or
  modify its partial output.
- A fresh forced-C-locale CUDA compile reported preferred encoding
  `ANSI_X3.4-1968` and loaded the extension successfully from the new ASCII-only
  staging path. At that smoke point no real manual soft-mask set existed; the
  later reviewed-v1 archive is recorded below.
- The user-operated `smoke_2_retry1` completed at global iteration 15,002 and R
  local step 2. Its `rtgs_stage_b` checkpoint contains 346,118 Diffuse and 4,096
  Reflection surfels, separate optimizer namespaces, and no non-finite tensor in
  recursive D/R state inspection. The independent D/R PLYs and both debug nodes
  are present. C03 source checkpoint and PLY hashes remain unchanged.
- The reviewed fixed view had 128,582 valid rays and 128,579 final alpha hits.
  Reflection color/alpha/depth were spatially structured, but the physical
  reflection contribution quantized to black and microfacet D display-clamped
  to white. B-1d now preserves these physical images while adding p99/log
  companion views, raw statistics, candidate/exact-intersection counts, CUDA
  phase timing, and peak-memory metadata only on requested debug renders.
- The first B-1d resume attempt loaded all cameras and the Stage B checkpoint,
  then stopped at 0 steps because `map_location="cuda"` moved the CPU PyTorch
  RNG ByteTensor onto CUDA. Its partial output contains only configuration,
  camera/input, and event artifacts; no checkpoint or D/R PLY. The corrected
  loader validates RNG uint8 tensors, restores contiguous CPU copies for both
  CPU and CUDA generators, passes an on-disk CUDA-map-location test, and restored
  the real user checkpoint RNG read-only.
- The corrected user resume completed from global 15,002 / R local 2 to global
  15,003 / R local 3 and wrote a finite checkpoint plus separate D/R PLYs. Its
  fixed view reports candidate p50/p95/p99 `111/205/256`, exact-intersection
  p50/p95/p99 `28/64/76`, about 152 ms raytrace time for 128,582 rays, and about
  319 MiB incremental peak allocation. Candidate p99 is 6.25% of the field, so
  the LBVH is not behaving as a 4,096-surfel brute-force scan.
- Microfacet D is finite and tightly distributed over `1.270–1.276`; its nearly
  white visualization is correct. Reflection contribution is nonzero and
  spatially structured in the companion map. No Reflection count or initial
  scale adjustment is justified before the controlled pilot.
- The aborted `health100` output contains completed TensorBoard scalars through
  global 15,019, but it is neither healthy nor blocked evidence and is not a
  resume source. The original read-only
  `resume_diag_15003_retry1/chkpnt15003.pth` remains the sole pilot resume point
  (SHA-256
  `ad92c7d75312cc5df60b7a1dd5d762d8e7d65b8de0f90264d742f969baf7e144`).
- The matched Nsight comparison uses global 15,004 bounded by consecutive D
  raster forwards. Wall time fell from `50.461 s` to `1.572 s`; the R backward
  envelope fell from `49.524 s` to `0.547 s`. The old 160 long
  `indexing_backward_kernel` calls (`49.356 s`) are absent; the nine remaining
  unrelated short calls total `0.430 ms`. The new 32 grouped reducers total
  `253.612 ms`. R forward, D rasterization, VGG, and optimizer timing remained
  in the same sub-second regime. The 15004/15005 losses are exactly unchanged:
  `0.08710507303476334` and `0.06420626491308212`.
- The new profile's 200 ms external sampler observed a 15,067 MiB device-total
  peak (302 MiB idle baseline, approximately 14,765 MiB attributable to the
  process). This is higher than the earlier separately observed 10,627 MiB
  total / 10,320 MiB process peak and remains a short-pilot risk; it does not
  negate removal of the measured backward bottleneck.
- The only post-repair health pilot completed at
  `output/stage_b_tihubird_reflection_dr_c03_health100_candidate_reduce_g15003_15103_retry1/`.
  Its 95 non-debug iteration intervals have p50/p95/mean
  `0.939/1.092/0.944 s`. Mean training raytrace forward and candidate-backward
  envelope are about `228.7 ms` and `643.0 ms`; all 100 iterations used 32
  candidate backward calls. Four evaluation/debug nodes and the accepted final
  15,103 debug node are complete with all companion maps.
- D/R counts remained `346,118/4,096`; no densify/prune event changed topology.
  Candidate p50/p95/p99 moved from `126/227/285` at 15,025 to `146/265/341` at
  15,103, while exact intersections moved from `33/72/86` to `38/83/109`.
  This is gradual growth, not an order-of-magnitude acceleration failure.
  Reflection contribution remained finite and essentially fully nonzero; its
  mean/p99 increased from `1.08e-4/5.58e-4` to `1.85e-4/1.12e-3`.
- A read-only fixed-view render of the final checkpoint found 128,582 valid
  surface pixels with ks min/mean/p1/p5/p50/p95/p99/max
  `0.08796/0.09726/0.09321/0.09468/0.09730/0.10000/0.10189/0.10640`;
  both `ks<0.01` and `ks>0.9` fractions are zero. The field did not collapse.
- Exact allocator evidence is peak allocated `18,094,459,392` bytes
  (16.85 GiB) and peak reserved `20,333,985,792` bytes (18.94 GiB). Current
  allocated memory after compute is non-monotonic (only 48.5% of transitions
  nondecreasing, 1.36--1.82 GB), so no live-tensor leak is evident. Reserved
  memory is 96.0% nondecreasing and rose from 12.51 GB to 20.33 GB as PyTorch
  cached larger per-view temporary blocks. This is a material long-run headroom
  risk even though the pilot did not OOM.
- Final checkpoint `chkpnt15103.pth` is `rtgs_stage_b` at global 15,103 / R
  local 103 with SHA-256
  `6f43ff1335f99e03f8ee08e4575ad4c91b29189cbe23a67942b23557adb87354`.
  Recursive inspection covered 63 tensors / 19,912,887 elements and found no
  NaN/Inf. It records `lambda_spec=0` and no specular-mask manifest.

## DR glass-mask proposal audit (not supervision)

- The strict input audit passed for all 111 real TiHuBird views. Source RGB is
  JPEG RGB uint8 at 3827x2152; stored DR RGB, normal, depth, basecolor, and
  diffuse_albedo are PNG RGB uint8 at 704x384. Source hashes, ordered stems,
  dimensions, manifest records, generation-time alignment validation, and all
  artifact paths/hashes were checked. Raw slots 111--119 are the nine final
  chunk padding records and are excluded from every real-view mapping.
- DR normal is decoded as `rgb / 127.5 - 1`, then normalized only for
  continuity cues. Its component/axis convention remains unconfirmed; no
  semantic axis interpretation is used. The 111 real normal maps have no
  non-finite or decoded-near-zero pixel. DR depth is a per-frame relative RGB
  visualization with no documented invalid sentinel or COLMAP/metric scale;
  only within-view boundaries and continuity are used.
- Nine automatic review proposals were generated for stems 000000, 000014,
  000028, 000042, 000055, 000069, 000083, 000097, and 000110 at
  `output/stage_b_tihubird_dr_glass_proposal_9views_v2/`. Each contains the
  requested RGB/DR/boundary/proposal/uncertainty views and metadata; the root
  contains the strict audit manifest, contact sheet, proposal index, and human
  review contract.
- These files are explicitly automatic drafts with no training role. No
  `reviewed_soft` directory, formal training mask manifest, mask loader input,
  `lambda_spec` run, or new scene training was created. Visual inspection finds
  the enclosure outline in all nine views, while every view conservatively
  flags possible bird inclusion because RGB texture exists behind the glass.
  Human review is required before any 111-view expansion decision.
- The user accepted the nine-view method and authorized a frozen-method
  111-view expansion. Commit `a2ee732` remains the proposal algorithm reference;
  no proposal formula, threshold, morphology, connectivity, or per-frame
  parameter changed. The completed review package is
  `output/stage_b_tihubird_dr_glass_proposal_111_v1/`.
- The package passed strict validation for ordered stems 000000--000110, source
  size 3827x2152, single-channel uint8 proposals containing both 0 and 255,
  input/output SHA-256 records, finite DR normals, nondegenerate area, and no
  padding intersection. It contains 111 review-queue entries, ten chronological
  overlay pages, ten risk-priority overlay pages, four boundary-review crops per
  frame, distribution plots/data, and an automatic anomaly list.
- Proposal area min/mean/p50/p95/max is
  `0.13856/0.22100/0.21222/0.31571/0.35572`. Uncertainty fraction above
  140/255 has min/mean/p50/p95/max
  `0.001783/0.003679/0.003812/0.004875/0.005206`. Five frames carry the
  background-inclusion review flag: 000012, 000013, 000039, 000040, and 000041.
  Fifteen frames carry the reflection-misclassification flag; all 111 retain the
  conservative bird-behind-glass flag.
- Visual inspection of all ten chronological pages found continuous enclosure
  outlines without a systematic table/wall capture or enclosure loss. The five
  automatic anomalies still align visually and remain first-priority human
  review items. No automatic proposal was deleted, rewritten, or promoted to a
  reviewed mask.
- A separate 30-view high-risk human-review pack is complete at
  `output/stage_b_tihubird_dr_glass_high_risk_review_pack_v1/`. Ordering is the
  user-specified five background-risk views, fifteen reflection-risk views, and
  ten highest-uncertainty views. Every 2748x2308 pack contains RGB, soft mask,
  proposal overlay, boundary-only RGB overlay, uncertainty views, statistics,
  and six direct source-pixel crops: top, bottom/yellow plate, left, right,
  base/black support, and strongest reflection.
- Each crop is 900x506 source pixels with no resampling and is stored both as
  raw RGB and boundary overlay. The package has 30 packs, 180 raw crops, 180
  boundary crops, one ordered contact sheet, and blank JSON/CSV/Markdown human
  checklists. The source proposal tree contains 1,110 files and has identical
  before/after SHA-256
  `428139079931e8091409be70c1c4442c457e0869b10fea894fd2986d0f6b6d1e`.
  Therefore no proposal file changed and no `reviewed_soft` was created.
- Human initial review marks 000039, 000040, and 000041 as failed and needing
  localized repair. The other 27 high-risk-pack views temporarily pass visual
  review. This is not `reviewed_soft` acceptance and authorizes no mask loading,
  `lambda_spec`, or training.
- Independent repair candidates are at
  `output/stage_b_tihubird_dr_glass_repair_candidates_v1/repair_candidates/`.
  The detected failure is the v1 convex top boundary reaching image row zero
  and absorbing dinosaur/ceiling background. Each repair uses RGB glass-edge,
  DR normal/depth boundary evidence, and 000038/000042 continuity to remove only
  the erroneous top region. Added hard pixels are zero for all three; no other
  proposal was regenerated.
- V1-to-repair hard changed ratios are 0.026212 / 0.028458 / 0.030071 for
  000039 / 000040 / 000041. Areas change from
  0.210159/0.215424/0.229978 to 0.183947/0.186966/0.199907. The complete source
  proposal tree retains identical before/after SHA-256
  `428139079931e8091409be70c1c4442c457e0869b10fea894fd2986d0f6b6d1e`.
  Visible top-edge spans cover about 55.9%, 63.1%, and 41.1% of the repair bbox;
  hidden spans behind dinosaur/reflection remain explicit manual-review risks.

## Formal reviewed-v1 soft-mask archive

- The user accepted the three repair candidates and authorized a complete
  formal mask archive. `data/TiHuBird/specular_masks_reviewed_v1/` now contains
  exactly 111 mode-L uint8 3827x2152 masks plus one manifest. Files 000039--
  000041 are byte-identical to accepted repair candidates; the other 108 are
  byte-identical to frozen proposal v1. The directory/files are read-only.
- The aggregate mask, canonical manifest-payload, and manifest-file SHA-256 are
  `54dbb7661efbb2a334d86cef1cfec1d15ff812e71856c0d88930754013abc2e6`,
  `026ad1fa28fb7c2a30656fd37976a4315585e056a317e59f7c44502d69831e4f`,
  and `056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551`.
  Manifest validation rehashes 111 RGB/mask pairs and proves padding 111--119
  contributes zero masks.
- Training now accepts only this formal manifest contract. It rejects raw
  proposal/repair paths, missing/extra/corrupt masks, non-L images, dimension
  mismatch, unknown sources, and manifest/hash mismatch. Soft resize is
  recorded as `opencv.INTER_LINEAR`; full-image RGB reconstruction is unchanged.
- The pre-telemetry Stage A/B suite was 95 passed. The current suite is 120
  GPU-enabled passes and 104 CPU-only passes with 16 CUDA skips after the v4
  allocator-lifecycle regressions. This includes formal archive exact-copy
  tests, fail-closed loading, continuous soft resize, L_spec mask-only gradient,
  whole-image RGB, debug transparent-mask/overlay, and all prior regressions.
  This remains implementation evidence, not Stage B acceptance.

## Tier-1 handoff observability readiness

- `--stage_b_telemetry_jsonl`, `--stage_b_telemetry_max_steps`, and
  `--stage_b_telemetry_phase_tag` enable a strict, append-only JSONL stream.
  The mode is off by default, refuses a missing/nonpositive bound, refuses a
  planned run longer than the bound, and refuses a nonempty destination.
- Ordinary telemetry reuses the existing training render, losses, gradients,
  counts, R topology version, CPU loop boundary, and allocator counters. It
  resets only PyTorch peak counters at loop entry. Debug steps combine the peak
  observed before debug begins with the post-reset peak; transient debug work
  before the raytrace's internal reset remains explicitly unavailable.
  Current allocated/reserved and combined scoped maxima remain distinct from
  external `nvidia-smi` process/device memory.
- Candidate/exact percentiles and raytrace-forward time are available only when
  the pre-existing smoke diagnostic path already produced them. They are null
  with explicit `unavailable_fields` in an ordinary Tier-1 step; candidate
  backward timing is always unavailable because no such boundary currently
  exists. Telemetry never enables diagnostics to fill these fields.
- Phase A (`lambda_spec=0`) does not validate or load the formal mask and writes
  null mask/ks fields. Phase B computes mask support and inside/outside ks
  summaries on CPU from the already-rendered forward tensor. These describe the
  state used by total loss before the optimizer step; no post-update rerender is
  performed, and mask-outside total-loss gradients are not claimed to be zero.
- `tools/compare_stage_b_pilots.py` imports neither training nor CUDA paths and
  hashes the selected input debug trees before and after processing. It writes
  common scales to metadata. If an existing physical 8-bit reflection map has
  quantized all spatial signal to zero, reflection and delta-reflection output
  are reported unavailable rather than reconstructed from scalar metadata.
- The user completed matched 7k/10k/15k Phase A/B Tier-1 pilots. Read-only
  results are 7k/10k `CONDITIONAL PASS`, 15k `HEALTHY`; none is a long-horizon
  onset comparison or Stage B acceptance.

## Tier-2 operator-pack state

- `--operator_gate_continuation` is default-off. When enabled it requires
  bounded telemetry, explicit restored global/R-local starts, full-state
  version-2 checkpoints, and an optimizer update at every bounded endpoint so
  segmented gates match one continuous optimization trajectory.
- Full-state checkpoints include D/R model and optimizer/densification state,
  global/R-local progress, Python/NumPy/CPU/CUDA RNG, remaining camera-deck
  indices, complete schedule/data contract, and an explicit completed-endpoint-
  update marker. Fresh R uses its independent seeded generator and is tested not
  to change restored global RNG or the shared source checkpoint.
- `tools/audit_tier2_gate.py` loads checkpoints on CPU, slices a bounded JSONL
  interval, scans every nested tensor, reads endpoint debug metadata and log
  anomalies, and prints a compact read-only packet. It imports no training,
  renderer, or raytracer path.
- `tools/tier2_operator.sh` uses finite internal phase actions and an explicitly
  acknowledged `run-all --execute` coordinator, read-only `status`, and
  CPU-only `final-audit`. For the v4 retry, the coordinator assigns Branch A to
  physical GPU 0 and Branch B to physical GPU 1, records process groups and
  terminal state, verifies the protected v1 shared 3k/7k sources without
  modifying them, and advances automatically without the former human GO
  pauses.
- Existing manual-v1 and failed `oneshot_v1/v2/v3` assets are preserved. Fresh
  `oneshot_v4` output/model/log/state/audit paths fail closed if any already
  exist. Failed
  v1 A/B checkpoints are explicitly excluded as resume sources.
- The approved experiment identity is exactly `C03-r8 Tier 2 onset study` at
  `resolution=8`. The operator wrapper, cfg_args, telemetry phase tags, output/
  log names, and version-2 checkpoint configs record that identity. Operator
  mode rejects any other name or resolution before entering the training loop.
- This experiment is internally matched across the shared bootstrap and Branch
  A/B, but is explicitly not a strict reproduction of legacy C03 resolution 2
  and must not be mixed with r2 artifacts or absolute-metric comparisons.

## Current boundary and next exact task

- Preserve all Tier-1, manual-v1, and `oneshot_v1` checkpoints, metrics,
  diagnostics, D/R PLYs, telemetry, state, audit, and logs.
- `lambda_spec=0` must keep `--specular_masks` empty and emits no mask/overlay.
  The authorized three-iteration `lambda_spec=0.2` smoke stopped at global
  15,006 / R-local 6 and the user accepted its transparent-mask/overlay
  alignment. The controlled 7k/10k/15k Tier-1 pilots are complete and remain
  evidence-only. The `oneshot_v1` long attempt is a preserved HARD_FAILED run.
  The user, not Codex, may execute the committed `oneshot_v4` retry. R-local
  1--100 remains mask-free; local 101+ uses the formal mask and
  `lambda_spec=0.2`. Both branches use the same adaptive memory policy, and the
  first formal step records advisory headroom without terminating a successful
  step. The output
  state and CPU-only final-audit JSON are authoritative for v4 progress and
  outcome.
- Stage B solves reflection only. It must not claim transmission,
  bird/background separation, transparent mesh, or two-hit geometry.
- Do not start a matched StableNormal-versus-C03 Stage B comparison in parallel.
  It may be designed later if the reflection hypothesis requires it.
- Historical Stage-B restrictions are closed. Stage C is accepted and Stage D
  is current under its own explicit release and run contracts.

Pending before acceptance: user evaluation of the real-scene `lambda_spec`
pilot's transparent mask, overlay, ks, reflection contribution, and L_spec
evidence. Exact whole-step allocated/reserved memory was not persisted in the
non-smoke training path, so reserved-memory headroom must remain a gating item
for any later pilot or long run. Stage B acceptance cannot advance before that
evidence is judged sufficient by the user.

## Stage D cuboid-front ownership v4 implementation ready (not executed)

- The user-executed v3 is preserved as `SEMANTIC_REPAIR_PILOT_BLOCKED`. It
  completed all 1,000 updates with D/R frozen and T fixed at 4,096, but final
  transparent Ain saturation was 0.8744 and the high-Ain/near-black fraction
  was 0.9723. Center-space outside filtering plus anti-veil did not repair the
  D-derived first-bounce path or branch ownership. Neither v2 nor v3 is a
  permitted resume/cache source.
- V4 implements `cuboid_front_v1`: transparent valid R/T paths derive front
  position, entering-face normal, direction, reflection direction, Fresnel
  normal, T origin, and absolute Din baseline from the immutable Stage-C
  release. Missing or inconsistent frozen geometry fails closed. Nontransparent
  D/R rendering remains on the legacy path.
- D/R/T share a finite 3-sigma support partition. Pilot T is support-safe
  strict-inside, fixed at 4,096 with no densification/pruning; Cout accepts only
  strict-outside-safe D support; transparent D direct and R contributions are
  zero. Per-class surfel/candidate/hit/energy and path-consistency diagnostics
  are retained. These versioned gates are permanent renderer configuration,
  while the pilot's D/R-off choice is only an isolation diagnostic.
- The matched A/B operator compares random strict-inside T with a copy-only,
  evidence-filtered D-inside transfer whose opacity is reduced and capped.
  All source, release, path, ownership, camera, loss, schedule, count, and seed
  settings match. The transfer is an ablation hypothesis, not semantic ground
  truth.
- The operator is fresh-only from source SHA-256
  `050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84`
  and Stage-C aggregate
  `4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d`.
  It builds an incompatible v4 cache, runs parity and two sequential 500-step
  arms, then a CPU audit at global nodes 15,000/15,100/15,250/15,500. L_depth
  remains disabled with its future marker at global 40,000.
- No v4 real-scene cache, benchmark, render, GPU training, or formal output has
  been launched or created by Codex. No independent versioned bird ROI was
  found by the read-only repository search, so a technically healthy execution
  must remain HOLD unless the operator finds a valid human-authored asset.
- V4 cannot authorize joint fine-tuning or Stage E. Even PASS would establish
  only T handoff conditions under cuboid-front/strong ownership isolation.
- The first user launch from commit `b697c25` stopped before cache construction
  or any training step because ownership mode was omitted from the cuboid-space
  construction gate. Source/release hashes stayed unchanged and there are no
  checkpoints or telemetry. The fix now shares the cuboid requirement across
  semantic and ownership modes. A same-command retry atomically preserves only
  this exact zero-step failure under the `_failed_preflight_b697c25` suffix;
  all other existing-output cases remain fail-closed. No retry was launched by
  Codex.
- The user retry on `17d2663` completed the 111-view v4 cache, parity, and all
  500 Arm-A updates. Arm B reached global 15,114, then correctly stopped when
  an optimized raw tangent scale became too large for any strict-inside 3-sigma
  support center. V4 now keeps copied/optimized raw scale unchanged but applies
  a rotation-aware uniform active-scale cap only at cuboid capacity, recording
  cap telemetry and auditing the same decode rule. The next same-command retry
  archives this attempt as `_failed_scale_17d2663` and rebuilds both arms from
  global 15,000 under one commit; no partial result is resumed or reused. Codex
  has not launched that retry.
- The `7c37088` ownership A/B subsequently completed with technical HOLD only
  because no independent bird ROI exists. Transferred Arm B is the selected
  bounded continuation: fixed-nine transparent L1 reaches 0.10741 versus
  random Arm A 0.13369, and T-off worsens every fixed view. Black-veil fraction
  remains approximately zero, but Ain p95 is already 0.9891 and 16 supports are
  scale-capped, so continuation requires rolling saturation/black/cap guards.
- A fresh-only transferred-T operator is prepared for global 15,501--20,000
  from the exact Arm-B 15,500 checkpoint. It freezes all D/R state, reuses the
  v4 cache, retains strong ownership isolation, fixes T at 4,096, keeps L_depth
  off, and cannot enter joint tuning or Stage E. No long run has been launched
  by Codex; without a bird ROI, a healthy endpoint remains HOLD.

## Three-step formal-mask L_spec smoke

- Source checkpoint remained the original read-only global 15,003 / R-local 3
  artifact with SHA-256 `ad92c7d75312cc5df60b7a1dd5d762d8e7d65b8de0f90264d742f969baf7e144`.
  The first `v1` launch restricted visible GPUs and stopped at 0 steps on CUDA
  RNG-state cardinality; it is preserved and excluded. The clean `v2` retry
  restored the original visibility and completed exactly 15004--15006 / 4--6.
- Per-step `(L_spec, total loss)` is `(0.164784, 0.120062)`,
  `(0.156079, 0.095425)`, `(0.130580, 0.085687)`. Formal-mask support fractions
  are 0.2158 / 0.2050 / 0.1719 for sampled views 000042 / 000079 / 000062.
  Inside-mask L_spec ks gradients are nonzero and point upward under gradient
  descent; outside-mask maximum absolute gradient is exactly zero every step.
- Inside-mask ks min/mean/p50/p95/p99 is
  `0.099775/0.099961/0.099978/0.100177/0.100220`,
  `0.099577/0.099931/0.099952/0.100222/0.100336`, and
  `0.099379/0.099950/0.099941/0.100344/0.100474`. Outside values remain finite
  and centered near 0.1.
- Candidate p50/p95/p99 is `113/194/253`, `106/215/267`, `121/219/267`;
  exact p50/p95/p99 is `26/55/73`, `25/64/78`, `28/65/75`. Warm raytrace wall
  is 218/208 ms. D/R counts remain 346,118/4,096. Allocator peak allocated/
  reserved is 13,538,393,088 / 15,101,591,552 bytes.
- Final checkpoint SHA-256 is
  `f46c00375699d3b4b7c018a4277b4ba3e93abc66f60ddc7f282a0caf979b93fb`.
  It records the exact formal manifest hashes and contains 63 recursively
  inspected tensors / 19,912,887 elements with zero non-finite values. Separate
  D/R PLYs and the complete fixed-view debug set exist at iteration 15,006.
- Technical classification: `L_SPEC_SMOKE_PASSED_AWAITING_USER_REVIEW`. It is
  neither Stage B acceptance nor authorization for a 100-step pilot.

## Controlled 100-step formal-mask L_spec pilot

- The user accepted the three-step smoke overlay and authorized exactly one
  controlled 100-step `lambda_spec=0.2` pilot. The run restored only the original
  read-only checkpoint
  `output/stage_b_tihubird_reflection_dr_c03_resume_diag_15003_retry1/chkpnt15003.pth`
  with SHA-256
  `ad92c7d75312cc5df60b7a1dd5d762d8e7d65b8de0f90264d742f969baf7e144`.
  It did not use the 15,006 smoke checkpoint, the `lambda_spec=0` health pilot,
  or any profile output.
- Output is
  `output/stage_b_tihubird_reflection_dr_c03_lspec_pilot100_g15003_15103_v1/`.
  The command used `--iterations 15103`, `--resolution 8`, R=4096,
  `--ray_chunk_size 4096`, formal manifest
  `specular_masks_reviewed_v1/manifest.json`, and `--lambda_spec 0.2`.
  The run completed exactly global 15,004--15,103 / R-local 4--103 and saved
  only the 15,103 Stage B checkpoint plus independent D/R PLYs.
- TensorBoard scalar intervals for the 95 non-debug ordinary steps have
  p50/p95/mean `0.931/1.089/0.939 s`. Intervals following the four debug nodes
  are 1.714, 1.845, 1.843, and 1.915 s. D/R counts stay 346,118/4,096.
- Per-view training `L_spec` is finite for all 100 sampled views. It is not
  monotonic because the sampled camera and mask support change; node values are
  0.164784, 0.165822, 0.261634, 0.173927, 0.126985, and 0.277022 at
  15,004 / 15,025 / 15,050 / 15,075 / 15,100 / 15,103. Total loss at the same
  nodes is 0.120062, 0.110359, 0.106250, 0.081835, 0.064169, and 0.119105.
- Fixed debug-view mask support is 0.156733. Quantized debug-map ks inside the
  fixed-view mask rises from the original 15,003 mean 0.095482 to
  0.096744 / 0.097544 / 0.098727 / 0.100213 / 0.100522 at
  15,025 / 15,050 / 15,075 / 15,100 / 15,103. The corresponding fixed-view
  outside-mask mean changes only from 0.087836 to
  0.089029 / 0.088253 / 0.088522 / 0.088176 / 0.088187. This distinguishes the
  previously verified L_spec-only zero outside gradient from normal total-loss
  drift outside the mask.
- Fixed-view candidate p50/p95/p99 is `126/227/285`, `134/243/310`,
  `141/256/329`, `147/266/343`, and `147/266/344`; exact intersections are
  `33/72/86`, `35/77/96`, `36/80/103`, `38/83/109`, and `38/84/109`. This is
  gradual growth, not an order-of-magnitude traversal failure. Reflection
  contribution remains finite and almost fully nonzero; mean/p99 rises from
  `1.087e-4/5.623e-4` to `1.923e-4/1.171e-3`.
- Fixed-view raytrace wall time rises from 216.5 ms at 15,025 to 252.1 ms at
  15,103, with traversal 73.3--85.0 ms and intersection/composite
  140.6--164.8 ms. All raw debug non-finite counts are zero.
- Final checkpoint
  `output/stage_b_tihubird_reflection_dr_c03_lspec_pilot100_g15003_15103_v1/chkpnt15103.pth`
  has SHA-256
  `c5e40e1a9025a3e191314759e8214e2eb11cba9e04ee2319c6f0c522e4cd6bca`.
  It is `rtgs_stage_b` at global 15,103 / R-local 103, records
  `lambda_spec=0.2` and the reviewed-v1 manifest hashes, and recursive
  inspection covered 63 tensors / 19,912,887 elements with no NaN/Inf.
- Evidence caveat: this non-smoke training path did not persist
  `torch.cuda.max_memory_allocated()` or `max_memory_reserved()` for ordinary
  steps before the process exited. Debug raytrace-local peaks are recorded, but
  they are not whole-step allocator peaks and must not be reported as such.
  Therefore classify this as
  `L_SPEC_PILOT_COMPLETED_AWAITING_USER_EVALUATION_WITH_MEMORY_TELEMETRY_GAP`,
  not Stage B acceptance or long-training authorization.
