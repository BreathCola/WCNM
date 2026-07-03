# Stage C — Transparent Mesh and Two-Hit Geometry

## Scope and fixed semantics

Stage C reads the accepted Branch-A Diffuse state, audits its glass geometry,
extracts one fixed mesh, and precomputes camera-ray front/back intersections.
It does not create or train a Transmittance Gaussian and does not implement a
second bounce.

Mask domains are fixed:

- Reflection rays continue to use every valid Diffuse surface; Stage C does not
  mask-gate them.
- `mask_hard` is the only camera-ray domain used to build two-hit caches.
- `mask_eroded` is used for Diffuse depth/mesh validation and TSDF fusion.
- `mask_soft` remains the Stage-B `L_spec` domain.
- RGB reconstruction remains full-frame.

## Accepted source and audit

The sole input is Branch A global 15,000:

`output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth`

SHA-256:
`050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84`.

The source-resolution, common-depth-scale audit is at
`output/stage_c_tihubird_c03r8_g15000_mesh_audit_ks090_v2/`. It rendered only
the checkpoint's Diffuse namespace and selected the audited high-specularity D
candidate (`ks >= 0.9`) to avoid feeding the entire scene into the transparent
mesh extractor. All 111 views passed finite alpha/depth/unit-normal coverage;
the median/ minimum eroded-mask valid coverage is 1.0 / 0.96384. Multi-view
voxel support >=2 covers 0.44312 of occupied voxels and the largest supported
component covers 0.83541. The recorded verdict is
`STAGE_C_MESH_AUDIT_PASS`.

This is an extractability verdict, not proof that every selected D surfel is
glass or that Reflection separation is complete. The pages visibly retain
some bird/background adhesion, so downstream topology and two-hit coverage are
mandatory acceptance evidence.

## Implemented preprocessing

- `geometry/stage_c_audit.py` and `tools/audit_stage_c_mesh.py`: finite
  alpha/depth/normal audit, formal mask variants, cross-view voxel support,
  source-resolution pages, raw physical arrays, and common display scales.
- `geometry/tsdf_fusion.py` and `tools/build_stage_c_mesh.py`: deterministic CPU
  multi-view TSDF fusion, component filtering, optional observed-occupancy outer
  shell cleanup, PLY export, and topology statistics.
- `geometry/csrc/mesh_bvh.cpp`, `geometry/mesh_intersector.py`, and
  `geometry/two_hit.py`: CPU BVH intersection, mask-hard camera rays,
  first/last positive mesh hits, back positions, and strict version/hash/load
  validation.
- `tools/build_stage_c_two_hit_cache.py`: 111 immutable NPZ caches plus shared-
  scale near/far/valid debug maps and a back-surface point cloud.
- `geometry/dr_cuboid.py` and `tools/build_stage_c_dr_cuboid_mesh.py`: decode
  the audited C03 normal convention, robustly calibrate each raw DR depth image
  against full-D metric depth outside the glass mask, fit three orthogonal glass
  axes, initialize metric bounds from the retained TSDF result, and optimize six
  enclosure planes against all formal masks.

The first retained TSDF mesh is
`output/stage_c_tihubird_c03r8_g15000_mesh_v3/glass_mesh.ply`, SHA-256
`86fe0d85c29cb789399a3ae5accddb2359e876fed9054211ea409d89dec208e2`.
It is one component with 118,003 vertices, 236,470 faces, zero boundary edges,
zero non-manifold edges, and is topologically watertight. Cleanup removes small
negative-TSDF components and fills enclosed occupancy only; it does not fit a
box or other primitive.

Its diagnostic caches and required debug products are at
`output/stage_c_tihubird_c03r8_g15000_two_hit_v2/`. All 111 entries pass strict
schema/hash/finite/depth-order reload. Every valid ray has `t_far > t_near`.

## Acceptance state

The pure-TSDF candidate is rejected for downstream use:

- mean hard-mask two-hit validity: 0.80313; minimum view: 0.67551;
- mean eroded-mask two-hit validity: 0.81843;
- mean fraction of valid rays with more than two mesh crossings: 0.44044;
- the fixed-view maps contain large structured missing regions and the far map
  retains internal/folded structure.

The independent DR-guided repair is retained at
`output/stage_c_tihubird_c03r8_g15000_mesh_dr_cuboid_v2/`. Its
`glass_mesh.ply` SHA-256 is
`f6846cc0d2e4ab87a4417f138bf674d8429c73e207103645597891b6fbab53e7`.
It is an automatic six-plane fit, not a hand-authored box: DR normals determine the orthogonal
directions, D/TSDF supplies metric initialization, source-resolution masks fit
the plane offsets, and calibrated DR depth supplies a separate front-depth
check. Raw DR depth is not treated as metric without calibration. Its per-view
calibration R² is 0.38935 minimum / 0.73163 median; its median per-view front
depth relative residual has a 0.12280 median across views.

The repaired mesh is watertight with 8 vertices and 12 triangles. Across all
111 source-resolution views, projected-mask recall is 0.97200 mean / 0.91240
minimum and IoU is 0.95128 mean / 0.88288 minimum. The worst IoU view `000053`
contains a formal-mask background protrusion that the cuboid deliberately does
not absorb; the worst-recall view is `000063`.

The accepted versioned caches are at
`output/stage_c_tihubird_c03r8_g15000_two_hit_v4/`:

- hard-mask two-hit validity: 0.97038 mean / 0.90821 minimum;
- eroded-mask two-hit validity: 0.99117 mean;
- every valid ray satisfies `t_far > t_near`;
- more-than-two-crossing fraction: exactly 0;
- all 111 entries pass schema, source/mesh hash, finite, shape, hit-count, and
  depth-order reload validation.

This passes the Stage C geometry gate under the explicit scene-specific
assumption that the TiHuBird glass enclosure is a six-plane cuboid. It does not
prove that DR depth is metrically calibrated by the model itself, nor authorize
using this method unchanged on curved glass.

Stage D now has its geometric prerequisite, but has not been entered. No T
model, T training, second bounce, or Stage D work is authorized by this commit.

## Reproduction commands

```bash
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n RT-GS \
  python tools/build_stage_c_mesh.py \
  --audit output/stage_c_tihubird_c03r8_g15000_mesh_audit_ks090_v2 \
  --output output/stage_c_tihubird_c03r8_g15000_mesh_v3 \
  --max-resolution 160 --minimum-weight 2 --margin-fraction 0.15 \
  --cleanup outer-shell --closing-iterations 2

CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n RT-GS \
  python tools/build_stage_c_two_hit_cache.py \
  --mesh-output output/stage_c_tihubird_c03r8_g15000_mesh_v3 \
  --audit output/stage_c_tihubird_c03r8_g15000_mesh_audit_ks090_v2 \
  --output output/stage_c_tihubird_c03r8_g15000_two_hit_v2
```

These commands fail closed rather than overwrite an existing output directory.

The accepted repair additionally uses:

```bash
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n RT-GS \
  python tools/build_stage_c_dr_cuboid_mesh.py \
  --full-d-audit output/stage_c_tihubird_c03r8_g15000_mesh_audit_v1 \
  --dr-raw output/stage_a_tihubird_dr_raw_111 \
  --mask-manifest data/TiHuBird/specular_masks_reviewed_v1/manifest.json \
  --source-mesh output/stage_c_tihubird_c03r8_g15000_mesh_v3/glass_mesh.ply \
  --output output/stage_c_tihubird_c03r8_g15000_mesh_dr_cuboid_v2

CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n RT-GS \
  python tools/build_stage_c_two_hit_cache.py \
  --mesh-output output/stage_c_tihubird_c03r8_g15000_mesh_dr_cuboid_v2 \
  --audit output/stage_c_tihubird_c03r8_g15000_mesh_audit_ks090_v2 \
  --output output/stage_c_tihubird_c03r8_g15000_two_hit_v4
```
