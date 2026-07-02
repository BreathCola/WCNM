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

The retained fixed mesh is
`output/stage_c_tihubird_c03r8_g15000_mesh_v3/glass_mesh.ply`, SHA-256
`86fe0d85c29cb789399a3ae5accddb2359e876fed9054211ea409d89dec208e2`.
It is one component with 118,003 vertices, 236,470 faces, zero boundary edges,
zero non-manifold edges, and is topologically watertight. Cleanup removes small
negative-TSDF components and fills enclosed occupancy only; it does not fit a
box or other primitive.

The versioned caches and required debug products are at
`output/stage_c_tihubird_c03r8_g15000_two_hit_v2/`. All 111 entries pass strict
schema/hash/finite/depth-order reload. Every valid ray has `t_far > t_near`.

## Acceptance state

Implementation and artifact generation are complete, but Stage C acceptance is
blocked on geometry quality:

- mean hard-mask two-hit validity: 0.80313; minimum view: 0.67551;
- mean eroded-mask two-hit validity: 0.81843;
- mean fraction of valid rays with more than two mesh crossings: 0.44044;
- the fixed-view maps contain large structured missing regions and the far map
  retains internal/folded structure.

The first and last intersections remain well-defined and are cached, but this
evidence is not yet strong enough to claim a reliable glass front/back shell
for Stage D. More aggressive smoothing or a hand-fitted cuboid was deliberately
not used because it would replace measured D geometry with an unvalidated shape.

Therefore `STATUS.md` remains at Stage C. No T model, T training, second bounce,
or Stage D work is authorized.

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
