# Stage C — Transparent Mesh and Two-Hit Geometry

## Summary

Add the transparent-mask toolchain, TSDF mesh extraction/filtering, front/back
mesh intersection, and versioned per-view two-hit caches. This stage does not
train the Transmittance field.

## Prerequisites

- Stages A and B are accepted and committed without regression.
- Diffuse geometry and depth are stable enough for mesh extraction.
- Transparent masks load with image-aligned dimensions.
- `docs/STATUS.md` explicitly advances the project to Stage C.

## Acceptance entry

Begin Stage C acceptance only when mask variants and overlays, a cleaned relevant
transparent mesh, valid `t_far > t_near` intersections, back-surface positions,
and versioned save/load caches are implemented and verified while Stages A/B
continue to pass.
