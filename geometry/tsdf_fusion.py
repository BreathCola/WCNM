"""Deterministic CPU TSDF fusion and mesh cleanup for Stage C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class TSDFVolume:
    values: np.ndarray
    weights: np.ndarray
    origin: np.ndarray
    voxel_size: float


def camera_intrinsics_from_transforms(
    world_view_transform: np.ndarray,
    full_proj_transform: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_view = np.asarray(world_view_transform, dtype=np.float64)
    full_proj = np.asarray(full_proj_transform, dtype=np.float64)
    c2w = np.linalg.inv(world_view.T)
    ndc2pix = np.array(
        [[width / 2, 0, 0, width / 2], [0, height / 2, 0, height / 2], [0, 0, 0, 1]],
        dtype=np.float64,
    ).T
    projection = c2w.T @ full_proj
    intrinsics = (projection @ ndc2pix)[:3, :3].T
    return intrinsics, c2w[:3, :3], c2w[:3, 3]


def make_volume_bounds(
    robust_min: np.ndarray,
    robust_max: np.ndarray,
    max_resolution: int = 128,
    margin_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    lo = np.asarray(robust_min, dtype=np.float64)
    hi = np.asarray(robust_max, dtype=np.float64)
    extent = hi - lo
    if lo.shape != (3,) or np.any(~np.isfinite(extent)) or np.any(extent <= 0):
        raise ValueError("invalid robust TSDF bounds")
    margin = extent.max() * float(margin_fraction)
    lo = lo - margin
    hi = hi + margin
    voxel_size = float((hi - lo).max() / (int(max_resolution) - 1))
    dims = np.ceil((hi - lo) / voxel_size).astype(np.int64) + 1
    return lo.astype(np.float32), hi.astype(np.float32), voxel_size, dims


def _grid_chunk(origin: np.ndarray, voxel_size: float, dims: np.ndarray, start: int, stop: int) -> np.ndarray:
    flat = np.arange(start, stop, dtype=np.int64)
    yz = int(dims[1] * dims[2])
    ix = flat // yz
    rem = flat % yz
    iy = rem // dims[2]
    iz = rem % dims[2]
    index = np.stack((ix, iy, iz), axis=1).astype(np.float32)
    return origin[None] + index * voxel_size


def fuse_tsdf(
    views: list[dict],
    bounds_min: np.ndarray,
    dims: np.ndarray,
    voxel_size: float,
    truncation_voxels: float = 4.0,
    chunk_voxels: int = 250_000,
) -> TSDFVolume:
    dims = np.asarray(dims, dtype=np.int64)
    count = int(np.prod(dims))
    value_sum = np.zeros(count, dtype=np.float32)
    weights = np.zeros(count, dtype=np.uint16)
    truncation = float(truncation_voxels * voxel_size)
    for view in views:
        depth = np.asarray(view["depth"], dtype=np.float32)
        mask = np.asarray(view["mask_eroded"], dtype=bool)
        height, width = depth.shape
        intrinsics, rotation, camera_center = camera_intrinsics_from_transforms(
            view["world_view_transform"], view["full_proj_transform"], width, height
        )
        for start in range(0, count, chunk_voxels):
            stop = min(start + chunk_voxels, count)
            world = _grid_chunk(bounds_min, voxel_size, dims, start, stop).astype(np.float64)
            camera = (world - camera_center[None]) @ rotation
            z = camera[:, 2]
            pixels_h = camera @ intrinsics.T
            safe_z = np.where(np.abs(pixels_h[:, 2]) > 1e-12, pixels_h[:, 2], 1.0)
            u = np.rint(pixels_h[:, 0] / safe_z).astype(np.int64)
            v = np.rint(pixels_h[:, 1] / safe_z).astype(np.int64)
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            local = np.flatnonzero(inside)
            if not local.size:
                continue
            observed_depth = depth[v[local], u[local]]
            observed = mask[v[local], u[local]] & np.isfinite(observed_depth) & (observed_depth > 0)
            local = local[observed]
            if not local.size:
                continue
            observed_depth = observed_depth[observed]
            sdf = observed_depth - z[local]
            keep = sdf >= -truncation
            local = local[keep]
            if not local.size:
                continue
            tsdf = np.minimum(1.0, sdf[keep] / truncation).astype(np.float32)
            target = start + local
            value_sum[target] += tsdf
            weights[target] += 1
    values = np.ones(count, dtype=np.float32)
    known = weights > 0
    values[known] = value_sum[known] / weights[known]
    return TSDFVolume(
        values.reshape(tuple(dims)), weights.reshape(tuple(dims)),
        np.asarray(bounds_min, dtype=np.float32), float(voxel_size),
    )


def extract_mesh(volume: TSDFVolume, minimum_weight: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = volume.values.copy()
    values[volume.weights < int(minimum_weight)] = 1.0
    if values.min() > 0 or values.max() < 0:
        raise ValueError("TSDF has no zero crossing")
    vertices, faces, normals, _ = marching_cubes(
        values, level=0.0, spacing=(volume.voxel_size,) * 3,
        allow_degenerate=False, method="lewiner",
    )
    vertices = vertices + volume.origin[None]
    return vertices.astype(np.float32), faces.astype(np.int32), normals.astype(np.float32)


def extract_outer_shell_mesh(
    volume: TSDFVolume,
    minimum_weight: int = 2,
    closing_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract one filled outer occupancy shell from the observed TSDF.

    The cleanup is deliberately geometry-only: it closes voxel-scale cracks,
    retains the largest connected negative TSDF region, and fills enclosed
    cavities before meshing its boundary. It does not fit a primitive or use
    image/RGB information.
    """
    known = volume.weights >= int(minimum_weight)
    inside_raw = known & (volume.values < 0)
    if not inside_raw.any():
        raise ValueError("TSDF has no observed negative occupancy")
    structure = ndimage.generate_binary_structure(3, 1)
    inside_closed = ndimage.binary_closing(
        inside_raw, structure=structure, iterations=int(closing_iterations)
    )
    labels, component_count = ndimage.label(inside_closed, structure=structure)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    inside_component = labels == largest_label
    inside_filled = ndimage.binary_fill_holes(inside_component, structure=structure)
    if inside_filled.all() or not inside_filled.any():
        raise ValueError("outer-shell cleanup produced invalid occupancy")
    # A distance-derived level set gives marching cubes a stable zero crossing
    # and removes all internal TSDF sheets while retaining the cleaned boundary.
    outside_distance = ndimage.distance_transform_edt(~inside_filled)
    inside_distance = ndimage.distance_transform_edt(inside_filled)
    level_set = (outside_distance - inside_distance).astype(np.float32)
    vertices, faces, normals, _ = marching_cubes(
        level_set, level=0.0, spacing=(volume.voxel_size,) * 3,
        allow_degenerate=False, method="lewiner",
    )
    vertices = vertices + volume.origin[None]
    report = {
        "method": "largest_negative_tsdf_component_filled_outer_shell",
        "closing_iterations": int(closing_iterations),
        "connectivity": 1,
        "raw_inside_voxels": int(inside_raw.sum()),
        "closed_inside_voxels": int(inside_closed.sum()),
        "component_count": int(component_count),
        "largest_component_voxels_before_fill": int(inside_component.sum()),
        "filled_inside_voxels": int(inside_filled.sum()),
        "filled_cavity_voxels": int(inside_filled.sum() - inside_component.sum()),
    }
    return (
        vertices.astype(np.float32), faces.astype(np.int32),
        normals.astype(np.float32), report,
    )


def largest_face_component(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    parent = np.arange(len(faces), dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_owner = {}
    for face_index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = face_index
            else:
                union(face_index, previous)
    roots = np.array([find(i) for i in range(len(faces))])
    labels, counts = np.unique(roots, return_counts=True)
    selected_label = labels[np.argmax(counts)]
    selected_faces = faces[roots == selected_label]
    used = np.unique(selected_faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    selected_faces = remap[selected_faces]
    fraction = float(len(selected_faces) / max(len(faces), 1))
    return vertices[used], selected_faces, {
        "component_count": int(len(labels)),
        "largest_face_fraction": fraction,
        "discarded_face_count": int(len(faces) - len(selected_faces)),
    }


def mesh_topology(vertices: np.ndarray, faces: np.ndarray) -> dict:
    edge_counts = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = sum(count == 1 for count in edge_counts.values())
    nonmanifold = sum(count > 2 for count in edge_counts.values())
    return {
        "vertex_count": int(len(vertices)), "face_count": int(len(faces)),
        "edge_count": int(len(edge_counts)), "boundary_edge_count": int(boundary),
        "nonmanifold_edge_count": int(nonmanifold),
        "watertight": boundary == 0 and nonmanifold == 0,
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
    }


def write_mesh_ply(path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertex = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = vertices.T
    face = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = faces
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")], text=False).write(path)
