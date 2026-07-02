import numpy as np

from geometry.tsdf_fusion import (
    TSDFVolume, camera_intrinsics_from_transforms, extract_mesh, extract_outer_shell_mesh,
    largest_face_component, mesh_topology,
)


def test_camera_intrinsics_round_trip_identity_camera():
    width, height = 100, 80
    world_view = np.eye(4, dtype=np.float32)
    projection = np.eye(4, dtype=np.float32)
    intrinsics, rotation, center = camera_intrinsics_from_transforms(
        world_view, projection, width, height
    )
    assert intrinsics.shape == (3, 3)
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(center, 0)


def test_extract_mesh_is_watertight_for_closed_sphere_tsdf():
    size = 32
    grid = np.stack(np.meshgrid(*(np.arange(size),) * 3, indexing="ij"), axis=-1)
    values = np.linalg.norm(grid - (size - 1) / 2, axis=-1) - 9
    volume = TSDFVolume(values.astype(np.float32), np.ones(values.shape, np.uint16), np.zeros(3), 1.0)
    vertices, faces, _ = extract_mesh(volume, minimum_weight=1)
    vertices, faces, components = largest_face_component(vertices, faces)
    topology = mesh_topology(vertices, faces)
    assert components["largest_face_fraction"] == 1.0
    assert topology["watertight"]
    assert topology["boundary_edge_count"] == 0


def test_outer_shell_cleanup_removes_enclosed_internal_sheet():
    size = 40
    grid = np.stack(np.meshgrid(*(np.arange(size),) * 3, indexing="ij"), axis=-1)
    radius = np.linalg.norm(grid - (size - 1) / 2, axis=-1)
    values = radius - 12
    # An artificial positive pocket creates an internal TSDF surface.
    values[(radius < 5)] = 1
    volume = TSDFVolume(
        values.astype(np.float32), np.ones(values.shape, np.uint16), np.zeros(3), 1.0
    )
    vertices, faces, _, cleanup = extract_outer_shell_mesh(
        volume, minimum_weight=1, closing_iterations=1
    )
    vertices, faces, components = largest_face_component(vertices, faces)
    topology = mesh_topology(vertices, faces)
    assert cleanup["filled_cavity_voxels"] > 0
    assert components["largest_face_fraction"] == 1.0
    assert topology["watertight"]
