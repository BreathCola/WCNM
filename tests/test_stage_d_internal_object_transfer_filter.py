from types import SimpleNamespace

import pytest
import torch

from stage_d_training import (
    _filter_transferred_candidates_by_internal_object_masks,
    _sha256_indices,
)
from utils.internal_object_mask import INTERNAL_OBJECT_SEMANTICS_VERSION, REVIEWED_ROLE


class _Diffuse:
    def __init__(self, xyz):
        self._xyz = torch.as_tensor(xyz, dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._xyz


class _Camera:
    def __init__(self, stem, union, glass=None, ignore=None):
        self.image_name = f"{stem}.jpg"
        self.image_width = int(union.shape[1])
        self.image_height = int(union.shape[0])
        self.internal_object_masks = {"internal_object_union": union[None].float()}
        if ignore is not None:
            self.internal_object_masks["internal_ignore"] = ignore[None].float()
        self.specular_mask = (torch.ones_like(union) if glass is None else glass)[None].float()

    def project_points(self, points):
        return points[:, 0].double(), points[:, 1].double(), points[:, 2].double()


class _StaticCache:
    def __init__(self, stems, shape):
        self.entries = {stem: {} for stem in stems}
        self.shape = shape

    def load(self, stem, device="cpu"):
        if stem not in self.entries:
            raise KeyError(stem)
        return {"package": {"two_hit_valid": torch.ones((*self.shape, 1), dtype=torch.float32)}}


def _manifest(stems):
    return {
        "role": REVIEWED_ROLE,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "manifest_path": "/tmp/internal_object_manifest.json",
        "manifest_file_sha256": "manifest-file",
        "manifest_payload_sha256": "manifest-payload",
        "aggregate_sha256": "aggregate",
        "entries": {stem: {"stem": stem} for stem in stems},
    }


def _opt(min_views=2, ratio=0.6, boundary=0):
    return SimpleNamespace(
        object_mask_min_views=min_views,
        object_mask_min_support_ratio=ratio,
        object_mask_boundary_ignore_px=boundary,
    )


def test_internal_object_filter_actually_changes_selected_d_indices():
    stems = ("000000", "000001", "000002")
    h, w = 8, 8
    union_a = torch.zeros(h, w)
    union_a[2, 2] = 1
    union_a[3, 3] = 1
    union_b = torch.zeros(h, w)
    union_b[2, 2] = 1
    union_c = torch.zeros(h, w)
    cameras = [_Camera(stems[0], union_a), _Camera(stems[1], union_b), _Camera(stems[2], union_c)]
    diffuse = _Diffuse([
        [2.0, 2.0, 1.0],
        [3.0, 3.0, 1.0],
        [6.0, 6.0, 1.0],
        [20.0, 20.0, 1.0],
    ])
    selected = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    filtered, metadata = _filter_transferred_candidates_by_internal_object_masks(
        selected, diffuse, cameras, _StaticCache(stems, (h, w)), _opt(),
        _manifest(stems), count=4,
    )

    assert filtered.tolist() == [0]
    assert metadata["status"] == "active"
    assert metadata["pre_object_mask_candidate_count"] == 4
    assert metadata["post_object_mask_candidate_count"] == 1
    assert metadata["selected_transferred_count"] == 1
    assert metadata["random_fill_count"] == 3
    assert metadata["rejected_invalid_projection_count"] == 1
    assert metadata["rejected_min_views_count"] == 2
    assert metadata["pre_filter_D_indices_sha256"] == _sha256_indices(selected)
    assert metadata["selected_D_indices_sha256"] == _sha256_indices(filtered)
    assert metadata["selected_D_indices_sha256"] != metadata["pre_filter_D_indices_sha256"]


def test_internal_object_filter_fails_when_camera_mask_is_not_reviewed():
    union = torch.zeros(4, 4)
    camera = _Camera("000000", union)
    diffuse = _Diffuse([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="no reviewed mask"):
        _filter_transferred_candidates_by_internal_object_masks(
            torch.tensor([0]), diffuse, [camera], _StaticCache(("000000",), (4, 4)),
            _opt(min_views=1), _manifest(("000001",)), count=1,
        )


def test_internal_object_filter_counts_boundary_rejection():
    stem = "000000"
    union = torch.zeros(5, 5)
    union[2, 2] = 1
    camera = _Camera(stem, union)
    diffuse = _Diffuse([[2.0, 2.0, 1.0]])

    filtered, metadata = _filter_transferred_candidates_by_internal_object_masks(
        torch.tensor([0]), diffuse, [camera], _StaticCache((stem,), (5, 5)),
        _opt(min_views=1, ratio=0.5, boundary=1), _manifest((stem,)), count=1,
    )

    assert filtered.numel() == 0
    assert metadata["rejected_boundary_count"] == 1


def test_internal_object_filter_uses_v3_union_and_ignores_explicit_ignore():
    stem = "000000"
    union = torch.zeros(5, 5)
    union[1, 1] = 1  # bird
    union[2, 2] = 1  # white platform
    union[3, 3] = 1  # yellow base
    ignore = torch.zeros(5, 5)
    ignore[3, 3] = 1
    camera = _Camera(stem, union, ignore=ignore)
    diffuse = _Diffuse([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
        [3.0, 3.0, 1.0],
    ])

    filtered, metadata = _filter_transferred_candidates_by_internal_object_masks(
        torch.tensor([0, 1, 2]), diffuse, [camera], _StaticCache((stem,), (5, 5)),
        _opt(min_views=1, ratio=1.0, boundary=0), _manifest((stem,)), count=3,
    )

    assert filtered.tolist() == [0, 1]
    assert metadata["mask_role"] == "internal_object_union"
    assert metadata["internal_object_semantics_version"] == INTERNAL_OBJECT_SEMANTICS_VERSION
    assert metadata["post_object_mask_candidate_count"] == 2
    assert metadata["random_fill_count"] == 1
