from types import SimpleNamespace

import pytest
import torch

from stage_d_training import (
    CACHED_NODES, CACHED_OUTPUT_NAME, FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256, _phase_for_iteration, _validate_cached_contract,
)
from tools.audit_stage_d_cached_twarmup import expected_r_local
from tools.run_stage_d_cached_twarmup_then_joint import training_command
from utils.stage_d_static_cache import (
    StaticDRCache, cpu_cache_payload, frozen_branch_hash, make_identity,
    renderer_contract, state_sha256, write_manifest,
)
from geometry.geometry_release import sha256_file
from utils.transmittance_debug import _mask_hwc


def dataset_config():
    return SimpleNamespace(
        resolution=8, ray_chunk_size=2048, ray_cutoff_sigma=3.0,
        ray_hit_threshold=1e-4, ray_epsilon_scale=1e-4,
        material_alpha_threshold=1e-4, roughness_min=0.03,
        roughness_remap=False, ray_background="scene",
        transmittance_compose="alpha_over",
    )


def test_renderer_identity_binds_all_required_inputs():
    contract = renderer_contract(dataset_config())
    first = make_identity("a" * 64, "b" * 64, "c" * 64, contract)
    second = make_identity("a" * 64, "b" * 64, "d" * 64, contract)
    assert first["identity_sha256"] != second["identity_sha256"]
    changed = dict(contract, ray_chunk_size=1024)
    assert make_identity("a" * 64, "b" * 64, "c" * 64, changed) \
        != first


def test_debug_mask_normalizes_chw_and_hwc_without_broadcasting():
    chw = torch.zeros((1, 269, 478))
    hwc = _mask_hwc(chw)
    assert hwc.shape == (269, 478, 1)
    assert _mask_hwc(hwc).shape == (269, 478, 1)
    with pytest.raises(ValueError, match="one-channel"):
        _mask_hwc(torch.zeros((3, 269, 478)))


def test_cache_rejects_target_rgb():
    identity = make_identity("a" * 64, "b" * 64, "c" * 64, {})
    with pytest.raises(ValueError, match="target data"):
        cpu_cache_payload(
            {"package": {"target_rgb": torch.zeros(1)}}, identity,
            "000000", "d" * 64,
        )


def test_cache_manifest_and_payload_fail_closed(tmp_path):
    identity = make_identity("a" * 64, "b" * 64, "c" * 64, {"x": 1})
    camera_ids = {"000000": "d" * 64, "000001": "e" * 64}
    entries = []
    for stem in camera_ids:
        payload = cpu_cache_payload(
            {"package": {"alpha": torch.ones(1, dtype=torch.float32)},
             "indices": torch.tensor([0])},
            identity, stem, camera_ids[stem],
        )
        relative = f"views/{stem}.pth"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        torch.save(payload, path)
        entries.append({
            "camera_stem": stem, "camera_identity_sha256": camera_ids[stem],
            "relative_path": relative, "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    write_manifest(tmp_path, identity, entries)
    cache = StaticDRCache(tmp_path, identity, camera_ids)
    loaded = cache.load("000000", device="cpu")
    assert loaded["package"]["alpha"].dtype == torch.float32
    path = tmp_path / "views/000000.pth"
    path.touch()
    with pytest.raises(ValueError, match="mutated"):
        cache.load("000000", device="cpu")
    with pytest.raises(ValueError, match="identity mismatch"):
        StaticDRCache(tmp_path, dict(identity, source_checkpoint_sha256="f" * 64), camera_ids)


class _FakeModel:
    def __init__(self):
        self.tensor = torch.tensor([1.0])

    def capture(self):
        return {
            "model": self.tensor, "optimizer": {"state": {}},
            "topology_version": 0, "densification": torch.zeros(1),
        }


def test_frozen_state_hash_covers_model_and_bookkeeping():
    model = _FakeModel()
    before = frozen_branch_hash(model)
    model.tensor.add_(1)
    assert frozen_branch_hash(model) != before
    assert state_sha256({"x": torch.tensor([1.0])}) \
        == state_sha256({"x": torch.tensor([1.0])})


def test_fixed_phase_and_r_local_semantics():
    assert _phase_for_iteration(15001) == "cached_t_warmup"
    assert _phase_for_iteration(18000) == "cached_t_warmup"
    assert _phase_for_iteration(18001) == "exact_joint"
    assert expected_r_local(15025) == 12000
    assert expected_r_local(18000) == 12000
    assert expected_r_local(19000) == 13000
    assert expected_r_local(20000) == 14000


def test_operator_is_one_exact_two_phase_command():
    command = training_command()
    assert "--stage_d_cached_twarmup" in command
    assert command[command.index("--stage_d_phase_a_end_iteration") + 1] == "18000"
    assert command[command.index("--iterations") + 1] == "20000"
    assert command[command.index("--stage_d_depth_start_iteration") + 1] == "40000"
    checkpoint_index = command.index("--checkpoint_iterations")
    assert tuple(map(int, command[checkpoint_index + 1:])) == CACHED_NODES
    assert "--stage_d_smoke" not in command


def test_cached_contract_rejects_any_schedule_drift(tmp_path):
    mask_file_hash = "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
    mask_aggregate = "5" * 64
    dataset = SimpleNamespace(
        resolution=8, ray_chunk_size=2048,
        transmittance_init_mode="random_bbox", transmittance_init_count=4096,
        transmittance_init_seed=20260703,
        model_path=str(tmp_path / CACHED_OUTPUT_NAME),
        _validated_specular_mask_manifest={
            "manifest_file_sha256": mask_file_hash,
            "aggregate_sha256": mask_aggregate,
        },
    )
    opt = SimpleNamespace(
        stage_d_cached_twarmup=True, iterations=20000,
        stage_d_phase_a_end_iteration=18000,
        stage_d_depth_start_iteration=40000, lambda_depth=0.2,
        lambda_spec=0.2, specular_k0=0.9,
    )
    release = SimpleNamespace(
        manifest={"geometry_release_id": FORMAL_RELEASE_ID},
        validation={"aggregate_sha256": FORMAL_RELEASE_SHA256},
    )
    source = {
        "sha256": FORMAL_SOURCE_SHA256, "global_iteration": 15000,
        "reflection_iteration": 12000,
        "stage_b_config": {
            "lambda_spec": 0.2, "specular_k0": 0.9,
            "specular_mask": {
                "manifest_file_sha256": mask_file_hash,
                "aggregate_sha256": mask_aggregate,
            },
        },
    }
    _validate_cached_contract(
        dataset, opt, release, source, True, CACHED_NODES, CACHED_NODES,
    )
    opt.stage_d_phase_a_end_iteration = 17999
    with pytest.raises(ValueError, match="phase_a_end"):
        _validate_cached_contract(
            dataset, opt, release, source, True, CACHED_NODES, CACHED_NODES,
        )
