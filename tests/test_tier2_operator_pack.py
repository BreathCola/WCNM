import ast
import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_a_state import make_full_stage_a_checkpoint, validate_full_stage_a_checkpoint
from scene.stage_b_state import (
    initialize_stage_b_from_diffuse,
    make_stage_b_checkpoint,
    restore_stage_b_checkpoint,
    sha256_file,
)
from stage_b_training import _validate_operator_restored_state
from tests.test_reflection_surfel_model import reflection_args
from tools.audit_tier2_gate import build_gate_packet
from tools.tier2_oneshot import (
    ALLOCATOR_POLICY,
    EXPERIMENT,
    RESOLUTION,
    RETRY_IDENTITY,
    audit_logs,
    branch_checkpoint_iterations,
    safe_build_final_audit,
    validate_live_telemetry,
)
from utils.d_bootstrap_telemetry import (
    DBootstrapTelemetryError,
    DBootstrapTelemetryWriter,
    REQUIRED_FIELDS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    validate_d_telemetry_record,
)
from utils.training_state import (
    capture_rng_state,
    make_camera_runtime_state,
    restore_camera_deck,
    rng_states_equal,
    should_step_optimizer,
    validate_tier2_experiment_identity,
)


def diffuse_args():
    return SimpleNamespace(
        percent_dense=0.01, position_lr_init=1e-4, position_lr_final=1e-6,
        position_lr_delay_mult=0.01, position_lr_max_steps=100,
        feature_lr=1e-3, material_lr=2e-3, opacity_lr=1e-2,
        scaling_lr=1e-3, rotation_lr=1e-3, exposure_lr_init=1e-2,
        exposure_lr_final=1e-3, exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0, iterations=100,
    )


def initialized_diffuse():
    model = DiffuseSurfelModel()
    model.spatial_lr_scale = 1.0
    tensors = {
        "_xyz": torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]]),
        "_rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        "_scaling": torch.zeros((3, 2)), "_opacity": torch.zeros((3, 1)),
        "_base_color": torch.zeros((3, 3)), "_roughness": torch.zeros((3, 1)),
        "_f0": torch.zeros((3, 3)), "_ks": torch.zeros((3, 1)),
    }
    for name, value in tensors.items():
        setattr(model, name, torch.nn.Parameter(value.clone()))
    model._exposure = torch.nn.Parameter(torch.eye(3, 4)[None])
    model.exposure_mapping = {"toy": 0}
    model.max_radii2D = torch.zeros(3)
    model.training_setup(diffuse_args())
    return model


@pytest.mark.parametrize("iteration", [3000, 7000])
def test_full_stage_a_checkpoint_fresh_r_preserves_rng_runtime_and_source(tmp_path, iteration):
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    source = initialized_diffuse()
    runtime = make_camera_runtime_state([2, 0], 3)
    checkpoint = make_full_stage_a_checkpoint(
        source, iteration, {"operator_gate_continuation": True}, runtime, True
    )
    validate_full_stage_a_checkpoint(checkpoint)
    assert "reflection" not in checkpoint and "reflection_optimizer" not in checkpoint
    path = tmp_path / f"chkpnt{iteration}.pth"
    torch.save(checkpoint, path)
    source_hash = sha256_file(path)
    expected_rng = checkpoint["rng_state"]

    random.random()
    np.random.rand()
    torch.rand(4)
    diffuse = DiffuseSurfelModel()
    reflection = ReflectionSurfelModel()
    values = initialize_stage_b_from_diffuse(
        path, diffuse, reflection, diffuse_args(), reflection_args(),
        reflection_count=8, reflection_seed=19, map_location="cpu",
        require_full_state=True, return_runtime_state=True, verify_rng_unchanged=True,
    )
    assert values[0:2] == (iteration, 0)
    assert values[3] == runtime
    assert reflection.get_xyz.shape == (8, 3)
    assert rng_states_equal(capture_rng_state(), expected_rng)
    assert sha256_file(path) == source_hash


def test_camera_deck_round_trip_preserves_remaining_order():
    cameras = ["a", "b", "c", "d"]
    runtime = make_camera_runtime_state([3, 1], len(cameras))
    deck, indices = restore_camera_deck(cameras, runtime)
    assert deck == ["d", "b"]
    assert indices == [3, 1]


def test_operator_gate_only_changes_endpoint_optimizer_continuation():
    assert should_step_optimizer(99, 100, False) is True
    assert should_step_optimizer(100, 100, False) is False
    assert should_step_optimizer(100, 100, True) is True
    validate_tier2_experiment_identity("C03-r8 Tier 2 onset study", 8)
    with pytest.raises(ValueError, match="experiment"):
        validate_tier2_experiment_identity("C03", 8)
    with pytest.raises(ValueError, match="resolution=8"):
        validate_tier2_experiment_identity("C03-r8 Tier 2 onset study", 2)


def test_full_stage_b_checkpoint_round_trip_carries_runtime_and_completed_update():
    diffuse = initialized_diffuse()
    reflection = ReflectionSurfelModel()
    reflection.create_random_bbox(torch.zeros(3), torch.ones(3), 4, 0)
    reflection.training_setup(reflection_args())
    runtime = make_camera_runtime_state([1], 3)
    checkpoint = make_stage_b_checkpoint(
        diffuse, reflection, 3205, 205, {}, {"operator_gate_continuation": True},
        runtime_state=runtime, optimizer_step_completed=True,
    )
    assert checkpoint["checkpoint_version"] == 2
    restored_d, restored_r = DiffuseSurfelModel(), ReflectionSurfelModel()
    values = restore_stage_b_checkpoint(
        checkpoint, restored_d, restored_r, diffuse_args(), reflection_args(),
        restore_rng=False, require_full_state=True, return_runtime_state=True,
    )
    assert values[0:2] == (3205, 205)
    assert values[4] == runtime
    assert checkpoint["optimizer_step_completed"] is True


def test_operator_phase_policy_is_fail_closed_at_r_local_100():
    phase_a = SimpleNamespace(
        operator_gate_continuation=True, operator_gate_expected_global_start=3000,
        operator_gate_expected_r_local_start=0, lambda_spec=0.0,
    )
    _validate_operator_restored_state(phase_a, 3000, 0)
    phase_b = SimpleNamespace(
        operator_gate_continuation=True, operator_gate_expected_global_start=3100,
        operator_gate_expected_r_local_start=100, lambda_spec=0.2,
    )
    _validate_operator_restored_state(phase_b, 3100, 100)
    phase_b.lambda_spec = 0.0
    with pytest.raises(ValueError, match=r"101\+"):
        _validate_operator_restored_state(phase_b, 3100, 100)


def _d_record(step=1):
    return {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "telemetry_step": step, "global_iteration": step, "camera_stem": "000001",
        "phase_tag": "shared_bootstrap", "total_loss": 0.2, "d_count": 3,
        "d_count_delta": 0, "d_topology_event": False, "whole_step_wall_ms": 10.0,
        "whole_step_wall_definition": "CPU perf_counter; no added CUDA synchronize",
        "cuda_memory_allocated_bytes": 10, "cuda_memory_reserved_bytes": 20,
        "cuda_max_memory_allocated_bytes": 30, "cuda_max_memory_reserved_bytes": 40,
        "cuda_peak_stats_reset_this_step": True, "cuda_peak_scope": "one loop",
        "nonfinite_count": 0, "unavailable_fields": [],
    }


def test_d_bootstrap_telemetry_is_strict_bounded_and_continuous(tmp_path):
    record = _d_record()
    assert set(record) == set(REQUIRED_FIELDS)
    validate_d_telemetry_record(record)
    writer = DBootstrapTelemetryWriter(tmp_path / "bootstrap.jsonl", 1, "shared_bootstrap")
    writer.validate_planned_steps(1)
    writer.append(record)
    with pytest.raises(RuntimeError, match="max_steps"):
        writer.append(_d_record(2))
    broken = dict(record)
    broken.pop("d_count")
    with pytest.raises(DBootstrapTelemetryError, match="missing"):
        validate_d_telemetry_record(broken)


def _stage_b_gate_record(global_step, local_step):
    return {
        "global_iteration": global_step, "reflection_local_iteration": local_step,
        "total_loss": 0.1, "l_spec": 0.2, "mask_support_fraction": 0.25,
        "ks_inside": {"mean": 0.11}, "ks_outside": {"mean": 0.09},
        "d_count": 3, "d_count_delta": 0, "d_topology_event": False,
        "r_count": 4, "r_count_delta": 0, "r_topology_event": False,
        "r_topology_version": 1, "whole_step_wall_ms": 12.0,
        "cuda_memory_allocated_bytes": 10, "cuda_memory_reserved_bytes": 20,
        "cuda_max_memory_allocated_bytes": 30, "cuda_max_memory_reserved_bytes": 40,
        "nonfinite_count": 0,
    }


def test_gate_audit_is_cpu_read_only_and_emits_complete_packet(tmp_path):
    run = tmp_path / "run"
    debug = run / "debug" / "iteration_003205"
    debug.mkdir(parents=True)
    for name in STAGE_B_FILES:
        (debug / name).write_bytes(b"x")
    for branch in ("diffuse", "reflection"):
        ply = run / "point_cloud" / branch / "iteration_3205" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"ply")
    metadata = {
        "raw_stats": {
            "ray_candidate_count": {"p50": 1, "p95": 2, "p99": 3, "max": 4},
            "ray_exact_intersection_count": {"p50": 1, "p95": 2, "p99": 3, "max": 4},
            "reflection_contribution": {
                "mean": 0.1, "p50": 0.1, "p95": 0.2, "p99": 0.3,
                "max": 0.4, "nonzero_fraction": 1.0, "nonfinite_count": 0,
            },
        },
        "raytrace": {
            "timing_ms": {"raytrace_wall": 2.0}, "peak_memory_allocated_bytes": 99,
        },
    }
    (debug / "reflection_metadata.json").write_text(json.dumps(metadata))
    diffuse, reflection = initialized_diffuse(), ReflectionSurfelModel()
    reflection.create_random_bbox(torch.zeros(3), torch.ones(3), 4, 0)
    reflection.training_setup(reflection_args())
    checkpoint = make_stage_b_checkpoint(
        diffuse, reflection, 3205, 205, {}, {},
        runtime_state=make_camera_runtime_state([0], 1), optimizer_step_completed=True,
    )
    checkpoint_path = run / "chkpnt3205.pth"
    torch.save(checkpoint, checkpoint_path)
    telemetry = run / "gate.jsonl"
    telemetry.write_text("\n".join(json.dumps(_stage_b_gate_record(g, r)) for g, r in [(3204, 204), (3205, 205)]) + "\n")
    log = run / "gate.log"
    log.write_text("Training complete.\n")
    before = {path.relative_to(run): hashlib.sha256(path.read_bytes()).hexdigest() for path in run.rglob("*") if path.is_file()}
    args = SimpleNamespace(
        checkpoint=str(checkpoint_path), telemetry=str(telemetry), run_dir=str(run),
        log=str(log), expected_global_start=3204, expected_global_end=3205,
        expected_r_local_start=204, expected_r_local_end=205, allow_running_log=False,
        require_ply=False,
    )
    packet = build_gate_packet(args)
    after = {path.relative_to(run): hashlib.sha256(path.read_bytes()).hexdigest() for path in run.rglob("*") if path.is_file()}
    assert packet["packet_ready_for_codex_review"] is True
    assert packet["checkpoint"]["finite_scan"]["nonfinite_count"] == 0
    assert packet["telemetry"]["global_continuous"] is True
    assert packet["debug"]["reflection_metadata"]["candidate"]["p99"] == 3
    assert before == after

    source = Path(__file__).parents[1] / "tools" / "audit_tier2_gate.py"
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"train", "gaussian_renderer", "raytracer"} & imports)


STAGE_B_FILES = {
    "ground_truth.png", "final.png", "diffuse_color.png", "normal.png", "ks.png",
    "reflection_color.png", "reflection_alpha.png", "reflection_depth.png",
    "reflection_hit_mask.png", "reflection_contribution.png",
    "reflection_contribution_vis.png", "ray_candidate_count.png",
    "ray_exact_intersection_count.png", "transparent_mask.png", "overlay.png",
}


def test_operator_shell_is_syntax_valid_print_first_and_finite():
    script = Path(__file__).parents[1] / "tools" / "tier2_operator.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text()
    assert 'MODE="${2:-print}"' in source
    assert "RTGS_TIER2_ACK_RESOLUTION8" in source
    assert 'EXPERIMENT="C03-r8 Tier 2 onset study"' in source
    assert 'RESOLUTION=8' in source
    assert "tier2_c03_r8_oneshot_shared_d_bootstrap" in source
    assert "tier2_c03_r8_oneshot_v4_rstart" in source
    assert 'RETRY_IDENTITY="oneshot_v4_memory_bounded_retry"' in source
    assert 'ALLOCATOR_POLICY="adaptive_pressure_cache_and_ray_retry_v1"' in source
    assert 'PYTORCH_ALLOCATOR_CONFIG="max_split_size_mb:128,garbage_collection_threshold:0.8"' in source
    assert "--ray_chunk_size 2048" in source
    assert "--stage_b_memory_retry_min_chunk_size" in source
    assert "PRESSURE_RELEASE_FREE_BYTES=2147483648" in source
    assert "MEMORY_RETRY_MIN_CHUNK_SIZE=512" in source
    assert "--stage_b_telemetry_max_steps" in source
    assert "--d_bootstrap_telemetry_jsonl" not in source
    assert "a-long" in source and "b-long" in source
    assert "nohup" not in source and "train.py &" not in source
    printed = subprocess.run(
        [str(script), "run-all"], check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    assert "RTGS_TIER2_ACK_RESOLUTION8=YES" in printed
    assert "run-all --execute" in printed
    coordinator = Path(__file__).parents[1] / "tools" / "tier2_oneshot.py"
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(coordinator.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"train", "gaussian_renderer", "raytracer"} & imports)


def test_oneshot_nodes_and_live_telemetry_are_fail_closed(tmp_path):
    assert branch_checkpoint_iterations(3000) == [
        3100, 3200, 3500, 4000, 5000, 6000, 7000, 8000, 9000,
        10000, 11000, 12000, 13000, 14000, 15000,
    ]
    assert branch_checkpoint_iterations(7000) == [
        7100, 7200, 7500, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
    ]
    path = tmp_path / "telemetry.jsonl"
    phase = (
        "experiment=C03-r8 Tier 2 onset study;"
        "retry=oneshot_v4_memory_bounded_retry;resolution=8;phase=a_warmup"
    )
    records = [
        {"global_iteration": 3001, "reflection_local_iteration": 1, "nonfinite_count": 0,
         "total_loss": 0.5, "phase_tag": phase},
        {"global_iteration": 3002, "reflection_local_iteration": 2, "nonfinite_count": 0,
         "total_loss": 0.4, "phase_tag": phase},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    summary = validate_live_telemetry(path, 3001, 1)
    assert summary["last_global"] == 3002 and summary["last_r_local"] == 2
    records[1]["global_iteration"] = 3003
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(RuntimeError, match="discontinuity"):
        validate_live_telemetry(path, 3001, 1)


def test_oneshot_final_audit_always_emits_fail_closed_record(monkeypatch):
    monkeypatch.setattr(
        "tools.tier2_oneshot.build_final_audit",
        lambda _state=None: (_ for _ in ()).throw(RuntimeError("synthetic reader failure")),
    )
    state = {"status": "HARD_FAILED"}
    report = safe_build_final_audit(state)
    assert report["healthy"] is False
    assert report["cpu_only"] is True
    assert report["experiment"] == EXPERIMENT
    assert report["resolution"] == RESOLUTION
    assert report["retry_identity"] == RETRY_IDENTITY
    assert report["allocator_policy"] == ALLOCATOR_POLICY
    assert report["operator_state"] == state
    assert "synthetic reader failure" in report["errors"][0]


def test_oneshot_audit_accepts_timestamped_memory_retry_json(tmp_path):
    log = tmp_path / "formal.log"
    retry = {
        "global_iteration": 7307,
        "reflection_local_iteration": 307,
        "failed_chunk_size": 2048,
        "failed_checkpoint_chunks": False,
        "next_attempt": {"chunk_size": 1024, "checkpoint_chunks": False},
    }
    log.write_text(
        "STAGE_B_MEMORY_RETRY " + json.dumps(retry) + " [02/07 18:49:48]\n"
        "Training complete.\n",
        encoding="utf-8",
    )

    result = audit_logs([log])

    assert result["errors"] == []
    assert result["memory_retries"] == [retry]


def test_oneshot_audit_rejects_unknown_retry_suffix(tmp_path):
    log = tmp_path / "formal.log"
    log.write_text(
        'STAGE_B_MEMORY_RETRY {"global_iteration": 1} trailing-garbage\n'
        "Training complete.\n",
        encoding="utf-8",
    )

    result = audit_logs([log])

    assert result["memory_retries"] == []
    assert result["errors"] == [f"malformed memory retry record: {log}"]
