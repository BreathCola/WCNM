import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from stage_b_training import _validate_stage_b_args, training_stage_b
from tools.compare_stage_b_pilots import compare_stage_b_pilots
from utils.stage_b_telemetry import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    StageBTelemetryWriter,
    TelemetrySchemaError,
    cuda_allocator_snapshot,
    mask_ks_stats_from_existing_tensors,
    validate_telemetry_options,
    validate_telemetry_record,
)


def _record(step=1, phase="phase_a"):
    unavailable = [
        "candidate_backward_ms", "candidate_p50", "candidate_p95", "candidate_p99",
        "exact_p50", "exact_p95", "exact_p99", "raytrace_forward_ms",
        "l_spec", "mask_support_fraction", "ks_inside", "ks_outside",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "telemetry_step": step,
        "global_iteration": 10_000 + step,
        "reflection_local_iteration": step,
        "camera_stem": "000001",
        "phase_tag": phase,
        "total_loss": 0.25,
        "l_spec": None,
        "d_count": 8,
        "r_count": 4,
        "d_count_delta": 0,
        "r_count_delta": 0,
        "d_topology_event": False,
        "r_topology_version": 0,
        "r_topology_event": False,
        "mask_support_fraction": None,
        "ks_inside": None,
        "ks_outside": None,
        "ks_state_definition": "pre_optimizer_forward_state_used_by_total_loss; no post-update rerender",
        "candidate_p50": None,
        "candidate_p95": None,
        "candidate_p99": None,
        "exact_p50": None,
        "exact_p95": None,
        "exact_p99": None,
        "raytrace_forward_ms": None,
        "candidate_backward_ms": None,
        "whole_step_wall_ms": 1.5,
        "whole_step_wall_definition": "CPU perf_counter; no added CUDA synchronize",
        "cuda_memory_allocated_bytes": 10,
        "cuda_memory_reserved_bytes": 20,
        "cuda_max_memory_allocated_bytes": 30,
        "cuda_max_memory_reserved_bytes": 40,
        "cuda_peak_stats_reset_this_step": False,
        "cuda_peak_scope": "process_start_or_preexisting_reset_before_telemetry",
        "nonfinite_count": 0,
        "nonfinite_scope": "observed existing scalar/tensor values only",
        "unavailable_fields": unavailable,
    }


def test_telemetry_defaults_off_and_never_enable_ray_diagnostics():
    assert validate_telemetry_options("", 0, "") is False
    source = inspect.getsource(training_stage_b)
    assert "return_ray_diagnostics=opt.specular_smoke_diagnostics" in source
    assert "return_ray_diagnostics=telemetry" not in source


def test_telemetry_jsonl_does_not_change_loss_update_or_rng(tmp_path):
    def run(enabled):
        torch.manual_seed(1234)
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        sample = torch.randn(2)
        loss = ((parameter - sample) ** 2).mean()
        loss_value = float(loss.item())
        loss.backward()
        optimizer.step()
        if enabled:
            writer = StageBTelemetryWriter(tmp_path / "telemetry.jsonl", 1, "phase_a")
            writer.validate_planned_steps(1)
            record = _record()
            record["total_loss"] = loss_value
            writer.append(record)
        return loss_value, parameter.detach().clone(), torch.get_rng_state().clone()

    baseline = run(False)
    observed = run(True)
    assert baseline[0] == observed[0]
    assert torch.equal(baseline[1], observed[1])
    assert torch.equal(baseline[2], observed[2])
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == 1


def test_phase_a_has_null_mask_stats_and_requires_no_manifest():
    ks = torch.full((2, 2, 1), 0.1)
    stats = mask_ks_stats_from_existing_tensors(ks, None, torch.ones_like(ks))
    assert stats == {
        "mask_support_fraction": None, "ks_inside": None,
        "ks_outside": None, "nonfinite_count": 0,
    }
    dataset = SimpleNamespace(
        model_type="surfel", stage="stage_b", ray_background="scene", specular_masks="",
        reflection_init_mode="random_bbox", reflection_init_count=4,
    )
    opt = SimpleNamespace(
        random_background=False, lambda_spec=0.0, specular_smoke_diagnostics=False,
        stage_b_telemetry_jsonl="phase_a.jsonl", stage_b_telemetry_max_steps=100,
        stage_b_telemetry_phase_tag="phase_a",
    )
    _validate_stage_b_args(dataset, opt, None, "diffuse.pth")


def test_phase_b_mask_ks_stats_are_correct_from_existing_forward_tensors():
    ks = torch.tensor([[[0.1], [0.2]], [[0.8], [0.9]]])
    mask = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])
    stats = mask_ks_stats_from_existing_tensors(ks, mask, torch.ones_like(ks))
    assert stats["mask_support_fraction"] == 0.5
    assert stats["ks_inside"]["min"] == pytest.approx(0.1)
    assert stats["ks_inside"]["mean"] == pytest.approx(0.45)
    assert stats["ks_outside"]["min"] == pytest.approx(0.2)
    assert stats["ks_outside"]["p99"] == pytest.approx(0.9, abs=0.01)
    assert stats["nonfinite_count"] == 0


def test_strict_schema_and_max_steps_fail_closed(tmp_path):
    record = _record()
    assert set(record) == set(REQUIRED_FIELDS)
    validate_telemetry_record(record)
    broken = dict(record)
    broken.pop("r_count")
    with pytest.raises(TelemetrySchemaError, match="missing"):
        validate_telemetry_record(broken)
    extra = dict(record, invented=1)
    with pytest.raises(TelemetrySchemaError, match="extra"):
        validate_telemetry_record(extra)

    writer = StageBTelemetryWriter(tmp_path / "bounded.jsonl", 1, "phase_a")
    with pytest.raises(ValueError, match="exceed"):
        writer.validate_planned_steps(2)
    writer.append(record)
    with pytest.raises(RuntimeError, match="max_steps"):
        writer.append(_record(step=2))
    with pytest.raises(ValueError, match="both"):
        validate_telemetry_options("only-path.jsonl", 0, "phase_a")


def test_allocator_schema_distinguishes_current_from_scoped_max(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    snapshot = cuda_allocator_snapshot(False, "test_scope_without_reset")
    assert snapshot["cuda_memory_allocated_bytes"] is None
    assert snapshot["cuda_max_memory_allocated_bytes"] is None
    assert snapshot["cuda_peak_stats_reset_this_step"] is False
    assert "requested_scope=test_scope_without_reset" in snapshot["cuda_peak_scope"]
    assert not any("whole_step" in key for key in snapshot)


def _write_debug(run: Path, iteration: int, offset: int):
    debug = run / "debug" / f"iteration_{iteration:06d}"
    debug.mkdir(parents=True)
    height, width = 5, 7
    ground_truth = np.full((height, width, 3), 80, dtype=np.uint8)
    ks = np.full((height, width, 3), 20 + offset, dtype=np.uint8)
    reflection = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3) + offset
    final = np.full((height, width, 3), 90 + offset, dtype=np.uint8)
    for name, values in (
        ("ground_truth.png", ground_truth), ("ks.png", ks),
        ("reflection_contribution.png", reflection), ("final.png", final),
    ):
        Image.fromarray(values).save(debug / name)
    (debug / "reflection_metadata.json").write_text(json.dumps({"iteration": iteration}))
    return debug


def test_postprocess_is_cpu_read_only_and_uses_common_scales(tmp_path):
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    debug_a, debug_b = _write_debug(run_a, 10_100, 0), _write_debug(run_b, 15_100, 3)
    before_a = {path.name: path.read_bytes() for path in debug_a.iterdir()}
    before_b = {path.name: path.read_bytes() for path in debug_b.iterdir()}
    output = tmp_path / "comparison"
    metadata = compare_stage_b_pilots(run_a, run_b, output)
    assert metadata["source_contract"]["renderer_invoked"] is False
    assert metadata["source_contract"]["cuda_invoked"] is False
    assert metadata["comparisons"]["ks"]["common_scale"] > 0
    assert metadata["comparisons"]["reflection_contribution"]["common_scale"] > 0
    assert (output / "delta_ks_b_minus_a.png").is_file()
    assert (output / "delta_reflection_contribution_b_minus_a.png").is_file()
    assert before_a == {path.name: path.read_bytes() for path in debug_a.iterdir()}
    assert before_b == {path.name: path.read_bytes() for path in debug_b.iterdir()}

    source = Path(inspect.getsourcefile(compare_stage_b_pilots)).read_text()
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"torch", "train", "raytracer", "gaussian_renderer"} & imports)
