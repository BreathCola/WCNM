import json
from pathlib import Path

import pytest

from utils.stage_b_memory import (
    DEFAULT_POLICY,
    V2_POLICY,
    V3_RETRY_IDENTITY,
    evaluate_headroom,
    release_allocator_cache,
    validate_stage_b_memory_policy,
    write_headroom_gate,
)


def test_allocator_policy_is_default_off_and_v2_fail_closed():
    validate_stage_b_memory_policy(DEFAULT_POLICY, "", 0, 0, False)
    with pytest.raises(ValueError, match="forbids"):
        validate_stage_b_memory_policy(DEFAULT_POLICY, V3_RETRY_IDENTITY, 1, 1, True)
    with pytest.raises(ValueError, match="operator-only"):
        validate_stage_b_memory_policy(V2_POLICY, V3_RETRY_IDENTITY, 1, 1, False)
    with pytest.raises(ValueError, match="identity"):
        validate_stage_b_memory_policy(V2_POLICY, "wrong", 1, 1, True)
    validate_stage_b_memory_policy(V2_POLICY, V3_RETRY_IDENTITY, 100, 20, True)


def test_release_policy_only_calls_empty_cache_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr("utils.stage_b_memory.torch.cuda.empty_cache", lambda: calls.append(True))
    release_allocator_cache(DEFAULT_POLICY)
    assert calls == []
    release_allocator_cache(V2_POLICY)
    assert calls == [True]


def test_headroom_gate_uses_capacity_reference_peak_and_margin(tmp_path):
    common = dict(
        experiment="C03-r8 Tier 2 onset study",
        retry_identity=V3_RETRY_IDENTITY,
        policy=V2_POLICY,
        global_iteration=3101,
        reflection_local_iteration=101,
        current_allocated_bytes=100,
        current_reserved_bytes=100,
        device_total_bytes=1000,
        current_step_peak_allocated_bytes=700,
        reference_peak_allocated_bytes=800,
        minimum_projected_headroom_bytes=100,
    )
    passed = evaluate_headroom(device_free_bytes=850, **common)
    assert passed["allocator_capacity_bytes"] == 950
    assert passed["required_peak_allocated_bytes"] == 800
    assert passed["projected_headroom_bytes"] == 150
    assert passed["passed"] is True
    failed = evaluate_headroom(device_free_bytes=750, **common)
    assert failed["projected_headroom_bytes"] == 50
    assert failed["passed"] is False

    path = tmp_path / "allocator_headroom_gate.json"
    write_headroom_gate(path, passed)
    assert json.loads(path.read_text()) == passed
    with pytest.raises(FileExistsError, match="refusing existing"):
        write_headroom_gate(path, passed)


def test_training_cleanup_precedes_cache_release_and_gate_is_local_101():
    source = (Path(__file__).parents[1] / "stage_b_training.py").read_text()
    assert source.count("telemetry_record = {") == 1
    assert source.index("record = {") < source.index("telemetry_record = {")
    assert source.index("del package, image") < source.index("release_allocator_cache(allocator_policy)")
    assert "reflection_iteration == 101" in source
    assert "and opt.lambda_spec > 0" in source
    assert "Stage B allocator headroom gate failed before formal long run" in source
