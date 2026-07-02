import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stage_b_training import _forward_backward_with_memory_retry
from utils.stage_b_memory import (
    DEFAULT_POLICY,
    V4_POLICY,
    V4_RETRY_IDENTITY,
    evaluate_headroom,
    ray_retry_attempts,
    release_allocator_cache,
    validate_stage_b_memory_policy,
    write_headroom_gate,
)


def test_allocator_policy_is_default_off_and_v4_fail_closed():
    validate_stage_b_memory_policy(DEFAULT_POLICY, "", 0, 0, 0, 2048, 0, False)
    with pytest.raises(ValueError, match="forbids"):
        validate_stage_b_memory_policy(DEFAULT_POLICY, V4_RETRY_IDENTITY, 1, 1, 1, 2048, 512, True)
    with pytest.raises(ValueError, match="operator-only"):
        validate_stage_b_memory_policy(V4_POLICY, V4_RETRY_IDENTITY, 1, 1, 1, 2048, 512, False)
    with pytest.raises(ValueError, match="identity"):
        validate_stage_b_memory_policy(V4_POLICY, "wrong", 1, 1, 1, 2048, 512, True)
    validate_stage_b_memory_policy(
        V4_POLICY, V4_RETRY_IDENTITY, 0, 20, 100, 2048, 512, True
    )


def test_release_policy_only_calls_empty_cache_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr("utils.stage_b_memory.torch.cuda.empty_cache", lambda: calls.append(True))
    release_allocator_cache(DEFAULT_POLICY)
    assert calls == []
    assert release_allocator_cache(
        V4_POLICY, device_free_bytes=101, pressure_free_bytes=100
    ) is False
    assert calls == []
    assert release_allocator_cache(
        V4_POLICY, device_free_bytes=99, pressure_free_bytes=100
    ) is True
    assert calls == [True]
    assert release_allocator_cache(V4_POLICY, force=True) is True
    assert calls == [True, True]


def test_ray_retry_attempts_are_bounded_and_checkpoint_only_at_floor():
    assert ray_retry_attempts(2048, 512) == [
        (2048, False), (1024, False), (512, False), (512, True),
    ]
    with pytest.raises(ValueError, match="base"):
        ray_retry_attempts(256, 512)


def test_headroom_gate_uses_capacity_reference_peak_and_margin(tmp_path):
    common = dict(
        experiment="C03-r8 Tier 2 onset study",
        retry_identity=V4_RETRY_IDENTITY,
        policy=V4_POLICY,
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
    assert source.index("del package, image") < source.index("allocator_cache_released = release_allocator_cache")
    assert "reflection_iteration == 101" in source
    assert "and opt.lambda_spec > 0" in source
    assert "STAGE_B_HEADROOM" in source
    assert "Stage B allocator headroom gate failed" not in source


def test_training_memory_retry_keeps_camera_clears_grads_and_recovers(monkeypatch):
    attempts = []
    recovered = {"loss": "finite"}

    def fake_attempt(camera, state, *_args):
        attempts.append((camera.image_name, state.ray_chunk_size, state.ray_checkpoint_chunks))
        if len(attempts) < 3:
            raise torch.cuda.OutOfMemoryError("synthetic pressure")
        return recovered

    class Optimizer:
        def __init__(self):
            self.zero_calls = 0

        def zero_grad(self, set_to_none=False):
            assert set_to_none is True
            self.zero_calls += 1

    exposure = Optimizer()
    diffuse_optimizer = Optimizer()
    reflection_optimizer = Optimizer()
    diffuse = SimpleNamespace(
        exposure_optimizer=exposure,
        optimizer=diffuse_optimizer,
    )
    reflection = SimpleNamespace(optimizer=reflection_optimizer)
    state = SimpleNamespace(ray_chunk_size=2048, ray_checkpoint_chunks=False)
    monkeypatch.setattr("stage_b_training._forward_backward_stage_b", fake_attempt)
    monkeypatch.setattr("stage_b_training.torch.cuda.max_memory_allocated", lambda: 123)
    monkeypatch.setattr("stage_b_training.torch.cuda.max_memory_reserved", lambda: 456)
    releases = []
    monkeypatch.setattr(
        "stage_b_training.release_allocator_cache",
        lambda *_args, **kwargs: releases.append(kwargs) or True,
    )
    result, records = _forward_backward_with_memory_retry(
        SimpleNamespace(image_name="000003.jpg"),
        state,
        None,
        None,
        SimpleNamespace(ray_chunk_size=2048),
        SimpleNamespace(stage_b_memory_retry_min_chunk_size=512),
        None,
        diffuse,
        reflection,
        3765,
        765,
    )
    assert result is recovered
    assert attempts == [
        ("000003.jpg", 2048, False),
        ("000003.jpg", 1024, False),
        ("000003.jpg", 512, False),
    ]
    assert len(records) == 2 and len(releases) == 2
    assert exposure.zero_calls == diffuse_optimizer.zero_calls == reflection_optimizer.zero_calls == 2
    assert state.ray_chunk_size == 2048 and state.ray_checkpoint_chunks is False
