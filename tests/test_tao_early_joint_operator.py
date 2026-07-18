from pathlib import Path

import numpy as np
import pytest

import tools.run_tao_layered_early_joint as operator
from utils.tao_early_joint_plan import support_safe_global0_initialization


def test_support_safe_global0_initialization_is_deterministic_and_owned():
    rng = np.random.default_rng(4)
    points = rng.uniform(-3, 3, size=(800, 3))
    first = support_safe_global0_initialization(
        points, np.eye(3), np.array([-1.0] * 3), np.array([1.0] * 3),
        reflection_count=64, transmittance_count=64,
    )
    second = support_safe_global0_initialization(
        points, np.eye(3), np.array([-1.0] * 3), np.array([1.0] * 3),
        reflection_count=64, transmittance_count=64,
    )
    assert first["R"]["complete_support_strict_outside"] is True
    assert first["T"]["complete_support_strict_inside"] is True
    assert first["R"]["position_sha256"] == second["R"]["position_sha256"]
    assert first["T"]["position_sha256"] == second["T"]["position_sha256"]
    assert first["T"]["semantic_mask_used"] is False


def test_operator_default_is_plan_only_and_execute_is_only_launch_gate(monkeypatch, capsys):
    launches = []
    monkeypatch.setattr(operator, "build_tao_early_joint_plan", lambda **kwargs: {
        "schema": "test", "output": str(kwargs["output"]),
    })
    monkeypatch.setattr(operator, "_launch_training", lambda plan: launches.append(plan))
    assert operator.main([]) == 0
    assert launches == []
    assert operator.main(["--execute"]) == 0
    assert len(launches) == 1
    assert '"schema": "test"' in capsys.readouterr().out


def test_checkpoint_sources_are_rejected_and_parser_has_no_checkpoint_action():
    actions = {action.dest for action in operator.build_parser()._actions}
    assert "start_checkpoint" not in actions
    assert "checkpoint_source" not in actions
    with pytest.raises(SystemExit, match="CHECKPOINT_SOURCE_FORBIDDEN"):
        operator.main(["--start_checkpoint", "old.pth"])


def test_missing_formal_mask_and_existing_output_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="BLOCKED_BY_GLASS_MASK_IDENTITY"):
        operator.build_tao_early_joint_plan(
            scene=operator.ROOT / "data/Tao",
            reviewed_mask_manifest=tmp_path / "missing.json",
            geometry_release=tmp_path / "missing_geometry.json",
            output=tmp_path / "fresh", execute=False,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing existing"):
        operator.build_tao_early_joint_plan(
            scene=operator.ROOT / "data/Tao",
            reviewed_mask_manifest=tmp_path / "missing.json",
            geometry_release=tmp_path / "missing_geometry.json",
            output=existing, execute=False,
        )
