from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from stage_d_training import (
    FORMAL_NODES, FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256, _validate_formal_contract,
)
from utils.transmittance_debug import make_stage_d_contact_sheet
from tools.audit_stage_d_formal import plot_curves
from tools.run_stage_d_formal_onset import (
    ZERO_STEP_ATTEMPT_COMMIT, classify_compute_processes,
    preserve_known_zero_step_attempt,
)


def formal_inputs(tmp_path):
    dataset = SimpleNamespace(
        resolution=8, ray_chunk_size=512, transmittance_init_mode="random_bbox",
        transmittance_init_count=4096, transmittance_init_seed=20260703,
        model_path=str(tmp_path / "stage_d_tihubird_c03r8_formal_onset_g15000_g20000_v1"),
        _validated_specular_mask_manifest={
            "manifest_file_sha256": "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551",
            "aggregate_sha256": "54dbb7661efbb2a334d86cef1cfec1d15ff812e71856c0d88930754013abc2e6",
        },
    )
    opt = SimpleNamespace(
        stage_d_formal_onset=True, iterations=20000,
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
                "manifest_file_sha256": "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551",
                "aggregate_sha256": "54dbb7661efbb2a334d86cef1cfec1d15ff812e71856c0d88930754013abc2e6",
            },
        },
    }
    return dataset, opt, release, source


def test_formal_contract_accepts_only_exact_fresh_trajectory(tmp_path):
    dataset, opt, release, source = formal_inputs(tmp_path)
    _validate_formal_contract(
        dataset, opt, release, source, True, list(FORMAL_NODES), list(FORMAL_NODES)
    )
    opt.stage_d_depth_start_iteration = 15001
    with pytest.raises(ValueError, match="depth_start"):
        _validate_formal_contract(
            dataset, opt, release, source, True, list(FORMAL_NODES), list(FORMAL_NODES)
        )


def test_contact_sheet_requires_and_places_every_panel(tmp_path):
    stems = ("000000", "000012")
    names = (
        "ground_truth.png", "final.png", "diffuse_contribution.png",
        "reflection_contribution.png", "transmittance_contribution.png",
        "inside_color.png", "outside_color.png", "din_vs_far_violation.png",
    )
    for row, stem in enumerate(stems):
        directory = tmp_path / stem
        directory.mkdir()
        for col, name in enumerate(names):
            Image.new("RGB", (64, 36), (row * 80, col * 20, 10)).save(directory / name)
    target = Path(make_stage_d_contact_sheet(tmp_path, stems))
    with Image.open(target) as image:
        assert image.size == (320 * 8, (180 + 28) * 2)


def test_cpu_curve_writer_has_no_optional_plot_dependency(tmp_path):
    rows = []
    for step in (15001, 15002):
        rows.append({
            "global_iteration": step,
            "loss": {"total": 1.0, "rgb": .8, "l_spec": .1, "l_depth": .2},
            "din_le_t_far_fraction": .5, "counts": {
                "diffuse": 10, "reflection": 5, "transmittance": step - 14000,
            },
            "whole_step_wall_ms": 1000.0,
            "cuda_max_memory_allocated_bytes": 2**30,
            "cuda_max_memory_reserved_bytes": 2 * 2**30,
        })
    node_metrics = {
        node: {
            stem: {
                "din_le_t_far_fraction": .5 + node / 1e6,
                "inside_alpha_mean": .1, "inside_energy": .2,
                "outside_energy": .3,
            }
            for stem in ("000000", "000012", "000039", "000040", "000041", "000053", "000063", "000083", "000110")
        }
        for node in FORMAL_NODES
    }
    target = tmp_path / "curves.png"
    node_target = plot_curves(rows, node_metrics, target)
    Image.open(target).verify()
    Image.open(node_target).verify()


def test_gpu_preflight_allows_only_bounded_remote_desktop_process():
    observed, conflicts = classify_compute_processes(
        "2219, /usr/libexec/gnome-remote-desktop-daemon, 260\n"
    )
    assert len(observed) == 1 and conflicts == []
    _, conflicts = classify_compute_processes("999, python, 1024\n")
    assert conflicts[0]["process_name"] == "python"


def test_known_zero_step_attempt_is_preserved_without_deletion(tmp_path):
    output = tmp_path / "formal"
    output.mkdir()
    log = tmp_path / "formal.log"
    for name in ("cameras.json", "cfg_args", "input.ply", "events.out.tfevents.test"):
        (output / name).write_text(name, encoding="utf-8")
    record = {
        "schema": "rtgs_stage_d_formal_operator_v1", "git_commit": ZERO_STEP_ATTEMPT_COMMIT,
        "status": "BLOCKED", "training_exit_code": 1,
        "source_sha256_before": FORMAL_SOURCE_SHA256,
        "source_sha256_after": FORMAL_SOURCE_SHA256,
        "release_aggregate_before": FORMAL_RELEASE_SHA256,
        "release_aggregate_after": FORMAL_RELEASE_SHA256,
    }
    (output / "formal_operator_record.json").write_text(
        __import__("json").dumps(record), encoding="utf-8"
    )
    log.write_text(
        "checkpoint CUDA RNG cardinality does not match visible CUDA devices",
        encoding="utf-8",
    )
    archive = Path(preserve_known_zero_step_attempt(output, log))
    assert not log.exists()
    assert {path.name for path in archive.iterdir()} == {
        "cameras.json", "cfg_args", "input.ply", "events.out.tfevents.test",
        "formal_operator_record.json", "operator.log",
    }
