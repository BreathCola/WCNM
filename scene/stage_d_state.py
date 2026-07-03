"""Versioned Stage D D/R/T checkpoint and initialization contract."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from scene.stage_b_state import load_stage_b_checkpoint, sha256_file
from utils.training_state import capture_rng_state, restore_rng_state


STAGE_D_FORMAT = "rtgs_stage_d"
STAGE_D_CHECKPOINT_VERSION = 1


def initialize_stage_d_from_stage_b(
    path,
    diffuse,
    reflection,
    transmittance,
    diffuse_args,
    reflection_args,
    transmittance_args,
    transmittance_bbox_min,
    transmittance_bbox_max,
    transmittance_count: int,
    transmittance_seed: int,
    map_location=None,
):
    before = sha256_file(path)
    global_iteration, reflection_iteration, provenance, saved_config, runtime_state = (
        load_stage_b_checkpoint(
            path, diffuse, reflection, diffuse_args, reflection_args,
            map_location=map_location, return_runtime_state=True,
        )
    )
    transmittance.create_random_bbox(
        transmittance_bbox_min, transmittance_bbox_max,
        transmittance_count, transmittance_seed,
    )
    transmittance.initialization["branch"] = "transmittance"
    transmittance.training_setup(transmittance_args)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("Stage B source checkpoint changed while initializing Stage D")
    source = {
        "path": str(Path(path).resolve()), "sha256": after,
        "format": "rtgs_stage_b", "global_iteration": global_iteration,
        "reflection_iteration": reflection_iteration,
        "stage_b_provenance": provenance, "stage_b_config": saved_config,
    }
    return global_iteration, reflection_iteration, 0, source, runtime_state


def make_stage_d_checkpoint(
    diffuse,
    reflection,
    transmittance,
    global_iteration: int,
    reflection_iteration: int,
    transmittance_iteration: int,
    source: Dict,
    config: Dict,
    geometry_release_id: str,
    geometry_release_aggregate_sha256: str,
    runtime_state: Dict,
) -> Dict:
    return {
        "format": STAGE_D_FORMAT,
        "checkpoint_version": STAGE_D_CHECKPOINT_VERSION,
        "global_iteration": int(global_iteration),
        "reflection_iteration": int(reflection_iteration),
        "transmittance_iteration": int(transmittance_iteration),
        "diffuse": diffuse.capture(), "reflection": reflection.capture(),
        "transmittance": transmittance.capture(), "source": dict(source),
        "config": dict(config),
        "geometry_release_id": str(geometry_release_id),
        "geometry_release_aggregate_sha256": str(geometry_release_aggregate_sha256),
        "runtime_state": dict(runtime_state), "optimizer_step_completed": True,
        "rng_state": capture_rng_state(),
    }


def restore_stage_d_checkpoint(
    checkpoint,
    diffuse,
    reflection,
    transmittance,
    diffuse_args,
    reflection_args,
    transmittance_args,
    expected_geometry_release_id: str,
    expected_geometry_release_aggregate_sha256: str,
    restore_rng: bool = True,
):
    if checkpoint.get("format") != STAGE_D_FORMAT:
        raise ValueError("Stage D resume requires an rtgs_stage_d checkpoint")
    if checkpoint.get("checkpoint_version") != STAGE_D_CHECKPOINT_VERSION:
        raise ValueError("unsupported Stage D checkpoint version")
    if checkpoint.get("optimizer_step_completed") is not True:
        raise ValueError("Stage D checkpoint does not contain its endpoint optimizer step")
    if checkpoint.get("geometry_release_id") != expected_geometry_release_id:
        raise ValueError("Stage D checkpoint geometry release ID mismatch")
    if (
        checkpoint.get("geometry_release_aggregate_sha256")
        != expected_geometry_release_aggregate_sha256
    ):
        raise ValueError("Stage D checkpoint geometry aggregate mismatch")
    diffuse.restore(checkpoint["diffuse"], diffuse_args)
    reflection.restore(checkpoint["reflection"], reflection_args)
    transmittance.restore(checkpoint["transmittance"], transmittance_args)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["global_iteration"]),
        int(checkpoint["reflection_iteration"]),
        int(checkpoint["transmittance_iteration"]),
        dict(checkpoint["source"]), dict(checkpoint["config"]),
        dict(checkpoint["runtime_state"]),
    )


def load_stage_d_checkpoint(path, *args, map_location=None, **kwargs):
    checkpoint = torch.load(path, map_location=map_location)
    return restore_stage_d_checkpoint(checkpoint, *args, **kwargs)
