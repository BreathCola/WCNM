"""Full-state Diffuse-only checkpoint contract for deterministic continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel
from utils.training_state import capture_rng_state, restore_rng_state


STAGE_A_FORMAT = "rtgs_stage_a"
FULL_STATE_CHECKPOINT_VERSION = 2


def make_full_stage_a_checkpoint(
    diffuse: DiffuseSurfelModel,
    iteration: int,
    config: Dict,
    runtime_state: Dict,
    optimizer_step_completed: bool,
) -> Dict:
    if not optimizer_step_completed:
        raise ValueError("full-state continuation checkpoint requires the endpoint optimizer step")
    return {
        "format": STAGE_A_FORMAT,
        "checkpoint_version": FULL_STATE_CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "model_state": diffuse.capture(),
        "config": dict(config),
        "rng_state": capture_rng_state(),
        "runtime_state": dict(runtime_state),
        "optimizer_step_completed": True,
    }


def validate_full_stage_a_checkpoint(checkpoint: Dict) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != STAGE_A_FORMAT:
        raise ValueError("checkpoint must be an rtgs_stage_a checkpoint")
    if checkpoint.get("checkpoint_version") != FULL_STATE_CHECKPOINT_VERSION:
        raise ValueError("Tier 2 requires a version-2 full-state Stage A checkpoint")
    for key in ("iteration", "model_state", "config", "rng_state", "runtime_state"):
        if key not in checkpoint:
            raise ValueError(f"full-state Stage A checkpoint is missing {key}")
    if checkpoint.get("optimizer_step_completed") is not True:
        raise ValueError("Stage A checkpoint does not include its endpoint optimizer step")


def restore_full_stage_a_checkpoint(
    checkpoint: Dict,
    diffuse: DiffuseSurfelModel,
    diffuse_args,
    restore_rng: bool = True,
) -> Tuple[int, Dict, Dict]:
    validate_full_stage_a_checkpoint(checkpoint)
    diffuse.restore(checkpoint["model_state"], diffuse_args)
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["iteration"]), dict(checkpoint["runtime_state"]), dict(checkpoint["config"])


def load_full_stage_a_checkpoint(
    path,
    diffuse: DiffuseSurfelModel,
    diffuse_args,
    map_location=None,
    restore_rng: bool = True,
):
    checkpoint = torch.load(Path(path), map_location=map_location)
    return restore_full_stage_a_checkpoint(checkpoint, diffuse, diffuse_args, restore_rng=restore_rng)
