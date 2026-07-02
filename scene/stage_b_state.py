"""Versioned Stage B initialization and checkpoint helpers."""

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_a_state import FULL_STATE_CHECKPOINT_VERSION, validate_full_stage_a_checkpoint
from utils.training_state import capture_rng_state, restore_rng_state, rng_states_equal


STAGE_B_FORMAT = "rtgs_stage_b"
STAGE_B_CHECKPOINT_VERSION = 1
STAGE_B_FULL_STATE_CHECKPOINT_VERSION = 2


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage_a_diffuse_checkpoint(
    path,
    diffuse: DiffuseSurfelModel,
    diffuse_args,
    map_location=None,
    require_full_state: bool = False,
    restore_rng: bool = False,
    return_runtime_state: bool = False,
):
    hash_before = sha256_file(path)
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "rtgs_stage_a":
        raise ValueError("--diffuse_init_checkpoint must be an rtgs_stage_a checkpoint")
    if checkpoint.get("iteration") is None or "model_state" not in checkpoint:
        raise ValueError("Stage A checkpoint is missing iteration or model_state")
    if require_full_state:
        validate_full_stage_a_checkpoint(checkpoint)
    diffuse.restore(checkpoint["model_state"], diffuse_args)
    if restore_rng:
        if "rng_state" not in checkpoint:
            raise ValueError("Stage A checkpoint has no RNG state")
        restore_rng_state(checkpoint["rng_state"])
    hash_after = sha256_file(path)
    if hash_before != hash_after:
        raise RuntimeError("Diffuse initialization checkpoint changed while it was being read")
    provenance = {
        "path": str(Path(path).resolve()),
        "sha256": hash_after,
        "format": checkpoint["format"],
        "iteration": int(checkpoint["iteration"]),
        "config": dict(checkpoint.get("config", {})),
    }
    runtime_state = checkpoint.get("runtime_state")
    if return_runtime_state:
        return int(checkpoint["iteration"]), provenance, (
            None if runtime_state is None else dict(runtime_state)
        )
    return int(checkpoint["iteration"]), provenance


def initialize_stage_b_from_diffuse(
    path,
    diffuse: DiffuseSurfelModel,
    reflection: ReflectionSurfelModel,
    diffuse_args,
    reflection_args,
    reflection_count: int,
    reflection_seed: int,
    map_location=None,
    require_full_state: bool = False,
    return_runtime_state: bool = False,
    verify_rng_unchanged: bool = False,
):
    global_iteration, provenance, runtime_state = load_stage_a_diffuse_checkpoint(
        path,
        diffuse,
        diffuse_args,
        map_location=map_location,
        require_full_state=require_full_state,
        restore_rng=require_full_state,
        return_runtime_state=True,
    )
    rng_before = capture_rng_state() if verify_rng_unchanged else None
    xyz = diffuse.get_xyz.detach()
    reflection.create_random_bbox(xyz.amin(dim=0), xyz.amax(dim=0), reflection_count, reflection_seed)
    reflection.training_setup(reflection_args)
    if verify_rng_unchanged and not rng_states_equal(rng_before, capture_rng_state()):
        raise RuntimeError("fresh Reflection initialization changed restored global RNG state")
    if return_runtime_state:
        return global_iteration, 0, provenance, runtime_state
    return global_iteration, 0, provenance


def make_stage_b_checkpoint(
    diffuse: DiffuseSurfelModel,
    reflection: ReflectionSurfelModel,
    global_iteration: int,
    reflection_iteration: int,
    provenance: Dict,
    config: Dict,
    runtime_state: Optional[Dict] = None,
    optimizer_step_completed: Optional[bool] = None,
) -> Dict:
    full_state = runtime_state is not None
    if full_state and optimizer_step_completed is not True:
        raise ValueError("full-state Stage B checkpoint requires the endpoint optimizer step")
    checkpoint = {
        "format": STAGE_B_FORMAT,
        "checkpoint_version": (
            STAGE_B_FULL_STATE_CHECKPOINT_VERSION if full_state else STAGE_B_CHECKPOINT_VERSION
        ),
        "global_iteration": int(global_iteration),
        "reflection_iteration": int(reflection_iteration),
        "diffuse": diffuse.capture(),
        "reflection": reflection.capture(),
        "provenance": dict(provenance),
        "config": dict(config),
        "rng_state": capture_rng_state(),
    }
    if full_state:
        checkpoint["runtime_state"] = dict(runtime_state)
        checkpoint["optimizer_step_completed"] = True
    return checkpoint


def restore_stage_b_checkpoint(
    checkpoint: Dict,
    diffuse: DiffuseSurfelModel,
    reflection: ReflectionSurfelModel,
    diffuse_args,
    reflection_args,
    restore_rng: bool = True,
    require_full_state: bool = False,
    return_runtime_state: bool = False,
) -> Tuple[int, int, Dict, Dict]:
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != STAGE_B_FORMAT:
        raise ValueError("--start_checkpoint must be an rtgs_stage_b checkpoint")
    version = checkpoint.get("checkpoint_version")
    if version not in (STAGE_B_CHECKPOINT_VERSION, STAGE_B_FULL_STATE_CHECKPOINT_VERSION):
        raise ValueError(f"unsupported Stage B checkpoint version: {checkpoint.get('checkpoint_version')}")
    if require_full_state:
        if version != STAGE_B_FULL_STATE_CHECKPOINT_VERSION:
            raise ValueError("Tier 2 requires a version-2 full-state Stage B checkpoint")
        if "runtime_state" not in checkpoint or checkpoint.get("optimizer_step_completed") is not True:
            raise ValueError("Stage B checkpoint lacks continuation runtime/optimizer state")
    diffuse.restore(checkpoint["diffuse"], diffuse_args)
    reflection.restore(checkpoint["reflection"], reflection_args)
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))
    values = (
        int(checkpoint["global_iteration"]),
        int(checkpoint["reflection_iteration"]),
        dict(checkpoint.get("provenance", {})),
        dict(checkpoint.get("config", {})),
    )
    if return_runtime_state:
        runtime_state = checkpoint.get("runtime_state")
        return values + (None if runtime_state is None else dict(runtime_state),)
    return values


def load_stage_b_checkpoint(path, *args, map_location=None, **kwargs):
    checkpoint = torch.load(path, map_location=map_location)
    return restore_stage_b_checkpoint(checkpoint, *args, **kwargs)
