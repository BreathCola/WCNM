"""Versioned Stage B initialization and checkpoint helpers."""

import hashlib
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel


STAGE_B_FORMAT = "rtgs_stage_b"
STAGE_B_CHECKPOINT_VERSION = 1


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage_a_diffuse_checkpoint(path, diffuse: DiffuseSurfelModel, diffuse_args, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "rtgs_stage_a":
        raise ValueError("--diffuse_init_checkpoint must be an rtgs_stage_a checkpoint")
    if checkpoint.get("iteration") is None or "model_state" not in checkpoint:
        raise ValueError("Stage A checkpoint is missing iteration or model_state")
    diffuse.restore(checkpoint["model_state"], diffuse_args)
    return int(checkpoint["iteration"]), {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "format": checkpoint["format"],
        "iteration": int(checkpoint["iteration"]),
        "config": dict(checkpoint.get("config", {})),
    }


def initialize_stage_b_from_diffuse(
    path,
    diffuse: DiffuseSurfelModel,
    reflection: ReflectionSurfelModel,
    diffuse_args,
    reflection_args,
    reflection_count: int,
    reflection_seed: int,
    map_location=None,
):
    global_iteration, provenance = load_stage_a_diffuse_checkpoint(
        path, diffuse, diffuse_args, map_location=map_location
    )
    xyz = diffuse.get_xyz.detach()
    reflection.create_random_bbox(xyz.amin(dim=0), xyz.amax(dim=0), reflection_count, reflection_seed)
    reflection.training_setup(reflection_args)
    return global_iteration, 0, provenance


def capture_rng_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_rng_byte_tensor(value, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.uint8:
        raise ValueError(f"{name} must be a torch uint8 RNG state tensor")
    return value.detach().to(device="cpu").contiguous()


def restore_rng_state(state: Optional[Dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_rng_byte_tensor(state["torch"], "rng_state.torch"))
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [
            _cpu_rng_byte_tensor(value, f"rng_state.cuda[{index}]")
            for index, value in enumerate(state["cuda"])
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def make_stage_b_checkpoint(
    diffuse: DiffuseSurfelModel,
    reflection: ReflectionSurfelModel,
    global_iteration: int,
    reflection_iteration: int,
    provenance: Dict,
    config: Dict,
) -> Dict:
    return {
        "format": STAGE_B_FORMAT,
        "checkpoint_version": STAGE_B_CHECKPOINT_VERSION,
        "global_iteration": int(global_iteration),
        "reflection_iteration": int(reflection_iteration),
        "diffuse": diffuse.capture(),
        "reflection": reflection.capture(),
        "provenance": dict(provenance),
        "config": dict(config),
        "rng_state": capture_rng_state(),
    }


def restore_stage_b_checkpoint(
    checkpoint: Dict,
    diffuse: DiffuseSurfelModel,
    reflection: ReflectionSurfelModel,
    diffuse_args,
    reflection_args,
    restore_rng: bool = True,
) -> Tuple[int, int, Dict, Dict]:
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != STAGE_B_FORMAT:
        raise ValueError("--start_checkpoint must be an rtgs_stage_b checkpoint")
    if checkpoint.get("checkpoint_version") != STAGE_B_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported Stage B checkpoint version: {checkpoint.get('checkpoint_version')}")
    diffuse.restore(checkpoint["diffuse"], diffuse_args)
    reflection.restore(checkpoint["reflection"], reflection_args)
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))
    return (
        int(checkpoint["global_iteration"]),
        int(checkpoint["reflection_iteration"]),
        dict(checkpoint.get("provenance", {})),
        dict(checkpoint.get("config", {})),
    )


def load_stage_b_checkpoint(path, *args, map_location=None, **kwargs):
    checkpoint = torch.load(path, map_location=map_location)
    return restore_stage_b_checkpoint(checkpoint, *args, **kwargs)
