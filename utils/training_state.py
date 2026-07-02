"""Deterministic RNG and camera-deck state used by bounded training resumes."""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch


RUNTIME_STATE_VERSION = 1


def should_step_optimizer(iteration: int, endpoint: int, continuation: bool) -> bool:
    """Include the bounded endpoint update only for resumable operator gates."""
    return int(iteration) < int(endpoint) or bool(continuation)


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
    required = {"python", "numpy", "torch", "cuda"}
    if set(state) != required:
        raise ValueError(f"RNG state keys must be exactly {sorted(required)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_rng_byte_tensor(state["torch"], "rng_state.torch"))
    if state["cuda"] is not None and torch.cuda.is_available():
        cuda_states = [
            _cpu_rng_byte_tensor(value, f"rng_state.cuda[{index}]")
            for index, value in enumerate(state["cuda"])
        ]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG cardinality does not match visible CUDA devices: "
                f"checkpoint={len(cuda_states)} visible={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def rng_states_equal(left: Dict, right: Dict) -> bool:
    if set(left) != set(right):
        return False
    if left["python"] != right["python"] or not _numpy_state_equal(left["numpy"], right["numpy"]):
        return False
    if not torch.equal(
        _cpu_rng_byte_tensor(left["torch"], "left.torch"),
        _cpu_rng_byte_tensor(right["torch"], "right.torch"),
    ):
        return False
    left_cuda, right_cuda = left.get("cuda"), right.get("cuda")
    if left_cuda is None or right_cuda is None:
        return left_cuda is None and right_cuda is None
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(
            _cpu_rng_byte_tensor(a, f"left.cuda[{index}]"),
            _cpu_rng_byte_tensor(b, f"right.cuda[{index}]"),
        )
        for index, (a, b) in enumerate(zip(left_cuda, right_cuda))
    )


def make_camera_runtime_state(remaining_indices: Sequence[int], camera_count: int) -> Dict:
    indices = [int(value) for value in remaining_indices]
    if camera_count <= 0:
        raise ValueError("camera_count must be positive")
    if len(indices) != len(set(indices)) or any(value < 0 or value >= camera_count for value in indices):
        raise ValueError("remaining camera indices must be unique and in range")
    return {
        "runtime_state_version": RUNTIME_STATE_VERSION,
        "camera_count": int(camera_count),
        "remaining_camera_indices": indices,
    }


def restore_camera_deck(cameras: Sequence, runtime_state: Optional[Dict]):
    cameras = list(cameras)
    if runtime_state is None:
        indices = list(range(len(cameras)))
        return cameras.copy(), indices
    if runtime_state.get("runtime_state_version") != RUNTIME_STATE_VERSION:
        raise ValueError("unsupported training runtime state version")
    if int(runtime_state.get("camera_count", -1)) != len(cameras):
        raise ValueError("checkpoint camera count does not match the current scene")
    indices = [int(value) for value in runtime_state.get("remaining_camera_indices", [])]
    if len(indices) != len(set(indices)) or any(value < 0 or value >= len(cameras) for value in indices):
        raise ValueError("checkpoint remaining camera indices are invalid")
    return [cameras[index] for index in indices], indices
