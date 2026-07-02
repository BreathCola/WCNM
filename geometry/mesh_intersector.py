"""ASCII-staged CPU BVH for fixed Stage C triangle meshes."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    source = Path(__file__).resolve().parent / "csrc" / "mesh_bvh.cpp"
    contents = source.read_bytes()
    digest = hashlib.sha256(contents + torch.__version__.encode() + sys.executable.encode()).hexdigest()[:16]
    root = Path(os.environ.get("RTGS_MESH_JIT_ROOT", f"/tmp/rtgs-mesh-jit-{os.getuid()}")).resolve()
    if not str(root).isascii():
        raise RuntimeError("RTGS_MESH_JIT_ROOT must be ASCII-only")
    directory = root / digest
    sources = directory / "sources"
    build = directory / "build"
    sources.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    staged = sources / "mesh_bvh.cpp"
    if not staged.exists() or staged.read_bytes() != contents:
        temporary = staged.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(contents)
        os.replace(temporary, staged)
    name = f"rtgs_mesh_bvh_{digest}"
    try:
        _EXTENSION = importlib.import_module(name)
    except ImportError:
        _EXTENSION = load(
            name=name, sources=[str(staged)], build_directory=str(build),
            extra_cflags=["-O3", "-fopenmp"], extra_ldflags=["-fopenmp"], verbose=False,
        )
    return _EXTENSION


class MeshIntersector:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        extension = load_extension()
        self._bvh = extension.MeshBVH(
            torch.from_numpy(np.asarray(vertices, dtype=np.float32)),
            torch.from_numpy(np.asarray(faces, dtype=np.int64)),
        )

    def intersect(self, origins: np.ndarray, directions: np.ndarray):
        outputs = self._bvh.intersect(
            torch.from_numpy(np.asarray(origins, dtype=np.float32)),
            torch.from_numpy(np.asarray(directions, dtype=np.float32)),
        )
        return tuple(value.numpy() for value in outputs)
