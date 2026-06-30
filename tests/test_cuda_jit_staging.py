from pathlib import Path

import pytest

from raytracer.acceleration_structure import _prepare_ascii_jit_build


def test_cuda_jit_stages_non_ascii_repository_sources_under_ascii_paths(tmp_path, monkeypatch):
    cache_root = tmp_path / "rtgs-bvh-jit"
    assert str(cache_root).isascii()
    monkeypatch.setenv("RTGS_BVH_JIT_ROOT", str(cache_root))

    module_name, staged_sources, build_directory = _prepare_ascii_jit_build()

    assert module_name.isascii()
    assert str(build_directory).isascii()
    assert all(path.isascii() for path in staged_sources)
    source_root = Path(__file__).resolve().parents[1] / "raytracer" / "csrc"
    assert Path(staged_sources[0]).read_bytes() == (source_root / "bvh_bindings.cpp").read_bytes()
    assert Path(staged_sources[1]).read_bytes() == (source_root / "bvh_cuda.cu").read_bytes()


def test_cuda_jit_rejects_non_ascii_cache_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RTGS_BVH_JIT_ROOT", str(tmp_path / "non-ascii-缓存"))

    with pytest.raises(RuntimeError, match="ASCII-only"):
        _prepare_ascii_jit_build()
