#!/usr/bin/env python3
"""Generate RT-GS Stage A monocular normal priors with StableNormal.

The on-disk contract matches ``utils.camera_utils.loadCam``:

* filename: ``<image stem>.npy`` in a flat output directory;
* layout: HWC with exactly three channels;
* dtype/range: float32 unit vectors with components in [-1, 1];
* space: StableNormal camera space (training transforms and face-forwards it).

StableNormal stays in its own environment. This script imports it lazily only
after forcing offline mode, and it never imports StableNormal from RT-GS code.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NORMAL_EPS = 1e-6
UNIT_ATOL = 5e-4
MANIFEST_SCHEMA_VERSION = 1
DINO_CHECKPOINT_NAME = "dinov2_vitl14_pretrain.pth"


class OfflineViolation(RuntimeError):
    """Raised when a dependency attempts network access in offline mode."""


def prior_filename(image_path: Path) -> str:
    """Return the exact filename convention used by the RT-GS loader."""
    return f"{Path(image_path).stem}.npy"


def prior_path_for_image(image_path: Path, output_directory: Path) -> Path:
    return Path(output_directory) / prior_filename(image_path)


def decode_rgb_normal(normal_rgb: Image.Image | np.ndarray) -> np.ndarray:
    """Decode an RGB normal map from [0,255] to normalized HWC float32."""
    values = np.asarray(normal_rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"RGB normal map must have shape [H,W,3], got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"RGB normal map must be numeric, got {values.dtype}")

    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("RGB normal map contains NaN or Inf")
    if values.size == 0 or values.min() < 0.0 or values.max() > 255.0:
        raise ValueError("RGB normal map values must be in [0,255]")

    normals = values / np.float32(127.5) - np.float32(1.0)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    invalid = (~np.isfinite(normals).all(axis=-1, keepdims=True)) | (lengths <= NORMAL_EPS)
    if invalid.any():
        raise ValueError(
            f"decoded normal map contains {int(invalid.sum())} non-finite or near-zero vectors"
        )
    normals = normals / lengths
    return np.ascontiguousarray(normals, dtype=np.float32)


def validate_prior_array(normals: np.ndarray, expected_hw: tuple[int, int] | None = None) -> None:
    """Validate the stricter HWC contract emitted by this generator."""
    if normals.dtype != np.float32:
        raise ValueError(f"prior dtype must be float32, got {normals.dtype}")
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(f"prior must have HWC shape [H,W,3], got {normals.shape}")
    if expected_hw is not None and tuple(normals.shape[:2]) != tuple(expected_hw):
        raise ValueError(
            f"prior/image size mismatch: prior={tuple(normals.shape[:2])}, image={expected_hw}"
        )
    if not np.isfinite(normals).all():
        raise ValueError("prior contains NaN or Inf")
    lengths = np.linalg.norm(normals, axis=-1)
    if (lengths <= NORMAL_EPS).any():
        raise ValueError("prior contains near-zero vectors")
    max_error = float(np.max(np.abs(lengths - 1.0)))
    if max_error > UNIT_ATOL:
        raise ValueError(f"prior vectors are not unit length (max error {max_error:.6g})")
    if float(np.max(np.abs(normals))) > 1.0 + UNIT_ATOL:
        raise ValueError("prior components fall outside [-1,1]")


def validate_prior_file(path: Path, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    with Path(path).open("rb") as handle:
        normals = np.load(handle, allow_pickle=False)
    validate_prior_array(normals, expected_hw)
    return normals


def atomic_save_npy(path: Path, normals: np.ndarray) -> None:
    """Validate, write, fsync, re-read, and atomically install one prior."""
    path = Path(path)
    validate_prior_array(normals)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, normals, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        validate_prior_file(temporary_path, tuple(normals.shape[:2]))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def collect_images(scene: Path, images_directory: str, requested: Sequence[str]) -> list[Path]:
    image_root = scene / images_directory
    if not image_root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_root}")
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if requested:
        requested_set = set(requested)
        selected = [
            path for path in images
            if path.name in requested_set
            or path.stem in requested_set
            or path.relative_to(image_root).as_posix() in requested_set
        ]
        matched = {
            item for item in requested_set
            if any(
                item in {path.name, path.stem, path.relative_to(image_root).as_posix()}
                for path in selected
            )
        }
        missing = sorted(requested_set - matched)
        if missing:
            raise FileNotFoundError(f"requested images were not found: {missing}")
        images = selected
    if not images:
        raise FileNotFoundError(f"no supported images found in {image_root}")

    stems: dict[str, Path] = {}
    for path in images:
        previous = stems.setdefault(path.stem, path)
        if previous != path:
            raise ValueError(
                "image stems must be unique because the loader uses flat <stem>.npy names: "
                f"{previous} and {path}"
            )
    return images


def resolve_dino_checkpoint(dino_source: Path) -> Path:
    candidates = (
        dino_source.parent / "checkpoints" / DINO_CHECKPOINT_NAME,
        dino_source / DINO_CHECKPOINT_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"offline DINOv2 checkpoint not found; expected {expected}")


def _git_commit(directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _per_image_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**31)


def configure_offline_environment(torch_home: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(torch_home)


@contextlib.contextmanager
def block_network() -> Iterator[None]:
    """Fail immediately if Python code attempts a network connection."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_urlopen = urllib.request.urlopen

    def deny(*_args: Any, **_kwargs: Any) -> Any:
        raise OfflineViolation("network access is disabled for normal-prior generation")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    urllib.request.urlopen = deny
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection
        urllib.request.urlopen = original_urlopen


@contextlib.contextmanager
def local_torch_hub(
    torch: Any,
    stable_source: Path,
    dino_source: Path,
    dino_checkpoint: Path,
) -> Iterator[None]:
    """Allow only local StableNormal/DINO hub loads and local DINO weights."""
    original_load = torch.hub.load
    original_load_state_dict = torch.hub.load_state_dict_from_url
    original_download = torch.hub.download_url_to_file
    stable_source = stable_source.resolve()
    dino_source = dino_source.resolve()

    def guarded_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if model == "dinov2_vitl14":
            kwargs["source"] = "local"
            kwargs["weights"] = str(dino_checkpoint)
            return original_load(str(dino_source), model, *args, **kwargs)
        source = kwargs.get("source", "github")
        try:
            repo_path = Path(repo_or_dir).expanduser().resolve()
        except (OSError, TypeError):
            repo_path = None
        if repo_path == stable_source and model == "StableNormal" and source == "local":
            return original_load(str(stable_source), model, *args, **kwargs)
        raise OfflineViolation(f"blocked non-local torch.hub.load({repo_or_dir!r}, {model!r})")

    def local_state_dict(url: str, *args: Any, **kwargs: Any) -> Any:
        parsed = urlparse(str(url))
        if parsed.scheme != "file":
            raise OfflineViolation(f"blocked torch weight download: {url}")
        local_path = Path(unquote(parsed.path)).resolve()
        if local_path != dino_checkpoint.resolve():
            raise OfflineViolation(f"unexpected local torch weight path: {local_path}")
        map_location = kwargs.get("map_location", "cpu")
        try:
            return torch.load(local_path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(local_path, map_location=map_location)

    def deny_download(url: str, *_args: Any, **_kwargs: Any) -> Any:
        raise OfflineViolation(f"blocked torch download: {url}")

    torch.hub.load = guarded_load
    torch.hub.load_state_dict_from_url = local_state_dict
    torch.hub.download_url_to_file = deny_download
    try:
        yield
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_load_state_dict
        torch.hub.download_url_to_file = original_download


def _load_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "files": []}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema in {path}")
    if not isinstance(manifest.get("files"), list):
        raise ValueError(f"manifest files must be a list: {path}")
    return manifest


def _record_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    invocation: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    entries = {
        item["output_file"]: item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and "output_file" in item
    }
    output_file = entry["output_file"]
    entries[output_file] = {**entries.get(output_file, {}), **entry}
    manifest.update(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "normal_format": {
                "layout": "HWC",
                "dtype": "float32",
                "component_range": "[-1,1]",
                "vector_length": "unit",
                "space": "camera",
                "encoding": "StableNormal RGB decoded as rgb/127.5-1 then normalized",
                "face_forward": "deferred to RT-GS training after camera-to-world transform",
            },
            "last_invocation": invocation,
            "files": [entries[key] for key in sorted(entries)],
        }
    )
    atomic_write_json(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline HWC float32 camera-space StableNormal priors for RT-GS Stage A."
    )
    parser.add_argument("--scene", type=Path, required=True, help="RT-GS scene root")
    parser.add_argument("--output", type=Path, required=True, help="output normal_priors directory")
    parser.add_argument("--stable-source", type=Path, required=True, help="local StableNormal source")
    parser.add_argument("--weights", type=Path, required=True, help="local StableNormal weights root")
    parser.add_argument("--dino-source", type=Path, required=True, help="local DINOv2 source tree")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=768, help="StableNormal max-side resolution")
    parser.add_argument("--steps", type=int, default=10, help="StableNormal denoising steps")
    parser.add_argument("--images", default="images", help="image directory relative to scene")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="process only this filename, relative path, or stem; repeat as needed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path("~/.cache/stablenormal/torch"),
        help="isolated Torch cache (network remains blocked)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    scene = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()
    stable_source = args.stable_source.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    dino_source = args.dino_source.expanduser().resolve()
    torch_home = args.torch_home.expanduser().resolve()

    if not scene.is_dir():
        raise FileNotFoundError(f"scene does not exist: {scene}")
    if not (stable_source / "hubconf.py").is_file():
        raise FileNotFoundError(f"StableNormal hubconf.py not found in {stable_source}")
    for model_directory in ("yoso-normal-v0-3", "stable-normal-v0-1"):
        if not (weights / model_directory / "model_index.json").is_file():
            raise FileNotFoundError(f"StableNormal model is incomplete: {weights / model_directory}")
    if not (dino_source / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 hubconf.py not found in {dino_source}")
    dino_checkpoint = resolve_dino_checkpoint(dino_source)

    images = collect_images(scene, args.images, args.image)
    output.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    configure_offline_environment(torch_home)

    manifest_path = output / "manifest.json"
    manifest = _load_existing_manifest(manifest_path)
    invocation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scene": str(scene),
        "image_directory": str((scene / args.images).resolve()),
        "output": str(output),
        "stable_source": str(stable_source),
        "stable_source_commit": _git_commit(stable_source),
        "weights": str(weights),
        "dino_source": str(dino_source),
        "dino_source_commit": _git_commit(dino_source),
        "dino_checkpoint": str(dino_checkpoint),
        "device": args.device,
        "resolution": args.resolution,
        "steps": args.steps,
        "seed": args.seed,
        "data_type": "indoor",
        "offline": True,
        "torch_home": str(torch_home),
    }

    pending: list[tuple[Path, Path, tuple[int, int]]] = []
    for image_path in images:
        prior_path = prior_path_for_image(image_path, output)
        expected_hw = _image_size(image_path)
        if prior_path.is_file():
            try:
                normals = validate_prior_file(prior_path, expected_hw)
            except (OSError, ValueError) as error:
                print(f"REGENERATE {prior_path}: {error}")
            else:
                print(f"SKIP verified {prior_path}")
                relative_source = image_path.relative_to(scene).as_posix()
                _record_manifest(
                    manifest,
                    manifest_path,
                    invocation,
                    {
                        "source_image": relative_source,
                        "output_file": prior_path.name,
                        "shape": list(normals.shape),
                        "dtype": str(normals.dtype),
                        "status": "verified_existing",
                        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                continue
        pending.append((image_path, prior_path, expected_hw))

    if not pending:
        print(f"All {len(images)} requested priors are already valid.")
        return 0

    # Heavy dependencies exist only in the dedicated StableNormal environment.
    import torch

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    torch.backends.cudnn.benchmark = False

    with block_network(), local_torch_hub(torch, stable_source, dino_source, dino_checkpoint):
        print("Loading StableNormal from local source and weights (offline)...")
        predictor = torch.hub.load(
            str(stable_source),
            "StableNormal",
            source="local",
            trust_repo=True,
            local_cache_dir=str(weights),
            device=args.device,
        )
        print(f"Generating {len(pending)} prior(s) in indoor mode...")
        for index, (image_path, prior_path, expected_hw) in enumerate(pending, start=1):
            relative_source = image_path.relative_to(scene).as_posix()
            image_seed = _per_image_seed(args.seed, relative_source)
            torch.manual_seed(image_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(image_seed)

            with Image.open(image_path) as opened:
                # Force RGB so StableNormal's RGBA/object-mask path is never used.
                image = opened.convert("RGB")
                normal_rgb = predictor(
                    image,
                    resolution=args.resolution,
                    match_input_resolution=True,
                    data_type="indoor",
                    num_inference_steps=args.steps,
                )
            if normal_rgb.size != image.size:
                raise ValueError(
                    f"StableNormal output size mismatch for {image_path}: "
                    f"output={normal_rgb.size}, input={image.size}"
                )
            normals = decode_rgb_normal(normal_rgb)
            validate_prior_array(normals, expected_hw)
            atomic_save_npy(prior_path, normals)
            print(f"[{index}/{len(pending)}] WROTE {prior_path} {normals.shape} float32 HWC camera")
            _record_manifest(
                manifest,
                manifest_path,
                invocation,
                {
                    "source_image": relative_source,
                    "output_file": prior_path.name,
                    "shape": list(normals.shape),
                    "dtype": str(normals.dtype),
                    "space": "camera",
                    "layout": "HWC",
                    "status": "generated",
                    "seed": image_seed,
                    "generation_parameters": {
                        "model": "StableNormal",
                        "yoso_version": "yoso-normal-v0-3",
                        "diffusion_version": "stable-normal-v0-1",
                        "data_type": "indoor",
                        "resolution": args.resolution,
                        "steps": args.steps,
                        "device": args.device,
                    },
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OfflineViolation, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
