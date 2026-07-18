#!/usr/bin/env python3
"""Plan or execute audited external DiffusionRenderer inverse inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.diffusion_renderer_raw import (
    PASSES,
    DiffusionRendererIdentityError,
    aggregate_files,
    atomic_json,
    file_identity,
    ordered_scene_images,
    validate_and_manifest,
)


DEFAULT_DR_SOURCE = Path("/home/hanglee/桌面/diffusion-renderer")
DEFAULT_HF_CACHE = Path("/home/hanglee/.cache/huggingface/hub")


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def discover_identity(dr_source: Path, hf_cache: Path, conda_env: str) -> dict:
    dr_source = dr_source.expanduser().resolve()
    if not (dr_source / ".git").exists():
        raise DiffusionRendererIdentityError(f"DiffusionRenderer source is not a Git repository: {dr_source}")
    status = _run_text(["git", "status", "--porcelain=v1"], cwd=dr_source)
    if status:
        raise DiffusionRendererIdentityError("DiffusionRenderer source worktree is dirty")
    head = _run_text(["git", "rev-parse", "HEAD"], cwd=dr_source)
    remote = _run_text(["git", "remote", "get-url", "origin"], cwd=dr_source)
    script = dr_source / "inference_svd_rgbx.py"
    config = dr_source / "configs/rgbx_inference.yaml"
    weights = dr_source / "checkpoints/diffusion_renderer-inverse-svd"
    files = {
        "inverse_model_index": file_identity(weights / "model_index.json"),
        "inverse_unet_config": file_identity(weights / "unet/config.json"),
        "inverse_unet_weights": file_identity(weights / "unet/diffusion_pytorch_model.safetensors"),
        "inverse_vae_config": file_identity(weights / "vae/config.json"),
        "inverse_vae_weights": file_identity(weights / "vae/diffusion_pytorch_model.safetensors"),
        "inverse_scheduler_config": file_identity(weights / "scheduler/scheduler_config.json"),
        "inference_script": file_identity(script),
        "inference_config": file_identity(config),
    }
    hf_model = hf_cache.expanduser().resolve() / "models--stabilityai--stable-video-diffusion-img2vid"
    ref = hf_model / "refs/main"
    if not ref.is_file():
        raise DiffusionRendererIdentityError("Stable Video Diffusion image encoder cache ref is missing")
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = hf_model / "snapshots" / revision
    files.update({
        "svd_image_encoder_config": file_identity(snapshot / "image_encoder/config.json"),
        "svd_image_encoder_weights": file_identity(snapshot / "image_encoder/model.safetensors"),
        "svd_feature_extractor_config": file_identity(snapshot / "feature_extractor/preprocessor_config.json"),
    })
    versions = _run_text([
        "conda", "run", "--no-capture-output", "-n", conda_env, "python", "-c",
        "import json,sys,torch,diffusers,transformers,omegaconf,PIL,numpy;"
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,'diffusers':diffusers.__version__,"
        "'transformers':transformers.__version__,'omegaconf':omegaconf.__version__,"
        "'pillow':PIL.__version__,'numpy':numpy.__version__},sort_keys=True))",
    ])
    try:
        environment = json.loads(versions.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise DiffusionRendererIdentityError(f"cannot identify DiffusionRenderer environment: {versions}") from error
    return {
        "source": {"path": str(dr_source), "head": head, "origin": remote, "clean": True},
        "checkpoint": {
            "repository": "nexuslrf/diffusion_renderer-inverse-svd",
            "path": str(weights),
            "stable_video_diffusion_revision": revision,
            "files": files,
            "aggregate_sha256": aggregate_files(files),
        },
        "environment": {"conda_env": conda_env, "packages": environment},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate raw RGB/normal/depth/basecolor/diffuse-albedo with the verified external DiffusionRenderer. Default is plan-only."
    )
    parser.add_argument("--execute", action="store_true", help="run external model inference; without this flag only print the audited plan")
    parser.add_argument("--scene", type=Path, default=ROOT / "data/Tao")
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", type=Path, default=ROOT / "output/stage_a_tao_dr_raw_112_v1")
    parser.add_argument("--dr-source", type=Path, default=DEFAULT_DR_SOURCE)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--conda-env", default="diffusion_renderer")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames-per-chunk", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-completed-build", type=Path,
        help="package a completed, unmodified build after a validator-only failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing DiffusionRenderer output: {output}")
    images = ordered_scene_images(scene, args.images)
    identity = discover_identity(args.dr_source, args.hf_cache, args.conda_env)
    effective_config = {
        "inference_model_weights": identity["checkpoint"]["path"],
        "inference_res": [args.height, args.width],
        "inference_n_frames": args.frames_per_chunk,
        "overlap_n_frames": 0,
        "inference_n_steps": args.steps,
        "model_passes": list(PASSES),
        "seed": args.seed,
        "chunk_mode": "all",
        "image_group_mode": "folder",
        "save_video": False,
        "save_image": True,
        "autocast": True,
        "weight_dtype": "fp16",
        "decode_chunk_size": 8,
        "cond_mode": "skip",
    }
    build = (
        args.resume_completed_build.expanduser().resolve()
        if args.resume_completed_build else output.parent / f".{output.name}.building-{os.getpid()}"
    )
    raw = build / "raw"
    input_group = build / "input" / scene.name
    command = [
        "conda", "run", "--no-capture-output", "-n", args.conda_env,
        "python", str(args.dr_source.expanduser().resolve() / "inference_svd_rgbx.py"),
        "--config", str(args.dr_source.expanduser().resolve() / "configs/rgbx_inference.yaml"),
        f"inference_input_dir={build / 'input'}",
        f"inference_save_dir={raw}",
        f"inference_model_weights={identity['checkpoint']['path']}",
        f"inference_res=[{args.height},{args.width}]",
        f"inference_n_frames={args.frames_per_chunk}",
        "overlap_n_frames=0", "chunk_mode=all",
        f"inference_n_steps={args.steps}",
        "model_passes=['basecolor','normal','depth','diffuse_albedo']",
        f"seed={args.seed}", "save_video=false", "save_image=true",
    ]
    plan = {
        "schema": "rtgs_diffusion_renderer_generation_plan_v1",
        "execute": bool(args.execute),
        "scene": str(scene),
        "images": args.images,
        "ordered_stems": [path.stem for path in images],
        "count": len(images),
        "output": str(output),
        "identity": identity,
        "effective_config": effective_config,
        "command": command,
        "training": False,
        "resume_completed_build": str(build) if args.resume_completed_build else None,
    }
    print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
    if not args.execute:
        return 0
    if build.exists() and not args.resume_completed_build:
        raise FileExistsError(f"refusing stale DiffusionRenderer build directory: {build}")
    if args.resume_completed_build:
        if not (raw / scene.name).is_dir() or not input_group.is_dir():
            raise FileNotFoundError(f"resume build is incomplete: {build}")
    else:
        input_group.mkdir(parents=True)
        for source in images:
            (input_group / source.name).symlink_to(source)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    try:
        if not args.resume_completed_build:
            subprocess.run(command, cwd=args.dr_source.expanduser().resolve(), env=env, check=True)
        manifest, validation = validate_and_manifest(
            scene=scene, images=args.images, raw_root=raw, group_name=scene.name,
            target_hw=(args.height, args.width), frames_per_chunk=args.frames_per_chunk,
            generation_identity=identity, effective_config=effective_config,
            repository_root=ROOT,
        )
        shutil.rmtree(build / "input")
        (build / "raw").rename(build / "artifacts")
        for record in manifest["frames_and_padding"]:
            record["diffusion_renderer_rgb_file"] = record["diffusion_renderer_rgb_file"].replace(
                f"{scene.name}/", f"artifacts/{scene.name}/", 1
            )
            record["raw_prior_files"] = {
                kind: value.replace(f"{scene.name}/", f"artifacts/{scene.name}/", 1)
                for kind, value in record["raw_prior_files"].items()
            }
        atomic_json(build / "generation_plan.json", plan)
        atomic_json(build / "manifest.json", manifest)
        atomic_json(build / "validation_summary.json", validation)
        os.replace(build, output)
    except Exception:
        raise
    print(f"DIFFUSION_RENDERER_RAW=PASS output={output} real={len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
