"""Stage D D/R/T training loop with one immutable geometry release."""

from __future__ import annotations

import json
import math
import os
import time
import gc
import random
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from plyfile import PlyData
from tqdm import tqdm

from gaussian_renderer.transmittance_renderer import (
    StageDRenderState, build_static_dr_inputs, render, render_from_static_dr,
)
from geometry.geometry_release import (
    GeometryRelease, camera_identity_sha256, sha256_file, validate_geometry_release,
)
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_d_scene import StageDScene
from scene.stage_d_state import (
    STAGE_D_FORMAT, initialize_stage_d_from_stage_b, load_stage_d_checkpoint,
    make_stage_d_checkpoint,
)
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from utils.loss_utils import (
    l1_loss, monocular_normal_loss, normal_depth_consistency_loss,
    specular_constraint_loss, ssim,
)
from utils.specular_mask import validate_specular_mask_set
from utils.surfel_utils import camera_normals_to_world, face_forward
from utils.training_state import (
    capture_rng_state, make_camera_runtime_state, restore_camera_deck,
    restore_rng_state,
)
from utils.transmittance_debug import make_stage_d_contact_sheet, save_transmittance_debug_maps
from utils.stage_d_static_cache import (
    TRAINING_PACKAGE_KEYS, StaticDRCache, cpu_cache_payload, frozen_branch_hash,
    make_identity, renderer_contract, write_manifest,
)

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


FORMAL_SOURCE_SHA256 = "050500d607e1910ca088049ae73619949ad183e23c85354a8408bb29571fbe84"
FORMAL_RELEASE_ID = "stage_c_geometry_release_v1"
FORMAL_RELEASE_SHA256 = "4fedb22dc2f2e6415a3d3948ab26fba54df91ba06b66d951a09b3dc5f761188d"
FORMAL_OUTPUT_NAME = "stage_d_tihubird_c03r8_formal_onset_g15000_g20000_v2"
FORMAL_NODES = (15100, 15500, 16000, 17500, 20000)
FORMAL_STEMS = ("000000", "000012", "000039", "000040", "000041", "000053", "000063", "000083", "000110")
FORMAL_RAY_CHUNKS = (2048, 1024, 512)
FORMAL_PRESSURE_FREE_BYTES = 2 * 1024**3
CACHED_OUTPUT_NAME = "stage_d_tihubird_c03r8_cached_twarmup_then_joint_g15000_g20000_v2"
CACHED_PHASE_A_END = 18000
CACHED_NODES = (15025, 15100, 15500, 16000, 17500, 18000, 19000, 20000)
CACHED_STEMS = FORMAL_STEMS
CACHED_RANDOM_VIEW_SEED = 20260703


def _mesh_bounds(path: Path, device):
    vertex = PlyData.read(path)["vertex"].data
    xyz = torch.tensor(
        [[row["x"], row["y"], row["z"]] for row in vertex],
        dtype=torch.float32, device=device,
    )
    return xyz.amin(dim=0), xyz.amax(dim=0)


def _erode_hard_mask(mask: torch.Tensor, iterations: int = 2) -> torch.Tensor:
    hard = (mask >= 0.5).float().unsqueeze(0)
    eroded = hard
    for _ in range(iterations):
        eroded = 1.0 - F.max_pool2d(1.0 - eroded, kernel_size=3, stride=1, padding=1)
    return eroded.squeeze(0).permute(1, 2, 0)


def _finite_models(models):
    counts, labels, checks = {}, [], []
    for name, model in models.items():
        counts[name] = 0
        for key, value in model.__dict__.items():
            if torch.is_tensor(value):
                counts[name] += value.numel()
                labels.append(f"{name}.{key}")
                checks.append(torch.isfinite(value).all())
                if value.grad is not None:
                    labels.append(f"{name}.{key}.grad")
                    checks.append(torch.isfinite(value.grad).all())
    if checks:
        finite = torch.stack(checks)
        if not bool(finite.all()):
            values = finite.detach().cpu().tolist()
            failed = [label for label, passed in zip(labels, values) if not passed]
            raise FloatingPointError(f"Stage D model state contains NaN/Inf: {failed}")
    return {name: int(count) for name, count in counts.items()}


def _config(dataset, opt, release, source):
    return {
        "stage": "stage_d", "model_type": "surfel",
        "experiment": dataset.experiment, "resolution": int(dataset.resolution),
        "geometry_release_manifest": str(Path(dataset.geometry_release_manifest).resolve()),
        "geometry_release_id": release.manifest["geometry_release_id"],
        "geometry_release_aggregate_sha256": release.validation["aggregate_sha256"],
        "source_stage_b_checkpoint_sha256": source["sha256"],
        "ray_chunk_size": int(dataset.ray_chunk_size),
        "ray_cutoff_sigma": float(dataset.ray_cutoff_sigma),
        "ray_hit_threshold": float(dataset.ray_hit_threshold),
        "ray_epsilon_scale": float(dataset.ray_epsilon_scale),
        "roughness_min": float(dataset.roughness_min),
        "roughness_remap": bool(dataset.roughness_remap),
        "bsdf_weight_mode": "brdf_times_cosine",
        "transmittance_compose": dataset.transmittance_compose,
        "lambda_norm": float(opt.lambda_norm), "lambda_mono": float(opt.lambda_mono),
        "lambda_perc": float(opt.lambda_perc), "lambda_spec": float(opt.lambda_spec),
        "specular_k0": float(opt.specular_k0), "lambda_depth": float(opt.lambda_depth),
        "stage_d_depth_start_iteration": int(opt.stage_d_depth_start_iteration),
        "stage_d_formal_onset": bool(opt.stage_d_formal_onset),
        "stage_d_cached_twarmup": bool(opt.stage_d_cached_twarmup),
        "cached_t_warmup": (
            {
                "phase_a_global": [15001, CACHED_PHASE_A_END],
                "phase_a_updates": {"diffuse": 0, "reflection": 0, "transmittance": 3000},
                "phase_b_global": [CACHED_PHASE_A_END + 1, 20000],
                "phase_b_mode": "exact_uncached_joint_d_r_t",
                "reflection_local_semantics": (
                    "R-local remains 12000 in Phase A and advances only for Phase-B optimizer updates"
                ),
                "cache_schema": "rtgs_stage_d_static_dr_cache_v1",
                "cache_fp": "float32",
                "cache_contains_target_rgb": False,
            }
            if opt.stage_d_cached_twarmup else None
        ),
        "transmittance_initialization": {
            "mode": dataset.transmittance_init_mode,
            "count": int(dataset.transmittance_init_count),
            "seed": int(dataset.transmittance_init_seed),
        },
        "formal_memory_policy": (
            {
                "name": "checkpointed_2048_oom_fallback_pressure_cache_v1",
                "checkpoint_chunks": True,
                "attempt_chunk_sizes": list(FORMAL_RAY_CHUNKS),
                "pressure_release_free_bytes": FORMAL_PRESSURE_FREE_BYTES,
                "retry_boundary": "before optimizer topology checkpoint telemetry commit",
            }
            if (opt.stage_d_formal_onset or opt.stage_d_cached_twarmup) else None
        ),
        "source_stage_b_contract": {
            "lambda_spec": source["stage_b_config"].get("lambda_spec"),
            "specular_k0": source["stage_b_config"].get("specular_k0"),
            "specular_mask": source["stage_b_config"].get("specular_mask"),
            "global_iteration": source.get("global_iteration"),
            "reflection_iteration": source.get("reflection_iteration"),
        },
        "transmittance_schedule": {
            "position_lr_init": float(opt.transmittance_position_lr_init),
            "position_lr_final": float(opt.transmittance_position_lr_final),
            "position_lr_delay_mult": float(opt.transmittance_position_lr_delay_mult),
            "position_lr_max_steps": int(opt.transmittance_position_lr_max_steps),
            "color_lr": float(opt.transmittance_color_lr),
            "opacity_lr": float(opt.transmittance_opacity_lr),
            "scaling_lr": float(opt.transmittance_scaling_lr),
            "rotation_lr": float(opt.transmittance_rotation_lr),
            "percent_dense": float(opt.transmittance_percent_dense),
            "densify_from_iter": int(opt.transmittance_densify_from_iter),
            "densify_until_iter": int(opt.transmittance_densify_until_iter),
            "densification_interval": int(opt.transmittance_densification_interval),
            "densify_grad_threshold": float(opt.transmittance_densify_grad_threshold),
        },
    }


def _validate_args(dataset, opt, start_checkpoint):
    if dataset.stage != "stage_d" or dataset.model_type != "surfel":
        raise ValueError("Stage D requires --stage stage_d --model_type surfel")
    if not start_checkpoint:
        raise ValueError("Stage D requires --start_checkpoint")
    if not dataset.geometry_release_manifest:
        raise ValueError("Stage D requires --geometry_release_manifest")
    if dataset.transmittance_init_mode != "random_bbox":
        raise ValueError("Stage D currently requires transmittance_init_mode=random_bbox")
    if dataset.transmittance_init_count <= 0:
        raise ValueError("transmittance_init_count must be positive")
    if dataset.transmittance_compose != "alpha_over":
        raise ValueError("Stage D v1 implements only transmittance_compose=alpha_over")
    if dataset.ray_background != "scene" or opt.random_background:
        raise ValueError("Stage D requires deterministic scene ray background")
    if not dataset.specular_masks or opt.lambda_spec <= 0:
        raise ValueError("Stage D requires the formal specular mask and positive L_spec")
    if opt.lambda_depth < 0:
        raise ValueError("lambda_depth must be non-negative")
    if opt.stage_d_smoke_max_steps <= 0:
        raise ValueError("stage_d_smoke_max_steps must be positive")
    if opt.stage_d_smoke and opt.stage_d_formal_onset:
        raise ValueError("formal Stage D onset must not use --stage_d_smoke")
    if opt.stage_d_cached_twarmup and (opt.stage_d_smoke or opt.stage_d_formal_onset):
        raise ValueError("cached T warm-up is a distinct formal mode")
    if int(opt.stage_d_phase_a_end_iteration) != CACHED_PHASE_A_END:
        raise ValueError("cached T warm-up Phase A must end at global 18000")
    if opt.stage_d_cache_parity_atol <= 0 or opt.stage_d_cache_parity_mean_atol <= 0:
        raise ValueError("cached T warm-up parity tolerances must be positive")


def _validate_formal_contract(dataset, opt, release, source, fresh_from_stage_b,
                              saving_iterations, checkpoint_iterations):
    if not opt.stage_d_formal_onset:
        return
    failures = []
    required = {
        "fresh_from_stage_b": fresh_from_stage_b,
        "source_sha256": source.get("sha256") == FORMAL_SOURCE_SHA256,
        "source_global": source.get("global_iteration") == 15000,
        "source_r_local": source.get("reflection_iteration") == 12000,
        "release_id": release.manifest.get("geometry_release_id") == FORMAL_RELEASE_ID,
        "release_sha256": release.validation.get("aggregate_sha256") == FORMAL_RELEASE_SHA256,
        "endpoint": int(opt.iterations) == 20000,
        "depth_start": int(opt.stage_d_depth_start_iteration) == 40000,
        "lambda_depth": float(opt.lambda_depth) == 0.2,
        "lambda_spec": float(opt.lambda_spec) == 0.2,
        "specular_k0": float(opt.specular_k0) == 0.9,
        "resolution": int(dataset.resolution) == 8,
        "ray_chunk_size": int(dataset.ray_chunk_size) == FORMAL_RAY_CHUNKS[0],
        "t_init": (
            dataset.transmittance_init_mode == "random_bbox"
            and int(dataset.transmittance_init_count) == 4096
            and int(dataset.transmittance_init_seed) == 20260703
        ),
        "output": Path(dataset.model_path).name == FORMAL_OUTPUT_NAME,
        "checkpoint_nodes": tuple(sorted(set(checkpoint_iterations))) == FORMAL_NODES,
        "ply_nodes": tuple(sorted(set(saving_iterations))) == FORMAL_NODES,
    }
    source_config = source.get("stage_b_config", {})
    source_mask = source_config.get("specular_mask", {})
    current_mask = getattr(dataset, "_validated_specular_mask_manifest", {})
    required["source_specular_contract"] = (
        source_config.get("lambda_spec") == 0.2
        and source_config.get("specular_k0") == 0.9
        and source_mask.get("manifest_file_sha256")
        == "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
        and current_mask.get("manifest_file_sha256")
        == source_mask.get("manifest_file_sha256")
        and current_mask.get("aggregate_sha256")
        == source_mask.get("aggregate_sha256")
    )
    for name, passed in required.items():
        if not passed:
            failures.append(name)
    if failures:
        raise ValueError("formal Stage D onset contract mismatch: " + ", ".join(failures))


def _validate_cached_contract(dataset, opt, release, source, fresh_from_stage_b,
                              saving_iterations, checkpoint_iterations):
    if not opt.stage_d_cached_twarmup:
        return
    source_config = source.get("stage_b_config", {})
    source_mask = source_config.get("specular_mask", {})
    current_mask = getattr(dataset, "_validated_specular_mask_manifest", {})
    required = {
        "fresh_from_stage_b": fresh_from_stage_b,
        "source_sha256": source.get("sha256") == FORMAL_SOURCE_SHA256,
        "source_global": source.get("global_iteration") == 15000,
        "source_r_local": source.get("reflection_iteration") == 12000,
        "release_id": release.manifest.get("geometry_release_id") == FORMAL_RELEASE_ID,
        "release_sha256": release.validation.get("aggregate_sha256") == FORMAL_RELEASE_SHA256,
        "endpoint": int(opt.iterations) == 20000,
        "phase_a_end": int(opt.stage_d_phase_a_end_iteration) == CACHED_PHASE_A_END,
        "depth_start": int(opt.stage_d_depth_start_iteration) == 40000,
        "lambda_depth": float(opt.lambda_depth) == 0.2,
        "lambda_spec": float(opt.lambda_spec) == 0.2,
        "specular_k0": float(opt.specular_k0) == 0.9,
        "resolution": int(dataset.resolution) == 8,
        "ray_chunk_size": int(dataset.ray_chunk_size) == 2048,
        "t_init": (
            dataset.transmittance_init_mode == "random_bbox"
            and int(dataset.transmittance_init_count) == 4096
            and int(dataset.transmittance_init_seed) == 20260703
        ),
        "output": Path(dataset.model_path).name == CACHED_OUTPUT_NAME,
        "checkpoint_nodes": tuple(sorted(set(checkpoint_iterations))) == CACHED_NODES,
        "ply_nodes": tuple(sorted(set(saving_iterations))) == CACHED_NODES,
        "source_specular_contract": (
            source_config.get("lambda_spec") == 0.2
            and source_config.get("specular_k0") == 0.9
            and source_mask.get("manifest_file_sha256")
            == "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
            and current_mask.get("manifest_file_sha256")
            == source_mask.get("manifest_file_sha256")
            and current_mask.get("aggregate_sha256")
            == source_mask.get("aggregate_sha256")
        ),
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise ValueError("cached T warm-up contract mismatch: " + ", ".join(failures))


def _phase_for_iteration(iteration):
    return "cached_t_warmup" if int(iteration) <= CACHED_PHASE_A_END else "exact_joint"


def _set_branch_trainable(model, enabled):
    for name in (
        "_xyz", "_rotation", "_scaling", "_opacity", "_base_color",
        "_roughness", "_f0", "_ks", "_exposure", "_color",
    ):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and value.is_floating_point():
            value.requires_grad_(bool(enabled))


def _render_formal_review_node(
    scene, state, pipe, background, release, iteration,
    stems=FORMAL_STEMS, static_cache=None,
):
    cameras = {
        Path(str(camera.image_name)).stem: camera
        for camera in (scene.getTestCameras() or scene.getTrainCameras())
    }
    missing = sorted(set(stems) - set(cameras))
    if missing:
        raise ValueError(f"formal review cameras are missing: {missing}")
    iteration_directory = os.path.join(scene.model_path, "debug", f"iteration_{iteration:06d}")
    prior_checkpoint_mode = state.ray_checkpoint_chunks
    prior_chunk_size = state.ray_chunk_size
    state.ray_checkpoint_chunks = False
    state.ray_chunk_size = 512
    try:
        for stem in stems:
            camera = cameras[stem]
            if static_cache is not None:
                static_inputs = static_cache.load(stem)
                debug = render_from_static_dr(
                    state, background, static_inputs,
                    return_ray_aux=False, return_ray_diagnostics=True,
                )
            else:
                debug = render(
                    camera, state, pipe, background,
                    return_ray_aux=False, return_ray_diagnostics=True,
                )
            directory = os.path.join(iteration_directory, stem)
            save_transmittance_debug_maps(
                debug, camera.original_image.cuda(), directory,
                camera.specular_mask, camera.specular_mask_sha256,
                release.manifest["geometry_release_id"],
                release.validation["aggregate_sha256"],
            )
            del debug
            if static_cache is not None:
                del static_inputs
        make_stage_d_contact_sheet(iteration_directory, stems)
    finally:
        state.ray_checkpoint_chunks = prior_checkpoint_mode
        state.ray_chunk_size = prior_chunk_size
    return iteration_directory


def _forward_backward_stage_d(
    camera, state, pipe, background, opt, perceptual, iteration,
    static_inputs=None,
):
    with record_function("stage_d.render_forward"):
        if static_inputs is None:
            package = render(camera, state, pipe, background, return_ray_aux=True)
        else:
            package = render_from_static_dr(
                state, background, static_inputs, return_ray_aux=True,
            )
    with record_function("stage_d.loss_forward"):
        image, gt = package["render"], camera.original_image.cuda()
        l1_value = l1_loss(image, gt)
        ssim_value = (
            fused_ssim(image.unsqueeze(0), gt.unsqueeze(0))
            if FUSED_SSIM_AVAILABLE else ssim(image, gt)
        )
        rgb_loss = (1.0 - opt.lambda_dssim) * l1_value + opt.lambda_dssim * (1.0 - ssim_value)
        normal_loss = normal_depth_consistency_loss(
            package["normal"], package["position"], package["alpha"]
        )
        mono_loss = image.new_zeros(())
        if camera.normal_prior is not None:
            target = camera.normal_prior.permute(1, 2, 0)
            if camera.normal_prior_space == "camera":
                target = camera_normals_to_world(camera, target)
            target = face_forward(target, package["position"], camera.camera_center)
            mono_loss = monocular_normal_loss(
                package["normal"], target, package["alpha"],
                camera.normal_prior_valid.permute(1, 2, 0),
            )
        perceptual_loss = image.new_zeros(()) if perceptual is None else perceptual(image, gt)
        specular_loss = specular_constraint_loss(
            package["surface_ks"], camera.specular_mask, opt.specular_k0
        )
        eroded = _erode_hard_mask(camera.specular_mask)
        depth_domain = eroded * package["two_hit_valid"]
        depth_loss = (
            (package["depth_violation"] * depth_domain).sum()
            / depth_domain.sum().clamp_min(1.0)
        )
        depth_enabled = int(iteration) >= opt.stage_d_depth_start_iteration
        loss = (
            rgb_loss + opt.lambda_norm * normal_loss + opt.lambda_mono * mono_loss
            + opt.lambda_perc * perceptual_loss + opt.lambda_spec * specular_loss
            + (opt.lambda_depth * depth_loss if depth_enabled else 0.0)
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Stage D total loss is NaN/Inf")
    with record_function("stage_d.backward"):
        loss.backward()
    return {
        "package": package, "image": image, "gt": gt, "loss": loss,
        "l1": l1_value, "ssim": ssim_value, "rgb": rgb_loss,
        "normal": normal_loss, "mono": mono_loss, "perceptual": perceptual_loss,
        "specular": specular_loss, "depth": depth_loss,
        "depth_enabled": bool(depth_enabled),
    }


def _zero_stage_d_gradients(diffuse, reflection, transmittance):
    diffuse.exposure_optimizer.zero_grad(set_to_none=True)
    diffuse.optimizer.zero_grad(set_to_none=True)
    reflection.optimizer.zero_grad(set_to_none=True)
    transmittance.optimizer.zero_grad(set_to_none=True)


def _allocator_cache_under_pressure(device_free_bytes):
    return int(device_free_bytes) < FORMAL_PRESSURE_FREE_BYTES


def _forward_backward_with_memory_retry(
    camera, state, pipe, background, opt, perceptual,
    diffuse, reflection, transmittance, iteration, static_inputs=None,
):
    chunks = (
        FORMAL_RAY_CHUNKS
        if (opt.stage_d_formal_onset or opt.stage_d_cached_twarmup)
        else (state.ray_chunk_size,)
    )
    retry_records = []
    for attempt_index, chunk_size in enumerate(chunks):
        state.ray_chunk_size = int(chunk_size)
        state.ray_checkpoint_chunks = True
        try:
            forward_args = (
                {} if static_inputs is None else {"static_inputs": static_inputs}
            )
            payload = _forward_backward_stage_d(
                camera, state, pipe, background, opt, perceptual, iteration,
                **forward_args,
            )
            return payload, retry_records, int(chunk_size)
        except torch.cuda.OutOfMemoryError:
            retry_records.append({
                "attempt": attempt_index + 1, "chunk_size": int(chunk_size),
                "checkpoint_chunks": True,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            })
            _zero_stage_d_gradients(diffuse, reflection, transmittance)
            gc.collect()
            torch.cuda.empty_cache()
            if attempt_index + 1 == len(chunks):
                raise
            print("STAGE_D_MEMORY_RETRY " + json.dumps({
                "camera": str(camera.image_name), **retry_records[-1],
                "next_chunk_size": int(chunks[attempt_index + 1]),
            }, sort_keys=True))
    raise AssertionError("unreachable Stage D memory retry state")


def _write_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _cacheable_static_inputs(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            cached = _cacheable_static_inputs(child)
            if cached is not None:
                result[key] = cached
        return result
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _release_camera_identities(release, cameras):
    expected = {
        row["stem"]: row["camera_identity_sha256"]
        for row in release.manifest["cache_entries"]
    }
    observed = {Path(str(camera.image_name)).stem for camera in cameras}
    if observed != set(expected):
        raise ValueError("training cameras do not exactly match the geometry-release camera set")
    for camera in cameras:
        stem = Path(str(camera.image_name)).stem
        cache = release.load_view(stem)
        if tuple(cache["valid_two_hit"].shape) != (
            int(camera.image_height), int(camera.image_width),
        ):
            raise ValueError(f"runtime camera resolution mismatch: {stem}")
        runtime_identity = camera_identity_sha256(stem, {
            "depth": np.empty((int(camera.image_height), int(camera.image_width)), np.float32),
            "world_view_transform": camera.world_view_transform.detach().cpu().numpy(),
            "full_proj_transform": camera.full_proj_transform.detach().cpu().numpy(),
            "camera_center": camera.camera_center.detach().cpu().numpy(),
        })
        if runtime_identity != expected[stem]:
            raise ValueError(f"runtime camera identity mismatch: {stem}")
    return expected


@torch.no_grad()
def _build_static_dr_cache(
    directory, cameras, state, pipe, background, identity, camera_identities,
):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite static D/R cache: {directory}")
    directory.mkdir(parents=True)
    entries = []
    for camera in sorted(cameras, key=lambda item: Path(str(item.image_name)).stem):
        stem = Path(str(camera.image_name)).stem
        static_inputs = build_static_dr_inputs(
            camera, state, pipe, background, return_ray_aux=False,
            return_ray_diagnostics=stem in CACHED_STEMS,
        )
        cacheable = _cacheable_static_inputs(static_inputs)
        if stem not in CACHED_STEMS:
            cacheable["package"] = {
                key: value for key, value in cacheable["package"].items()
                if key in TRAINING_PACKAGE_KEYS
            }
        payload = cpu_cache_payload(
            cacheable, identity, stem,
            camera_identities[stem],
        )
        relative = Path("views") / f"{stem}.pth"
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".pth.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        entries.append({
            "camera_stem": stem, "camera_identity_sha256": camera_identities[stem],
            "relative_path": relative.as_posix(), "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        })
        del static_inputs, cacheable, payload
    write_manifest(directory, identity, entries)
    return StaticDRCache(directory, identity, camera_identities)


def _absolute_difference(first, second):
    difference = torch.abs(first.detach().float() - second.detach().float())
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
    }


def _t_gradients(transmittance):
    values = {
        "xyz": transmittance._xyz, "rotation": transmittance._rotation,
        "scale": transmittance._scaling, "opacity": transmittance._opacity,
        "color": transmittance._color,
    }
    return {
        name: (
            torch.zeros_like(value) if value.grad is None else value.grad.detach().clone()
        )
        for name, value in values.items()
    }


def _parity_snapshot(payload, gradients):
    package = payload["package"]
    return {
        "outputs": {
            "final_rgb": package["final"].detach().clone(),
            "Cin": package["inside_color"].detach().clone(),
            "Ain": package["inside_alpha"].detach().clone(),
            "Din": package["inside_depth"].detach().clone(),
            "Ct": package["transmittance_color"].detach().clone(),
            "At": package["transmittance_alpha"].detach().clone(),
        },
        "losses": {
            name: payload[name].detach().reshape(1).clone()
            for name in (
                "loss", "l1", "ssim", "rgb", "normal", "mono",
                "perceptual", "specular", "depth",
            )
        },
        "gradients": gradients,
    }


def _run_cache_parity(
    target, cameras, static_cache, state, pipe, background, opt, perceptual,
    diffuse, reflection, transmittance,
):
    by_stem = {Path(str(camera.image_name)).stem: camera for camera in cameras}
    remaining = sorted(set(by_stem) - set(CACHED_STEMS))
    if not remaining:
        raise ValueError("cache parity requires a random view outside the fixed nine")
    random_stem = random.Random(CACHED_RANDOM_VIEW_SEED).choice(remaining)
    stems = list(CACHED_STEMS) + [random_stem]
    rows, failures = [], []
    for stem in stems:
        camera = by_stem[stem]
        _zero_stage_d_gradients(diffuse, reflection, transmittance)
        uncached_payload = _forward_backward_stage_d(
            camera, state, pipe, background, opt, perceptual, 15001,
        )
        uncached = _parity_snapshot(uncached_payload, _t_gradients(transmittance))
        _zero_stage_d_gradients(diffuse, reflection, transmittance)
        static_inputs = static_cache.load(stem, training_only=True)
        cached_payload = _forward_backward_stage_d(
            camera, state, pipe, background, opt, perceptual, 15001,
            static_inputs=static_inputs,
        )
        cached = _parity_snapshot(cached_payload, _t_gradients(transmittance))
        comparisons = {}
        for group in ("outputs", "losses", "gradients"):
            comparisons[group] = {
                name: _absolute_difference(uncached[group][name], cached[group][name])
                for name in uncached[group]
            }
            for name, difference in comparisons[group].items():
                if (
                    difference["max_abs"] > float(opt.stage_d_cache_parity_atol)
                    or difference["mean_abs"] > float(opt.stage_d_cache_parity_mean_atol)
                ):
                    failures.append(f"{stem}:{group}:{name}")
        rows.append({"camera_stem": stem, "comparisons": comparisons})
        _zero_stage_d_gradients(diffuse, reflection, transmittance)
        del uncached_payload, cached_payload, uncached, cached, static_inputs
    report = {
        "schema": "rtgs_stage_d_cache_parity_v1",
        "fixed_camera_stems": list(CACHED_STEMS),
        "deterministic_random_camera_stem": random_stem,
        "tolerances": {
            "max_absolute": float(opt.stage_d_cache_parity_atol),
            "mean_absolute": float(opt.stage_d_cache_parity_mean_atol),
        },
        "rows": rows, "failures": failures,
        "status": "PASS" if not failures else "CACHED_T_WARMUP_BLOCKED",
    }
    _atomic_json(Path(target), report)
    if failures:
        raise RuntimeError("static D/R cache parity failed: " + ", ".join(failures))
    return report


def _run_cache_performance_benchmark(
    target, camera, static_cache, state, pipe, background, opt, perceptual,
    diffuse, reflection, transmittance,
):
    stem = Path(str(camera.image_name)).stem
    results = {}
    for mode in ("uncached_frozen_dr", "cached_t_only"):
        samples = []
        for _ in range(3):
            _zero_stage_d_gradients(diffuse, reflection, transmittance)
            static_inputs = (
                static_cache.load(stem, training_only=True)
                if mode == "cached_t_only" else None
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            payload = _forward_backward_stage_d(
                camera, state, pipe, background, opt, perceptual, 15001,
                static_inputs=static_inputs,
            )
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            del payload, static_inputs
        results[mode] = {
            "samples_ms": samples,
            "median_ms": float(sorted(samples)[len(samples) // 2]),
        }
    _zero_stage_d_gradients(diffuse, reflection, transmittance)
    report = {
        "schema": "rtgs_stage_d_cached_t_performance_v1",
        "optimizer_updates": 0, "camera_stem": stem,
        "results": results,
        "speedup_median": (
            results["uncached_frozen_dr"]["median_ms"]
            / max(results["cached_t_only"]["median_ms"], 1e-12)
        ),
    }
    _atomic_json(Path(target), report)
    return report


def training_stage_d(
    dataset, opt, pipe, testing_iterations, saving_iterations,
    checkpoint_iterations, start_checkpoint, debug_from, prepare_output_and_logger,
):
    del testing_iterations, debug_from
    _validate_args(dataset, opt, start_checkpoint)
    release = GeometryRelease(Path(dataset.geometry_release_manifest))
    mask_manifest = validate_specular_mask_set(
        dataset.source_path, dataset.images, dataset.specular_masks
    )
    dataset._validated_specular_mask_manifest = mask_manifest
    writer = prepare_output_and_logger(dataset)
    diffuse = DiffuseSurfelModel(dataset.roughness_min, opt.optimizer_type)
    reflection = ReflectionSurfelModel()
    transmittance = TransmittanceSurfelModel()
    scene = StageDScene(dataset, diffuse, reflection, transmittance, shuffle=True)

    checkpoint_header = torch.load(start_checkpoint, map_location="cpu")
    checkpoint_format = checkpoint_header.get("format")
    fresh_from_stage_b = checkpoint_format == "rtgs_stage_b"
    restored_rng_state = None
    del checkpoint_header
    if fresh_from_stage_b:
        bbox_min, bbox_max = _mesh_bounds(
            release.root / release.manifest["glass_mesh_relative_path"], "cuda"
        )
        (
            global_iteration, reflection_iteration, transmittance_iteration,
            source, runtime_state,
        ) = initialize_stage_d_from_stage_b(
            start_checkpoint, diffuse, reflection, transmittance, opt, opt, opt,
            bbox_min, bbox_max, dataset.transmittance_init_count,
            dataset.transmittance_init_seed, map_location="cuda",
        )
        restored_rng_state = capture_rng_state()
    elif checkpoint_format == STAGE_D_FORMAT:
        (
            global_iteration, reflection_iteration, transmittance_iteration,
            source, saved_config, runtime_state,
        ) = load_stage_d_checkpoint(
            start_checkpoint, diffuse, reflection, transmittance, opt, opt, opt,
            release.manifest["geometry_release_id"],
            release.validation["aggregate_sha256"], map_location="cuda",
        )
        restored_rng_state = capture_rng_state()
        current = _config(dataset, opt, release, source)
        for key in (
            "geometry_release_id", "geometry_release_aggregate_sha256",
            "source_stage_b_checkpoint_sha256", "ray_cutoff_sigma",
            "ray_hit_threshold", "ray_epsilon_scale", "roughness_min",
            "roughness_remap", "bsdf_weight_mode", "transmittance_compose",
            "transmittance_schedule",
        ):
            if saved_config.get(key) != current.get(key):
                raise ValueError(f"Stage D resume config mismatch for {key}")
    else:
        raise ValueError("Stage D start checkpoint must be rtgs_stage_b or rtgs_stage_d")
    if global_iteration >= opt.iterations:
        raise ValueError("--iterations must exceed the restored global iteration")
    added_steps = int(opt.iterations - global_iteration)
    if opt.stage_d_smoke and added_steps > opt.stage_d_smoke_max_steps:
        raise ValueError("Stage D smoke exceeds its fail-closed step bound")
    config = _config(dataset, opt, release, source)
    _validate_formal_contract(
        dataset, opt, release, source, fresh_from_stage_b,
        saving_iterations, checkpoint_iterations,
    )
    _validate_cached_contract(
        dataset, opt, release, source, fresh_from_stage_b,
        saving_iterations, checkpoint_iterations,
    )
    if opt.stage_d_formal_onset:
        metadata = {
            "schema": "rtgs_stage_d_formal_onset_run_v2",
            "source": source,
            "config": config,
            "actual_transmittance_initialization": dict(transmittance.initialization),
            "required_nodes": list(FORMAL_NODES),
            "review_stems": list(FORMAL_STEMS),
            "first_update": 15001,
            "last_update": 20000,
            "future_depth_activation_global": 40000,
            "depth_enabled_during_run": False,
        }
        Path(scene.model_path, "formal_run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    elif opt.stage_d_cached_twarmup:
        metadata = {
            "schema": "rtgs_stage_d_cached_twarmup_then_joint_v1",
            "source": source, "config": config,
            "actual_transmittance_initialization": dict(transmittance.initialization),
            "required_nodes": list(CACHED_NODES), "review_stems": list(CACHED_STEMS),
            "phase_a": {
                "global": [15001, 18000], "mode": "cached_t_only",
                "diffuse_optimizer_updates": 0, "reflection_optimizer_updates": 0,
                "transmittance_optimizer_updates": 3000,
            },
            "phase_b": {
                "global": [18001, 20000], "mode": "exact_uncached_joint_d_r_t",
                "static_dr_cache_enabled": False,
            },
            "reflection_local_semantics": (
                "R-local is held at 12000 in Phase A and reaches 14000 after "
                "the 2000 exact Phase-B joint updates; Phase A is not full joint training."
            ),
            "future_depth_activation_global": 40000,
            "depth_enabled_during_run": False,
        }
        _atomic_json(Path(scene.model_path, "cached_twarmup_run_metadata.json"), metadata)

    state = StageDRenderState(
        diffuse=diffuse, reflection=reflection, transmittance=transmittance,
        geometry_release=release, scene_radius=scene.cameras_extent,
        ray_chunk_size=dataset.ray_chunk_size,
        ray_cutoff_sigma=dataset.ray_cutoff_sigma,
        ray_hit_threshold=dataset.ray_hit_threshold,
        ray_epsilon_scale=dataset.ray_epsilon_scale,
        material_alpha_threshold=dataset.material_alpha_threshold,
        roughness_min=diffuse.roughness_min,
        roughness_remap=dataset.roughness_remap,
        ray_checkpoint_chunks=True,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )
    perceptual = None
    if opt.lambda_perc > 0:
        from utils.perceptual_loss import VGG16PerceptualLoss
        perceptual = VGG16PerceptualLoss(pretrained=True).cuda().eval()
    if restored_rng_state is not None:
        # Process-local scene/helper construction is outside the resumed trajectory.
        restore_rng_state(restored_rng_state)

    cameras = scene.getTrainCameras().copy()
    static_cache = None
    frozen_hash_before = frozen_hash_after = None
    if opt.stage_d_cached_twarmup:
        _zero_stage_d_gradients(diffuse, reflection, transmittance)
        _set_branch_trainable(diffuse, False)
        _set_branch_trainable(reflection, False)
        frozen_hash_before = {
            "diffuse": frozen_branch_hash(diffuse),
            "reflection": frozen_branch_hash(reflection),
        }
        camera_identities = _release_camera_identities(release, cameras)
        renderer_config = renderer_contract(dataset)
        cache_identity = make_identity(
            source["sha256"], release.validation["aggregate_sha256"],
            mask_manifest["manifest_file_sha256"], renderer_config,
        )
        static_cache = _build_static_dr_cache(
            Path(scene.model_path) / "static_dr_cache", cameras, state, pipe,
            background, cache_identity, camera_identities,
        )
        config["cached_t_warmup"].update({
            "cache_identity_sha256": cache_identity["identity_sha256"],
            "cache_aggregate_sha256": static_cache.aggregate_sha256,
            "mask_manifest_sha256": mask_manifest["manifest_file_sha256"],
            "renderer_config_sha256": cache_identity["renderer_config_sha256"],
            "phase_a_frozen_hash_before": frozen_hash_before,
        })
        _run_cache_parity(
            Path(scene.model_path) / "cache_parity_report.json", cameras,
            static_cache, state, pipe, background, opt, perceptual,
            diffuse, reflection, transmittance,
        )
        benchmark_camera = next(
            camera for camera in cameras
            if Path(str(camera.image_name)).stem == "000018"
        )
        _run_cache_performance_benchmark(
            Path(scene.model_path) / "cache_performance_benchmark.json",
            benchmark_camera, static_cache, state, pipe, background, opt,
            perceptual, diffuse, reflection, transmittance,
        )
        frozen_hash_after_preflight = {
            "diffuse": frozen_branch_hash(diffuse),
            "reflection": frozen_branch_hash(reflection),
        }
        if frozen_hash_after_preflight != frozen_hash_before:
            raise RuntimeError("CACHED_T_WARMUP_BLOCKED: D/R changed during cache preflight")
        metadata["cache_identity"] = cache_identity
        metadata["cache_aggregate_sha256"] = static_cache.aggregate_sha256
        metadata["phase_a_frozen_hash_before"] = frozen_hash_before
        _atomic_json(Path(scene.model_path, "cached_twarmup_run_metadata.json"), metadata)
        restore_rng_state(restored_rng_state)
    viewpoints, camera_indices = restore_camera_deck(cameras, runtime_state)
    progress = tqdm(range(global_iteration, opt.iterations), desc="Stage D training progress")
    telemetry_path = Path(
        opt.stage_d_telemetry_jsonl
        or os.path.join(scene.model_path, "stage_d_telemetry.jsonl")
    )
    if telemetry_path.exists() and fresh_from_stage_b:
        raise FileExistsError(f"refusing to append a fresh Stage D run to {telemetry_path}")
    last_record = None
    for iteration in range(global_iteration + 1, opt.iterations + 1):
        phase = (
            _phase_for_iteration(iteration)
            if opt.stage_d_cached_twarmup else "exact_joint"
        )
        phase_a = phase == "cached_t_warmup"
        if opt.stage_d_cached_twarmup and iteration == CACHED_PHASE_A_END + 1:
            _set_branch_trainable(diffuse, True)
            _set_branch_trainable(reflection, True)
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        counts_before = {
            "diffuse": int(diffuse.get_xyz.shape[0]),
            "reflection": int(reflection.get_xyz.shape[0]),
            "transmittance": int(transmittance.get_xyz.shape[0]),
        }
        if not phase_a:
            reflection_iteration += 1
        transmittance_iteration += 1
        if not phase_a:
            diffuse.update_learning_rate(iteration)
            reflection.update_learning_rate(reflection_iteration)
        transmittance.update_learning_rate(transmittance_iteration)
        if not viewpoints:
            viewpoints, camera_indices = cameras.copy(), list(range(len(cameras)))
        selected = randint(0, len(camera_indices) - 1)
        camera = viewpoints.pop(selected)
        camera_indices.pop(selected)

        static_inputs = (
            static_cache.load(Path(str(camera.image_name)).stem, training_only=True)
            if phase_a else None
        )
        payload, memory_retries, used_chunk_size = _forward_backward_with_memory_retry(
            camera, state, pipe, background, opt, perceptual,
            diffuse, reflection, transmittance, iteration,
            static_inputs=static_inputs,
        )
        package, image, gt, loss = (
            payload["package"], payload["image"], payload["gt"], payload["loss"]
        )
        l1_value, ssim_value, rgb_loss = payload["l1"], payload["ssim"], payload["rgb"]
        normal_loss, mono_loss = payload["normal"], payload["mono"]
        perceptual_loss, specular_loss = payload["perceptual"], payload["specular"]
        depth_loss, depth_enabled = payload["depth"], payload["depth_enabled"]

        with torch.no_grad():
            reflection_aux = package.get("ray_aux")
            if not phase_a:
                reflection.add_densification_stats(
                    None if reflection_aux is None else reflection_aux.contributing_indices,
                    None if reflection_aux is None else reflection_aux.contributing_weights,
                )
            trans_aux = package["inside_ray_aux"]
            transmittance.add_densification_stats(
                None if trans_aux is None else trans_aux.contributing_indices,
                None if trans_aux is None else trans_aux.contributing_weights,
            )
            if not phase_a:
                diffuse.exposure_optimizer.step()
                diffuse.exposure_optimizer.zero_grad(set_to_none=True)
                diffuse.optimizer.step(); diffuse.optimizer.zero_grad(set_to_none=True)
                reflection.optimizer.step(); reflection.optimizer.zero_grad(set_to_none=True)
            transmittance.optimizer.step(); transmittance.optimizer.zero_grad(set_to_none=True)
            if phase_a:
                state.mark_transmittance_updated()
            else:
                state.mark_parameters_updated()
            topology_called = False
            if (
                transmittance_iteration > opt.transmittance_densify_from_iter
                and transmittance_iteration < opt.transmittance_densify_until_iter
                and transmittance_iteration % opt.transmittance_densification_interval == 0
            ):
                topology_called = True
                transmittance.densify_and_prune(
                    opt.transmittance_densify_grad_threshold,
                    opt.transmittance_min_opacity, scene.cameras_extent,
                    allow_unhit_prune=(
                        transmittance_iteration >= opt.transmittance_prune_unhit_after
                    ),
                )
            if opt.stage_d_cached_twarmup and iteration == CACHED_PHASE_A_END:
                frozen_hash_after = {
                    "diffuse": frozen_branch_hash(diffuse),
                    "reflection": frozen_branch_hash(reflection),
                }
                if frozen_hash_after != frozen_hash_before:
                    raise RuntimeError(
                        "CACHED_T_WARMUP_BLOCKED: D/R model/optimizer/scheduler/"
                        "densification/topology state changed in Phase A"
                    )
                config["cached_t_warmup"]["phase_a_frozen_hash_after"] = frozen_hash_after
                metadata["phase_a_frozen_hash_after"] = frozen_hash_after
                _atomic_json(
                    Path(scene.model_path, "cached_twarmup_run_metadata.json"), metadata
                )
            finite_counts = _finite_models({
                "diffuse": diffuse, "reflection": reflection,
                "transmittance": transmittance,
            })
            torch.cuda.synchronize()
            valid = package["transmittance_valid"] > 0.5
            depth_order = package["inside_depth"][valid] <= package["far_depth"][valid]
            wall_ms = float((time.perf_counter() - wall_start) * 1000.0)
            last_record = {
                "schema": (
                    "rtgs_stage_d_cached_twarmup_telemetry_v1"
                    if opt.stage_d_cached_twarmup else (
                        "rtgs_stage_d_formal_telemetry_v2" if opt.stage_d_formal_onset
                        else "rtgs_stage_d_smoke_telemetry_v1"
                    )
                ),
                "global_iteration": int(iteration),
                "training_phase": phase,
                "static_dr_cache_enabled": bool(phase_a),
                "optimizer_updates_this_step": {
                    "diffuse": int(not phase_a),
                    "reflection": int(not phase_a),
                    "transmittance": 1,
                },
                "reflection_local_iteration": int(reflection_iteration),
                "transmittance_local_iteration": int(transmittance_iteration),
                "camera_stem": str(camera.image_name),
                "geometry_release_id": release.manifest["geometry_release_id"],
                "geometry_release_aggregate_sha256": release.validation["aggregate_sha256"],
                "static_dr_cache_aggregate_sha256": (
                    static_cache.aggregate_sha256 if phase_a else None
                ),
                "counts": {
                    "diffuse": int(diffuse.get_xyz.shape[0]),
                    "reflection": int(reflection.get_xyz.shape[0]),
                    "transmittance": int(transmittance.get_xyz.shape[0]),
                },
                "counts_before": counts_before,
                "topology_event": {
                    "transmittance_densify_prune_called": topology_called,
                    "diffuse_delta": int(diffuse.get_xyz.shape[0]) - counts_before["diffuse"],
                    "reflection_delta": int(reflection.get_xyz.shape[0]) - counts_before["reflection"],
                    "transmittance_delta": int(transmittance.get_xyz.shape[0]) - counts_before["transmittance"],
                },
                "ray_memory_policy": {
                    "used_chunk_size": used_chunk_size,
                    "checkpoint_chunks": True,
                    "oom_retry_count": len(memory_retries),
                    "oom_retries": memory_retries,
                },
                "loss": {
                    "rgb": float(rgb_loss), "l1": float(l1_value),
                    "ssim": float(ssim_value), "l_norm": float(normal_loss),
                    "l_mono": float(mono_loss), "l_perc": float(perceptual_loss),
                    "l_spec": float(specular_loss), "l_depth": float(depth_loss),
                    "lambda_depth_enabled": bool(depth_enabled), "total": float(loss),
                },
                "valid_two_hit_fraction_hard_release": release.validation["coverage"]["valid_fraction_hard_mean"],
                "transmittance_valid_pixels": int(valid.sum()),
                "din_le_t_far_fraction": float(depth_order.float().mean()) if depth_order.numel() else 0.0,
                "whole_step_wall_ms": wall_ms,
                "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "finite_model_elements": finite_counts,
                "nonfinite_count": 0,
            }

            checkpoint_due = iteration in checkpoint_iterations or iteration == opt.iterations
            if checkpoint_due:
                checkpoint = make_stage_d_checkpoint(
                    diffuse, reflection, transmittance, iteration,
                    reflection_iteration, transmittance_iteration, source, config,
                    release.manifest["geometry_release_id"],
                    release.validation["aggregate_sha256"],
                    make_camera_runtime_state(camera_indices, len(cameras)),
                )
                torch.save(checkpoint, os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))
                del checkpoint
            if iteration in saving_iterations or iteration == opt.iterations:
                scene.save(iteration)
            if opt.stage_d_formal_onset and iteration in FORMAL_NODES:
                _render_formal_review_node(
                    scene, state, pipe, background, release, iteration
                )
                last_record["formal_review_node"] = True
            elif opt.stage_d_cached_twarmup and iteration in CACHED_NODES:
                _render_formal_review_node(
                    scene, state, pipe, background, release, iteration,
                    stems=CACHED_STEMS,
                    static_cache=static_cache if phase_a else None,
                )
                last_record["formal_review_node"] = True
            else:
                last_record["formal_review_node"] = False
            if writer:
                writer.add_scalar("stage_d/loss", float(loss), iteration)
                writer.add_scalar("stage_d/l_depth", float(depth_loss), iteration)
                writer.add_scalar("scene/transmittance_count", transmittance.get_xyz.shape[0], iteration)
        progress.update(1)
        del package, image, gt, loss, payload, static_inputs
        del l1_value, ssim_value, rgb_loss, normal_loss, mono_loss
        del perceptual_loss, specular_loss, depth_loss
        del reflection_aux, trans_aux, valid, depth_order
        device_free_before_release, _ = torch.cuda.mem_get_info()
        cache_released = _allocator_cache_under_pressure(device_free_before_release)
        if cache_released:
            torch.cuda.empty_cache()
        last_record["allocator_cache_released"] = bool(cache_released)
        last_record["device_free_before_cache_release_bytes"] = int(device_free_before_release)
        last_record["whole_step_wall_ms"] = float((time.perf_counter() - wall_start) * 1000.0)
        _write_jsonl(telemetry_path, last_record)

    if opt.stage_d_formal_onset or opt.stage_d_cached_twarmup:
        directory = os.path.join(scene.model_path, "debug", f"iteration_{opt.iterations:06d}")
        debug = None
    else:
        fixed_candidates = scene.getTestCameras() or scene.getTrainCameras()
        fixed = next(
            (
                camera for camera in fixed_candidates
                if Path(str(camera.image_name)).stem == "000039"
            ),
            fixed_candidates[0],
        )
        with torch.no_grad():
            debug = render(
                fixed, state, pipe, background,
                return_ray_aux=False, return_ray_diagnostics=True,
            )
            directory = os.path.join(scene.model_path, "debug", f"iteration_{opt.iterations:06d}")
            save_transmittance_debug_maps(
                debug, fixed.original_image.cuda(), directory,
                fixed.specular_mask, fixed.specular_mask_sha256,
                release.manifest["geometry_release_id"],
                release.validation["aggregate_sha256"],
            )
    final_release_validation = validate_geometry_release(dataset.geometry_release_manifest)
    summary = {
        "status": (
            "STAGE_D_CACHED_TWARMUP_AND_JOINT_COMPLETED"
            if opt.stage_d_cached_twarmup else (
                "STAGE_D_FORMAL_ONSET_COMPLETED" if opt.stage_d_formal_onset
                else "STAGE_D_SMOKE_COMPLETED"
            )
        ),
        "start_checkpoint": str(Path(start_checkpoint).resolve()),
        "final_checkpoint": str(Path(scene.model_path) / f"chkpnt{opt.iterations}.pth"),
        "geometry_release_validation": final_release_validation,
        "last_telemetry": last_record,
        "debug_directory": directory,
        "independent_parameter_storage": len({
            diffuse._xyz.data_ptr(), reflection._xyz.data_ptr(), transmittance._xyz.data_ptr()
        }) == 3,
        "composition": "Ct = Cin + (1 - Ain) * Cout",
        "first_bounce": "T from D position + epsilon*d_cam",
        "second_bounce": "D from frozen back_position + epsilon*d_cam",
        "phase_a_frozen_hash_before": frozen_hash_before,
        "phase_a_frozen_hash_after": frozen_hash_after,
        "phase_b_static_dr_cache_enabled": False if opt.stage_d_cached_twarmup else None,
    }
    summary_name = (
        "stage_d_cached_twarmup_summary.json" if opt.stage_d_cached_twarmup else (
            "stage_d_formal_summary.json" if opt.stage_d_formal_onset
            else "stage_d_smoke_summary.json"
        )
    )
    with open(os.path.join(scene.model_path, summary_name), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    progress.close()
    print(summary["status"] + " " + json.dumps(summary["last_telemetry"], sort_keys=True))
