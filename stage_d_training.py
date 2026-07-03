"""Stage D D/R/T training loop with one immutable geometry release."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
from plyfile import PlyData
from tqdm import tqdm

from gaussian_renderer.transmittance_renderer import StageDRenderState, render
from geometry.geometry_release import GeometryRelease, validate_geometry_release
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
from utils.training_state import make_camera_runtime_state, restore_camera_deck
from utils.transmittance_debug import save_transmittance_debug_maps

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


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


def _finite_model(model, name):
    count = 0
    for key, value in model.__dict__.items():
        if torch.is_tensor(value):
            count += value.numel()
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{name}.{key} contains NaN/Inf")
            if value.grad is not None and not torch.isfinite(value.grad).all():
                raise FloatingPointError(f"{name}.{key}.grad contains NaN/Inf")
    return int(count)


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


def _write_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


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
        runtime_state = None
    elif checkpoint_format == STAGE_D_FORMAT:
        (
            global_iteration, reflection_iteration, transmittance_iteration,
            source, saved_config, runtime_state,
        ) = load_stage_d_checkpoint(
            start_checkpoint, diffuse, reflection, transmittance, opt, opt, opt,
            release.manifest["geometry_release_id"],
            release.validation["aggregate_sha256"], map_location="cuda",
        )
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

    cameras = scene.getTrainCameras().copy()
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
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        reflection_iteration += 1
        transmittance_iteration += 1
        diffuse.update_learning_rate(iteration)
        reflection.update_learning_rate(reflection_iteration)
        transmittance.update_learning_rate(transmittance_iteration)
        if not viewpoints:
            viewpoints, camera_indices = cameras.copy(), list(range(len(cameras)))
        selected = randint(0, len(camera_indices) - 1)
        camera = viewpoints.pop(selected)
        camera_indices.pop(selected)

        package = render(camera, state, pipe, background, return_ray_aux=True)
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
        depth_enabled = iteration >= opt.stage_d_depth_start_iteration
        loss = (
            rgb_loss + opt.lambda_norm * normal_loss + opt.lambda_mono * mono_loss
            + opt.lambda_perc * perceptual_loss + opt.lambda_spec * specular_loss
            + (opt.lambda_depth * depth_loss if depth_enabled else 0.0)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage D total loss is NaN/Inf")
        loss.backward()

        with torch.no_grad():
            reflection_aux = package["ray_aux"]
            reflection.add_densification_stats(
                None if reflection_aux is None else reflection_aux.contributing_indices,
                None if reflection_aux is None else reflection_aux.contributing_weights,
            )
            trans_aux = package["inside_ray_aux"]
            transmittance.add_densification_stats(
                None if trans_aux is None else trans_aux.contributing_indices,
                None if trans_aux is None else trans_aux.contributing_weights,
            )
            diffuse.exposure_optimizer.step()
            diffuse.exposure_optimizer.zero_grad(set_to_none=True)
            diffuse.optimizer.step(); diffuse.optimizer.zero_grad(set_to_none=True)
            reflection.optimizer.step(); reflection.optimizer.zero_grad(set_to_none=True)
            transmittance.optimizer.step(); transmittance.optimizer.zero_grad(set_to_none=True)
            state.mark_parameters_updated()
            if (
                transmittance_iteration > opt.transmittance_densify_from_iter
                and transmittance_iteration < opt.transmittance_densify_until_iter
                and transmittance_iteration % opt.transmittance_densification_interval == 0
            ):
                transmittance.densify_and_prune(
                    opt.transmittance_densify_grad_threshold,
                    opt.transmittance_min_opacity, scene.cameras_extent,
                    allow_unhit_prune=(
                        transmittance_iteration >= opt.transmittance_prune_unhit_after
                    ),
                )
            finite_counts = {
                "diffuse": _finite_model(diffuse, "diffuse"),
                "reflection": _finite_model(reflection, "reflection"),
                "transmittance": _finite_model(transmittance, "transmittance"),
            }
            torch.cuda.synchronize()
            valid = package["transmittance_valid"] > 0.5
            depth_order = package["inside_depth"][valid] <= package["far_depth"][valid]
            wall_ms = float((time.perf_counter() - wall_start) * 1000.0)
            last_record = {
                "schema": "rtgs_stage_d_smoke_telemetry_v1",
                "global_iteration": int(iteration),
                "reflection_local_iteration": int(reflection_iteration),
                "transmittance_local_iteration": int(transmittance_iteration),
                "camera_stem": str(camera.image_name),
                "geometry_release_id": release.manifest["geometry_release_id"],
                "geometry_release_aggregate_sha256": release.validation["aggregate_sha256"],
                "counts": {
                    "diffuse": int(diffuse.get_xyz.shape[0]),
                    "reflection": int(reflection.get_xyz.shape[0]),
                    "transmittance": int(transmittance.get_xyz.shape[0]),
                },
                "loss": {
                    "rgb": float(rgb_loss), "l1": float(l1_value),
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
            if iteration in saving_iterations or iteration == opt.iterations:
                scene.save(iteration)
            _write_jsonl(telemetry_path, last_record)
            if writer:
                writer.add_scalar("stage_d/loss", float(loss), iteration)
                writer.add_scalar("stage_d/l_depth", float(depth_loss), iteration)
                writer.add_scalar("scene/transmittance_count", transmittance.get_xyz.shape[0], iteration)
        progress.update(1)
        del package, image, gt, loss
        torch.cuda.empty_cache()

    fixed_candidates = scene.getTestCameras() or scene.getTrainCameras()
    fixed = next((camera for camera in fixed_candidates if str(camera.image_name) == "000039"), fixed_candidates[0])
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
        "status": "STAGE_D_SMOKE_COMPLETED",
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
    }
    with open(os.path.join(scene.model_path, "stage_d_smoke_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    progress.close()
    print("STAGE_D_SMOKE_COMPLETED " + json.dumps(summary["last_telemetry"], sort_keys=True))
