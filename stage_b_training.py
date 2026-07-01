"""Stage B-only training loop. It contains no Stage C/D paths or outputs."""

import json
import os
from random import randint

import torch
from tqdm import tqdm

from gaussian_renderer.reflection_renderer import StageBRenderState, render
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_b_scene import StageBScene
from scene.stage_b_state import (
    initialize_stage_b_from_diffuse,
    load_stage_b_checkpoint,
    make_stage_b_checkpoint,
)
from utils.image_utils import psnr
from utils.loss_utils import (
    l1_loss,
    monocular_normal_loss,
    normal_depth_consistency_loss,
    specular_constraint_loss,
    ssim,
)
from utils.reflection_debug import save_reflection_debug_maps
from utils.specular_mask import validate_specular_mask_set

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def _finite_stats(values):
    values = values.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() != values.numel() or not finite.numel():
        raise FloatingPointError("requested Stage B diagnostic values contain NaN/Inf or are empty")
    q = torch.quantile(finite, torch.tensor([0.5, 0.95, 0.99], device=finite.device))
    return {
        "count": int(finite.numel()),
        "min": float(finite.min().item()),
        "mean": float(finite.mean().item()),
        "p50": float(q[0].item()),
        "p95": float(q[1].item()),
        "p99": float(q[2].item()),
        "max": float(finite.max().item()),
    }


def specular_gradient_diagnostics(surface_ks, soft_mask, valid_surface, specular_gradient):
    """Audit L_spec support without changing accumulated model gradients."""
    mask = soft_mask.permute(1, 2, 0) if soft_mask.shape[0] == 1 else soft_mask
    valid = valid_surface > 0.5
    inside = valid & (mask > 0.0)
    outside = valid & (mask == 0.0)
    if not inside.any() or not outside.any():
        raise RuntimeError("formal mask must expose both inside and outside valid D surface pixels")
    inside_gradient = specular_gradient[inside]
    outside_gradient = specular_gradient[outside]
    inside_nonzero = inside_gradient.abs() > 0
    return {
        "mask_mean": float(mask.mean().item()),
        "mask_support_fraction": float((mask > 0).float().mean().item()),
        "inside_ks": _finite_stats(surface_ks[inside]),
        "outside_ks": _finite_stats(surface_ks[outside]),
        "l_spec_gradient": {
            "inside_nonzero_count": int(inside_nonzero.sum().item()),
            "inside_nonzero_fraction": float(inside_nonzero.float().mean().item()),
            "inside_min": float(inside_gradient.min().item()),
            "inside_mean": float(inside_gradient.mean().item()),
            "inside_max": float(inside_gradient.max().item()),
            "outside_max_abs": float(outside_gradient.abs().max().item()),
            "gradient_descent_ks_direction": "increase" if inside_gradient.mean() < 0 else "not_increase",
        },
    }


def _model_finite_summary(model):
    checked = 0
    for name, value in model.__dict__.items():
        if torch.is_tensor(value):
            checked += value.numel()
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"model tensor {name} contains NaN/Inf")
            if value.grad is not None and not torch.isfinite(value.grad).all():
                raise FloatingPointError(f"model gradient {name} contains NaN/Inf")
    return {"checked_elements": int(checked), "finite": True}


def _append_specular_smoke_record(model_path, record):
    path = os.path.join(model_path, "specular_smoke_metrics.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _config(dataset, opt, diffuse, mask_manifest):
    return {
        "stage": "stage_b",
        "model_type": "surfel",
        "roughness_min": float(diffuse.roughness_min),
        "roughness_remap": bool(dataset.roughness_remap),
        "material_alpha_threshold": float(dataset.material_alpha_threshold),
        "ray_background": dataset.ray_background,
        "ray_chunk_size": int(dataset.ray_chunk_size),
        "ray_cutoff_sigma": float(dataset.ray_cutoff_sigma),
        "ray_hit_threshold": float(dataset.ray_hit_threshold),
        "ray_epsilon_scale": float(dataset.ray_epsilon_scale),
        "bsdf_weight_mode": "brdf_times_cosine",
        "lambda_norm": float(opt.lambda_norm),
        "lambda_mono": float(opt.lambda_mono),
        "lambda_perc": float(opt.lambda_perc),
        "lambda_spec": float(opt.lambda_spec),
        "specular_k0": float(opt.specular_k0),
        "diffuse_schedule": {
            "position_lr_init": float(opt.position_lr_init),
            "position_lr_final": float(opt.position_lr_final),
            "position_lr_delay_mult": float(opt.position_lr_delay_mult),
            "position_lr_max_steps": int(opt.position_lr_max_steps),
            "densify_from_iter": int(opt.densify_from_iter),
            "densify_until_iter": int(opt.densify_until_iter),
            "densification_interval": int(opt.densification_interval),
            "densify_grad_threshold": float(opt.densify_grad_threshold),
        },
        "reflection_schedule": {
            "position_lr_init": float(opt.reflection_position_lr_init),
            "position_lr_final": float(opt.reflection_position_lr_final),
            "position_lr_delay_mult": float(opt.reflection_position_lr_delay_mult),
            "position_lr_max_steps": int(opt.reflection_position_lr_max_steps),
            "color_lr": float(opt.reflection_color_lr),
            "opacity_lr": float(opt.reflection_opacity_lr),
            "scaling_lr": float(opt.reflection_scaling_lr),
            "rotation_lr": float(opt.reflection_rotation_lr),
            "percent_dense": float(opt.reflection_percent_dense),
            "densify_from_iter": int(opt.reflection_densify_from_iter),
            "densify_until_iter": int(opt.reflection_densify_until_iter),
            "densification_interval": int(opt.reflection_densification_interval),
            "densify_grad_threshold": float(opt.reflection_densify_grad_threshold),
            "min_opacity": float(opt.reflection_min_opacity),
            "prune_unhit_after": int(opt.reflection_prune_unhit_after),
        },
        "specular_mask": None if mask_manifest is None else {
            "role": mask_manifest["role"],
            "count": mask_manifest["count"],
            "aggregate_sha256": mask_manifest["aggregate_sha256"],
            "manifest_payload_sha256": mask_manifest["manifest_payload_sha256"],
            "manifest_file_sha256": mask_manifest["manifest_file_sha256"],
            "mask_interpolation": mask_manifest["mask_interpolation"],
        },
    }


def _validate_stage_b_args(dataset, opt, start_checkpoint, diffuse_init_checkpoint):
    if dataset.model_type != "surfel" or dataset.stage != "stage_b":
        raise ValueError("Stage B requires --model_type surfel --stage stage_b")
    if bool(start_checkpoint) == bool(diffuse_init_checkpoint):
        raise ValueError("specify exactly one of --start_checkpoint or --diffuse_init_checkpoint")
    if dataset.ray_background != "scene":
        raise ValueError("Stage B supports only --ray_background scene")
    if opt.random_background:
        raise ValueError("Stage B forbids random/image-dependent reflection backgrounds")
    if opt.lambda_spec == 0 and dataset.specular_masks:
        raise ValueError("--lambda_spec=0 requires an empty --specular_masks path")
    if opt.lambda_spec > 0 and not dataset.specular_masks:
        raise ValueError("--lambda_spec>0 requires a complete formal --specular_masks manifest")
    if opt.specular_smoke_diagnostics and opt.lambda_spec <= 0:
        raise ValueError("--specular_smoke_diagnostics requires --lambda_spec>0")
    if dataset.reflection_init_mode != "random_bbox":
        raise ValueError("B-1 currently implements reflection_init_mode=random_bbox only")
    if diffuse_init_checkpoint and dataset.reflection_init_count <= 0:
        raise ValueError("new Stage B runs require an explicit positive --reflection_init_count")


def _report(iteration, testing_iterations, scene, state, pipe, background):
    if iteration not in testing_iterations:
        return
    torch.cuda.empty_cache()
    configs = (
        ("test", scene.getTestCameras()),
        ("train", [scene.getTrainCameras()[index % len(scene.getTrainCameras())] for index in range(5, 30, 5)]),
    )
    for name, cameras in configs:
        if not cameras:
            continue
        l1_value = 0.0
        psnr_value = 0.0
        for camera in cameras:
            package = render(camera, state, pipe, background)
            image = package["render"].clamp(0.0, 1.0)
            gt = camera.original_image.to("cuda").clamp(0.0, 1.0)
            l1_value += l1_loss(image, gt).double()
            psnr_value += psnr(image, gt).mean().double()
        l1_value /= len(cameras)
        psnr_value /= len(cameras)
        print(f"\n[ITER {iteration}] Evaluating {name}: L1 {l1_value} PSNR {psnr_value}")
    torch.cuda.empty_cache()


def training_stage_b(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    start_checkpoint,
    diffuse_init_checkpoint,
    debug_from,
    prepare_output_and_logger,
):
    _validate_stage_b_args(dataset, opt, start_checkpoint, diffuse_init_checkpoint)
    mask_manifest = None
    if opt.lambda_spec > 0:
        mask_manifest = validate_specular_mask_set(dataset.source_path, dataset.images, dataset.specular_masks)
        dataset._validated_specular_mask_manifest = mask_manifest
        print(
            "Stage B formal specular masks: count={} aggregate_sha256={} manifest_payload_sha256={} interpolation={}".format(
                mask_manifest["count"], mask_manifest["aggregate_sha256"],
                mask_manifest["manifest_payload_sha256"], mask_manifest["mask_interpolation"],
            )
        )

    writer = prepare_output_and_logger(dataset)
    diffuse = DiffuseSurfelModel(dataset.roughness_min, opt.optimizer_type)
    reflection = ReflectionSurfelModel()
    scene = StageBScene(dataset, diffuse, reflection, shuffle=True, write_metadata=True)

    if start_checkpoint:
        global_iteration, reflection_iteration, provenance, saved_config = load_stage_b_checkpoint(
            start_checkpoint,
            diffuse,
            reflection,
            opt,
            opt,
            map_location="cuda",
        )
        current_config = _config(dataset, opt, diffuse, mask_manifest)
        for key in (
            "roughness_remap", "material_alpha_threshold", "ray_background", "ray_cutoff_sigma",
            "ray_hit_threshold", "ray_epsilon_scale", "bsdf_weight_mode",
            "diffuse_schedule", "reflection_schedule",
        ):
            if saved_config.get(key) != current_config.get(key):
                raise ValueError(f"Stage B resume config mismatch for {key}")
    else:
        global_iteration, reflection_iteration, provenance = initialize_stage_b_from_diffuse(
            diffuse_init_checkpoint,
            diffuse,
            reflection,
            opt,
            opt,
            reflection_count=dataset.reflection_init_count,
            reflection_seed=dataset.reflection_init_seed,
            map_location="cuda",
        )
    if global_iteration >= opt.iterations:
        raise ValueError("--iterations must be greater than the restored global iteration")
    if opt.specular_smoke_diagnostics and opt.iterations - global_iteration > 3:
        raise ValueError("--specular_smoke_diagnostics is fail-closed to at most three added iterations")

    print(
        "Stage B initialization: D_iteration={} R_local_step={} R_count={} mode={} bbox_min={} bbox_max={}".format(
            global_iteration,
            reflection_iteration,
            reflection.get_xyz.shape[0],
            reflection.initialization.get("mode"),
            reflection.initialization.get("bbox_min"),
            reflection.initialization.get("bbox_max"),
        )
    )
    state = StageBRenderState(
        diffuse=diffuse,
        reflection=reflection,
        scene_radius=scene.cameras_extent,
        ray_chunk_size=dataset.ray_chunk_size,
        ray_cutoff_sigma=dataset.ray_cutoff_sigma,
        ray_hit_threshold=dataset.ray_hit_threshold,
        ray_epsilon_scale=dataset.ray_epsilon_scale,
        material_alpha_threshold=dataset.material_alpha_threshold,
        roughness_min=diffuse.roughness_min,
        roughness_remap=dataset.roughness_remap,
        ray_background=dataset.ray_background,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    perceptual = None
    if opt.lambda_perc > 0:
        from utils.perceptual_loss import VGG16PerceptualLoss
        perceptual = VGG16PerceptualLoss(pretrained=True).cuda().eval()

    viewpoints = scene.getTrainCameras().copy()
    indices = list(range(len(viewpoints)))
    progress = tqdm(range(global_iteration, opt.iterations), desc="Stage B training progress")
    ema_loss = 0.0
    for iteration in range(global_iteration + 1, opt.iterations + 1):
        if opt.specular_smoke_diagnostics:
            torch.cuda.reset_peak_memory_stats()
        reflection_iteration += 1
        diffuse.update_learning_rate(iteration)
        reflection.update_learning_rate(reflection_iteration)
        if not viewpoints:
            viewpoints = scene.getTrainCameras().copy()
            indices = list(range(len(viewpoints)))
        selected = randint(0, len(indices) - 1)
        camera = viewpoints.pop(selected)
        indices.pop(selected)
        if (iteration - 1) == debug_from:
            pipe.debug = True

        package = render(
            camera,
            state,
            pipe,
            background,
            return_ray_aux=True,
            return_ray_diagnostics=opt.specular_smoke_diagnostics,
        )
        image = package["render"]
        gt = camera.original_image.cuda()
        l1_value = l1_loss(image, gt)
        ssim_value = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt)
        loss = (1.0 - opt.lambda_dssim) * l1_value + opt.lambda_dssim * (1.0 - ssim_value)
        normal_loss = normal_depth_consistency_loss(package["normal"], package["position"], package["alpha"])
        loss = loss + opt.lambda_norm * normal_loss
        mono_loss = image.new_zeros(())
        if camera.normal_prior is not None:
            from utils.surfel_utils import camera_normals_to_world, face_forward
            target = camera.normal_prior.permute(1, 2, 0)
            if camera.normal_prior_space == "camera":
                target = camera_normals_to_world(camera, target)
            target = face_forward(target, package["position"], camera.camera_center)
            mono_loss = monocular_normal_loss(
                package["normal"], target, package["alpha"], camera.normal_prior_valid.permute(1, 2, 0)
            )
            loss = loss + opt.lambda_mono * mono_loss
        perceptual_loss = image.new_zeros(())
        if perceptual is not None:
            perceptual_loss = perceptual(image, gt)
            loss = loss + opt.lambda_perc * perceptual_loss
        specular_loss = image.new_zeros(())
        specular_probe = None
        if opt.lambda_spec > 0:
            if camera.specular_mask is None:
                raise RuntimeError("validated specular mask was not loaded")
            specular_loss = specular_constraint_loss(package["surface_ks"], camera.specular_mask, opt.specular_k0)
            loss = loss + opt.lambda_spec * specular_loss
            if opt.specular_smoke_diagnostics:
                specular_probe = torch.autograd.grad(
                    opt.lambda_spec * specular_loss,
                    package["surface_ks"],
                    retain_graph=True,
                    allow_unused=False,
                )[0]
        loss.backward()

        with torch.no_grad():
            aux = package["ray_aux"]
            reflection.add_densification_stats(
                None if aux is None else aux.contributing_indices,
                None if aux is None else aux.contributing_weights,
            )
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if opt.specular_smoke_diagnostics:
                diagnostics = package.get("ray_diagnostics")
                if diagnostics is None:
                    raise RuntimeError("specular smoke requires real ray diagnostics")
                record = {
                    "global_iteration": int(iteration),
                    "reflection_local_iteration": int(reflection_iteration),
                    "camera_stem": camera.image_name,
                    "loss": float(loss.item()),
                    "l1": float(l1_value.item()),
                    "l_spec": float(specular_loss.item()),
                    "lambda_spec": float(opt.lambda_spec),
                    "counts": {
                        "diffuse": int(diffuse.get_xyz.shape[0]),
                        "reflection": int(reflection.get_xyz.shape[0]),
                        "valid_rays": int(package["valid_ray_count"]),
                    },
                    "mask": specular_gradient_diagnostics(
                        package["surface_ks"], camera.specular_mask,
                        package["valid_surface_mask"], specular_probe,
                    ),
                    "ray": {
                        "candidate_count": _finite_stats(package["ray_candidate_count"][package["valid_surface_mask"] > 0.5]),
                        "exact_intersection_count": _finite_stats(package["ray_exact_intersection_count"][package["valid_surface_mask"] > 0.5]),
                        "chunk_count": int(diagnostics.chunk_count),
                        "timing_ms": {key: float(value) for key, value in diagnostics.timing_ms.items()},
                    },
                    "memory": {
                        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    },
                    "finite": {
                        "loss": bool(torch.isfinite(loss).item()),
                        "diffuse": _model_finite_summary(diffuse),
                        "reflection": _model_finite_summary(reflection),
                    },
                }
                gradient = record["mask"]["l_spec_gradient"]
                if record["l_spec"] <= 0 or gradient["inside_nonzero_count"] == 0:
                    raise RuntimeError("L_spec smoke requires positive loss and nonzero inside-mask ks gradient")
                if gradient["outside_max_abs"] != 0.0 or gradient["gradient_descent_ks_direction"] != "increase":
                    raise RuntimeError("L_spec gradient support/direction contract failed")
                _append_specular_smoke_record(scene.model_path, record)
                print("\nSPECULAR_SMOKE " + json.dumps(record, sort_keys=True, allow_nan=False))
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": f"{ema_loss:.7f}", "D": diffuse.get_xyz.shape[0], "R": reflection.get_xyz.shape[0]})
                progress.update(10)
            if writer:
                writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
                writer.add_scalar("train_loss_patches/normal_depth", normal_loss.item(), iteration)
                writer.add_scalar("train_loss_patches/monocular_normal", mono_loss.item(), iteration)
                writer.add_scalar("train_loss_patches/perceptual", perceptual_loss.item(), iteration)
                writer.add_scalar("train_loss_patches/specular", specular_loss.item(), iteration)
                writer.add_scalar("scene/diffuse_count", diffuse.get_xyz.shape[0], iteration)
                writer.add_scalar("scene/reflection_count", reflection.get_xyz.shape[0], iteration)

            _report(iteration, testing_iterations, scene, state, pipe, background)
            if iteration % opt.debug_interval == 0 or iteration == opt.iterations:
                fixed = (scene.getTestCameras() or scene.getTrainCameras())[0]
                debug = render(
                    fixed, state, pipe, background, return_ray_diagnostics=True
                )
                directory = os.path.join(scene.model_path, "debug", f"iteration_{iteration:06d}")
                save_reflection_debug_maps(
                    debug,
                    fixed.original_image.cuda(),
                    directory,
                    specular_mask=fixed.specular_mask if opt.lambda_spec > 0 else None,
                    mask_sha256=fixed.specular_mask_sha256 if opt.lambda_spec > 0 else None,
                )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving independent D/R point clouds")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                visibility = package["visibility_filter"]
                radii = package["radii"]
                diffuse.max_radii2D[visibility] = torch.maximum(diffuse.max_radii2D[visibility], radii[visibility])
                diffuse.add_densification_stats(package["viewspace_points"], visibility)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    threshold = 20 if iteration > opt.opacity_reset_interval else None
                    diffuse.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, threshold, radii)
                if iteration % opt.opacity_reset_interval == 0:
                    diffuse.reset_opacity()

            if iteration < opt.iterations:
                diffuse.exposure_optimizer.step()
                diffuse.exposure_optimizer.zero_grad(set_to_none=True)
                diffuse.optimizer.step()
                diffuse.optimizer.zero_grad(set_to_none=True)
                reflection.optimizer.step()
                reflection.optimizer.zero_grad(set_to_none=True)
                state.acceleration.mark_parameters_updated()
                if (
                    reflection_iteration > opt.reflection_densify_from_iter
                    and reflection_iteration < opt.reflection_densify_until_iter
                    and reflection_iteration % opt.reflection_densification_interval == 0
                ):
                    reflection.densify_and_prune(
                        opt.reflection_densify_grad_threshold,
                        opt.reflection_min_opacity,
                        scene.cameras_extent,
                        allow_unhit_prune=reflection_iteration >= opt.reflection_prune_unhit_after,
                    )

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Stage B Checkpoint")
                checkpoint = make_stage_b_checkpoint(
                    diffuse,
                    reflection,
                    iteration,
                    reflection_iteration,
                    provenance,
                    _config(dataset, opt, diffuse, mask_manifest),
                )
                torch.save(checkpoint, os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))
    progress.close()
