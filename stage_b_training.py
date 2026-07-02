"""Stage B-only training loop. It contains no Stage C/D paths or outputs."""

import json
import math
import os
import time
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
from utils.stage_b_telemetry import (
    KS_STAT_KEYS,
    SCHEMA_VERSION as TELEMETRY_SCHEMA_VERSION,
    StageBTelemetryWriter,
    cuda_allocator_snapshot,
    mask_ks_stats_from_existing_tensors,
    validate_telemetry_options,
)
from utils.stage_b_memory import (
    V2_POLICY,
    evaluate_headroom,
    release_allocator_cache,
    validate_stage_b_memory_policy,
    write_headroom_gate,
)
from utils.training_state import (
    capture_rng_state,
    make_camera_runtime_state,
    restore_camera_deck,
    restore_rng_state,
    should_step_optimizer,
    validate_tier2_experiment_identity,
)

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
    operator_gate = bool(getattr(opt, "operator_gate_continuation", False))
    return {
        "stage": "stage_b",
        "model_type": "surfel",
        "experiment": getattr(dataset, "experiment", ""),
        "resolution": int(dataset.resolution),
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
        "operator_gate_continuation": operator_gate,
        "tier2_retry_identity": str(getattr(opt, "tier2_retry_identity", "")),
        "allocator_policy": {
            "name": str(getattr(opt, "stage_b_allocator_policy", "default")),
            "reference_peak_allocated_bytes": int(
                getattr(opt, "stage_b_reference_peak_allocated_bytes", 0)
            ),
            "minimum_projected_headroom_bytes": int(
                getattr(opt, "stage_b_minimum_projected_headroom_bytes", 0)
            ),
            "release_boundary": (
                "completed training step after telemetry statistics and before JSON serialization"
            ),
        },
        "operator_contract": None if not operator_gate else {
            "source_path": dataset.source_path,
            "images": dataset.images,
            "experiment": getattr(dataset, "experiment", ""),
            "resolution": int(dataset.resolution),
            "normal_priors": dataset.normal_priors,
            "normal_prior_space": dataset.normal_prior_space,
            "optimizer_type": opt.optimizer_type,
            "random_background": bool(opt.random_background),
            "lambda_dssim": float(opt.lambda_dssim),
            "feature_lr": float(opt.feature_lr),
            "material_lr": float(opt.material_lr),
            "opacity_lr": float(opt.opacity_lr),
            "scaling_lr": float(opt.scaling_lr),
            "rotation_lr": float(opt.rotation_lr),
            "exposure_lr_init": float(opt.exposure_lr_init),
            "exposure_lr_final": float(opt.exposure_lr_final),
            "exposure_lr_delay_steps": int(opt.exposure_lr_delay_steps),
            "exposure_lr_delay_mult": float(opt.exposure_lr_delay_mult),
            "percent_dense": float(opt.percent_dense),
            "opacity_reset_interval": int(opt.opacity_reset_interval),
            "tier2_retry_identity": str(getattr(opt, "tier2_retry_identity", "")),
            "stage_b_allocator_policy": str(
                getattr(opt, "stage_b_allocator_policy", "default")
            ),
            "stage_b_reference_peak_allocated_bytes": int(
                getattr(opt, "stage_b_reference_peak_allocated_bytes", 0)
            ),
            "stage_b_minimum_projected_headroom_bytes": int(
                getattr(opt, "stage_b_minimum_projected_headroom_bytes", 0)
            ),
        },
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
    operator_gate = bool(getattr(opt, "operator_gate_continuation", False))
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
    validate_telemetry_options(
        getattr(opt, "stage_b_telemetry_jsonl", ""),
        getattr(opt, "stage_b_telemetry_max_steps", 0),
        getattr(opt, "stage_b_telemetry_phase_tag", ""),
    )
    validate_stage_b_memory_policy(
        getattr(opt, "stage_b_allocator_policy", "default"),
        getattr(opt, "tier2_retry_identity", ""),
        int(getattr(opt, "stage_b_reference_peak_allocated_bytes", 0)),
        int(getattr(opt, "stage_b_minimum_projected_headroom_bytes", 0)),
        operator_gate,
    )
    if getattr(opt, "d_bootstrap_telemetry_jsonl", ""):
        raise ValueError("D bootstrap telemetry is valid only on the D-only training path")
    if operator_gate:
        validate_tier2_experiment_identity(dataset.experiment, dataset.resolution)
        if not getattr(opt, "stage_b_telemetry_jsonl", ""):
            raise ValueError("operator-gated Stage B continuation requires bounded telemetry")
        if (
            getattr(opt, "operator_gate_expected_global_start", -1) < 0
            or getattr(opt, "operator_gate_expected_r_local_start", -1) < 0
        ):
            raise ValueError("operator gate requires explicit expected global and R-local starts")


def _validate_operator_restored_state(opt, global_iteration, reflection_iteration):
    if not getattr(opt, "operator_gate_continuation", False):
        return
    if global_iteration != int(opt.operator_gate_expected_global_start):
        raise ValueError(
            f"operator gate expected global start {opt.operator_gate_expected_global_start}, "
            f"restored {global_iteration}"
        )
    if reflection_iteration != int(opt.operator_gate_expected_r_local_start):
        raise ValueError(
            f"operator gate expected R-local start {opt.operator_gate_expected_r_local_start}, "
            f"restored {reflection_iteration}"
        )
    if reflection_iteration < 100 and opt.lambda_spec != 0:
        raise ValueError("Tier 2 R-local 1--100 requires lambda_spec=0 and no mask")
    if reflection_iteration >= 100 and abs(float(opt.lambda_spec) - 0.2) > 1e-12:
        raise ValueError("Tier 2 R-local 101+ requires lambda_spec=0.2 and the formal mask")


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

    runtime_state = None
    operator_rng_state = None
    if start_checkpoint:
        global_iteration, reflection_iteration, provenance, saved_config, runtime_state = load_stage_b_checkpoint(
            start_checkpoint,
            diffuse,
            reflection,
            opt,
            opt,
            map_location="cuda",
            require_full_state=opt.operator_gate_continuation,
            return_runtime_state=True,
        )
        current_config = _config(dataset, opt, diffuse, mask_manifest)
        for key in (
            "roughness_remap", "material_alpha_threshold", "ray_background", "ray_cutoff_sigma",
            "ray_hit_threshold", "ray_epsilon_scale", "bsdf_weight_mode",
            "diffuse_schedule", "reflection_schedule",
        ):
            if saved_config.get(key) != current_config.get(key):
                raise ValueError(f"Stage B resume config mismatch for {key}")
        if "operator_gate_continuation" in saved_config and (
            saved_config["operator_gate_continuation"] != current_config["operator_gate_continuation"]
        ):
            raise ValueError("Stage B resume config mismatch for operator_gate_continuation")
        if opt.operator_gate_continuation and (
            saved_config.get("operator_contract") != current_config.get("operator_contract")
        ):
            raise ValueError("Stage B resume config mismatch for operator_contract")
    else:
        global_iteration, reflection_iteration, provenance, runtime_state = initialize_stage_b_from_diffuse(
            diffuse_init_checkpoint,
            diffuse,
            reflection,
            opt,
            opt,
            reflection_count=dataset.reflection_init_count,
            reflection_seed=dataset.reflection_init_seed,
            map_location="cuda",
            require_full_state=opt.operator_gate_continuation,
            return_runtime_state=True,
            verify_rng_unchanged=opt.operator_gate_continuation,
        )
    if opt.operator_gate_continuation:
        _validate_operator_restored_state(opt, global_iteration, reflection_iteration)
        operator_rng_state = capture_rng_state()
    if global_iteration >= opt.iterations:
        raise ValueError("--iterations must be greater than the restored global iteration")
    if opt.specular_smoke_diagnostics and opt.iterations - global_iteration > 3:
        raise ValueError("--specular_smoke_diagnostics is fail-closed to at most three added iterations")

    telemetry = None
    if getattr(opt, "stage_b_telemetry_jsonl", ""):
        telemetry = StageBTelemetryWriter(
            opt.stage_b_telemetry_jsonl,
            opt.stage_b_telemetry_max_steps,
            opt.stage_b_telemetry_phase_tag,
        )
        telemetry.validate_planned_steps(opt.iterations - global_iteration)

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
    if operator_rng_state is not None:
        restore_rng_state(operator_rng_state)

    all_train_cameras = scene.getTrainCameras().copy()
    viewpoints, indices = restore_camera_deck(all_train_cameras, runtime_state)
    progress = tqdm(range(global_iteration, opt.iterations), desc="Stage B training progress")
    ema_loss = 0.0
    allocator_policy = str(getattr(opt, "stage_b_allocator_policy", "default"))
    allocator_peak_scope = "process_start_or_preexisting_reset_before_telemetry"
    for iteration in range(global_iteration + 1, opt.iterations + 1):
        telemetry_wall_start = time.perf_counter() if telemetry is not None else None
        d_count_before = int(diffuse.get_xyz.shape[0]) if telemetry is not None else None
        r_count_before = int(reflection.get_xyz.shape[0]) if telemetry is not None else None
        r_topology_before = int(reflection.topology_version) if telemetry is not None else None
        d_densify_prune_called = False
        allocator_peak_before_debug = None
        debug_package = None
        checkpoint_payload = None
        target = None
        visibility = None
        radii = None
        telemetry_record = None
        telemetry_peak_reset = telemetry is not None or bool(opt.specular_smoke_diagnostics)
        if telemetry is not None:
            # Reset allocator counters only. This launches no CUDA work and
            # gives ordinary telemetry steps an exact per-loop peak scope.
            torch.cuda.reset_peak_memory_stats()
            allocator_peak_scope = f"since_explicit_telemetry_step_start_reset_at_global_{iteration}"
        elif opt.specular_smoke_diagnostics:
            torch.cuda.reset_peak_memory_stats()
            allocator_peak_scope = f"since_explicit_step_start_reset_at_global_{iteration}"
        reflection_iteration += 1
        diffuse.update_learning_rate(iteration)
        reflection.update_learning_rate(reflection_iteration)
        if not viewpoints:
            viewpoints = all_train_cameras.copy()
            indices = list(range(len(all_train_cameras)))
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
        if opt.specular_smoke_diagnostics:
            # The existing diagnostic raytrace resets allocator peaks internally.
            allocator_peak_scope = f"since_internal_training_raytrace_reset_at_global_{iteration}"
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
            loss_value = float(loss.item())
            telemetry_l_spec_value = (
                float(specular_loss.item())
                if telemetry is not None and opt.lambda_spec > 0 else None
            )
            ema_loss = 0.4 * loss_value + 0.6 * ema_loss
            specular_smoke_record = None
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
                specular_smoke_record = record
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": f"{ema_loss:.7f}", "D": diffuse.get_xyz.shape[0], "R": reflection.get_xyz.shape[0]})
                progress.update(10)
            if writer:
                writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
                writer.add_scalar("train_loss_patches/normal_depth", normal_loss.item(), iteration)
                writer.add_scalar("train_loss_patches/monocular_normal", mono_loss.item(), iteration)
                writer.add_scalar("train_loss_patches/perceptual", perceptual_loss.item(), iteration)
                writer.add_scalar(
                    "train_loss_patches/specular",
                    telemetry_l_spec_value
                    if telemetry_l_spec_value is not None else specular_loss.item(),
                    iteration,
                )
                writer.add_scalar("scene/diffuse_count", diffuse.get_xyz.shape[0], iteration)
                writer.add_scalar("scene/reflection_count", reflection.get_xyz.shape[0], iteration)

            _report(iteration, testing_iterations, scene, state, pipe, background)
            if iteration % opt.debug_interval == 0 or iteration == opt.iterations:
                if telemetry is not None:
                    # The existing debug raytrace resets peak counters. Save
                    # the pre-reset segment and combine it with the final
                    # segment below; no synchronization is introduced.
                    allocator_peak_before_debug = cuda_allocator_snapshot(
                        True,
                        f"telemetry_step_start_to_pre_debug_global_{iteration}",
                    )
                fixed = (scene.getTestCameras() or scene.getTrainCameras())[0]
                debug_package = render(
                    fixed, state, pipe, background, return_ray_diagnostics=True
                )
                telemetry_peak_reset = True
                allocator_peak_scope = f"since_internal_debug_raytrace_reset_at_global_{iteration}"
                directory = os.path.join(scene.model_path, "debug", f"iteration_{iteration:06d}")
                save_reflection_debug_maps(
                    debug_package,
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
                    d_densify_prune_called = True
                    threshold = 20 if iteration > opt.opacity_reset_interval else None
                    diffuse.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, threshold, radii)
                if iteration % opt.opacity_reset_interval == 0:
                    diffuse.reset_opacity()

            optimizer_step_completed = should_step_optimizer(
                iteration, opt.iterations, opt.operator_gate_continuation
            )
            if optimizer_step_completed:
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
                checkpoint_payload = make_stage_b_checkpoint(
                    diffuse,
                    reflection,
                    iteration,
                    reflection_iteration,
                    provenance,
                    _config(dataset, opt, diffuse, mask_manifest),
                    runtime_state=(
                        make_camera_runtime_state(indices, len(all_train_cameras))
                        if opt.operator_gate_continuation else None
                    ),
                    optimizer_step_completed=(
                        optimizer_step_completed if opt.operator_gate_continuation else None
                    ),
                )
                torch.save(
                    checkpoint_payload,
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth"),
                )

            if telemetry is not None:
                # This CPU wall boundary includes the normal loop work through
                # debug/PLY/checkpoint and optimizer/densification, but excludes
                # telemetry statistics and JSON serialization.  It deliberately
                # performs no CUDA synchronize.
                whole_step_wall_ms = float((time.perf_counter() - telemetry_wall_start) * 1000.0)
                d_count = int(diffuse.get_xyz.shape[0])
                r_count = int(reflection.get_xyz.shape[0])
                r_topology_version = int(reflection.topology_version)
                mask_stats = mask_ks_stats_from_existing_tensors(
                    package["surface_ks"],
                    camera.specular_mask if opt.lambda_spec > 0 else None,
                    package["valid_surface_mask"],
                )
                candidate = exact = timing = None
                if specular_smoke_record is not None:
                    candidate = specular_smoke_record["ray"]["candidate_count"]
                    exact = specular_smoke_record["ray"]["exact_intersection_count"]
                    timing = specular_smoke_record["ray"]["timing_ms"]
                unavailable = ["candidate_backward_ms"]
                if candidate is None:
                    unavailable.extend(("candidate_p50", "candidate_p95", "candidate_p99"))
                if exact is None:
                    unavailable.extend(("exact_p50", "exact_p95", "exact_p99"))
                if timing is None:
                    unavailable.append("raytrace_forward_ms")
                if opt.lambda_spec <= 0:
                    unavailable.extend(("l_spec", "mask_support_fraction", "ks_inside", "ks_outside"))
                allocator = cuda_allocator_snapshot(telemetry_peak_reset, allocator_peak_scope)
                if allocator_peak_before_debug is not None:
                    for key in (
                        "cuda_max_memory_allocated_bytes",
                        "cuda_max_memory_reserved_bytes",
                    ):
                        allocator[key] = max(allocator_peak_before_debug[key], allocator[key])
                    allocator["cuda_peak_scope"] = (
                        f"combined telemetry_step_start_to_pre_debug and "
                        f"internal_debug_reset_to_step_end_at_global_{iteration}; "
                        "transient_debug_work_before_internal_reset_is_unavailable"
                    )
                l_spec_value = telemetry_l_spec_value
                observed_nonfinite = int(not math.isfinite(loss_value)) + int(mask_stats["nonfinite_count"])
                telemetry_record = {
                    "schema_version": TELEMETRY_SCHEMA_VERSION,
                    "telemetry_step": telemetry.count + 1,
                    "global_iteration": int(iteration),
                    "reflection_local_iteration": int(reflection_iteration),
                    "camera_stem": str(camera.image_name),
                    "phase_tag": telemetry.phase_tag,
                    "total_loss": loss_value,
                    "l_spec": l_spec_value,
                    "d_count": d_count,
                    "r_count": r_count,
                    "d_count_delta": d_count - d_count_before,
                    "r_count_delta": r_count - r_count_before,
                    "d_topology_event": bool(d_densify_prune_called or d_count != d_count_before),
                    "r_topology_version": r_topology_version,
                    "r_topology_event": bool(r_topology_version != r_topology_before),
                    "mask_support_fraction": mask_stats["mask_support_fraction"],
                    "ks_inside": mask_stats["ks_inside"],
                    "ks_outside": mask_stats["ks_outside"],
                    "ks_state_definition": (
                        "pre_optimizer_forward_state_used_by_total_loss; no post-update rerender"
                    ),
                    "candidate_p50": None if candidate is None else float(candidate["p50"]),
                    "candidate_p95": None if candidate is None else float(candidate["p95"]),
                    "candidate_p99": None if candidate is None else float(candidate["p99"]),
                    "exact_p50": None if exact is None else float(exact["p50"]),
                    "exact_p95": None if exact is None else float(exact["p95"]),
                    "exact_p99": None if exact is None else float(exact["p99"]),
                    "raytrace_forward_ms": None if timing is None else float(timing["raytrace_wall"]),
                    "candidate_backward_ms": None,
                    "whole_step_wall_ms": whole_step_wall_ms,
                    "whole_step_wall_definition": (
                        "CPU perf_counter from loop entry through normal debug/PLY/checkpoint and "
                        "optimizer/densification; excludes telemetry work; no added CUDA synchronize"
                    ),
                    **allocator,
                    "nonfinite_count": observed_nonfinite,
                    "nonfinite_scope": (
                        "total_loss scalar plus finite check of existing ks samples; renderer retains "
                        "its existing hard finite checks; model-wide scan unavailable without extra GPU work"
                    ),
                    "unavailable_fields": sorted(set(unavailable)),
                }
                # Keep the nested ks schema explicit even if future internal
                # helpers return additional bookkeeping fields.
                for name in ("ks_inside", "ks_outside"):
                    if telemetry_record[name] is not None:
                        telemetry_record[name] = {
                            key: telemetry_record[name][key] for key in KS_STAT_KEYS
                        }

            if allocator_policy == V2_POLICY:
                # The next RHS `package = render(...)` would otherwise overlap
                # with the previous step's still-referenced output/graph.  Debug
                # and checkpoint payloads are also intentionally bounded to the
                # step that created them.  Deleting references and releasing
                # only unused cached blocks changes no tensor values, RNG, or
                # optimizer state.
                step_peak_allocated = int(torch.cuda.max_memory_allocated())
                allocator_release_wall_start = time.perf_counter()
                del package, image, gt, l1_value, ssim_value, loss
                del normal_loss, mono_loss, perceptual_loss, specular_loss
                del aux, specular_probe, debug_package, checkpoint_payload
                del target, visibility, radii
                release_allocator_cache(allocator_policy)
                allocator_release_wall_ms = float(
                    (time.perf_counter() - allocator_release_wall_start) * 1000.0
                )
                if telemetry_record is not None:
                    telemetry_record["whole_step_wall_ms"] += allocator_release_wall_ms
                    telemetry_record["whole_step_wall_definition"] = (
                        "CPU perf_counter from loop entry through normal debug/PLY/checkpoint and "
                        "optimizer/densification plus v2 allocator release; excludes telemetry "
                        "statistics/JSON serialization; no telemetry-added CUDA synchronize"
                    )

            if telemetry_record is not None:
                telemetry.append(telemetry_record)

            if (
                allocator_policy == V2_POLICY
                and reflection_iteration == 101
                and opt.lambda_spec > 0
            ):
                device_free, device_total = torch.cuda.mem_get_info()
                headroom = evaluate_headroom(
                    experiment=dataset.experiment,
                    retry_identity=opt.tier2_retry_identity,
                    policy=allocator_policy,
                    global_iteration=iteration,
                    reflection_local_iteration=reflection_iteration,
                    current_allocated_bytes=int(torch.cuda.memory_allocated()),
                    current_reserved_bytes=int(torch.cuda.memory_reserved()),
                    device_free_bytes=int(device_free),
                    device_total_bytes=int(device_total),
                    current_step_peak_allocated_bytes=step_peak_allocated,
                    reference_peak_allocated_bytes=int(
                        opt.stage_b_reference_peak_allocated_bytes
                    ),
                    minimum_projected_headroom_bytes=int(
                        opt.stage_b_minimum_projected_headroom_bytes
                    ),
                )
                headroom_path = os.path.join(
                    scene.model_path, "allocator_headroom_gate.json"
                )
                write_headroom_gate(headroom_path, headroom)
                print("\nSTAGE_B_HEADROOM " + json.dumps(
                    headroom, sort_keys=True, allow_nan=False
                ))
                if not headroom["passed"]:
                    raise RuntimeError(
                        "Stage B allocator headroom gate failed before formal long run"
                    )
    progress.close()
