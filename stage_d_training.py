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
from geometry.cuboid_space import (
    CuboidSpace, INSIDE, SUPPORT_CLASS_NAMES, SUPPORT_STRICT_INSIDE,
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
from utils.transmittance_debug import (
    make_semantic_repair_contact_sheet, make_stage_d_contact_sheet,
    save_transmittance_debug_maps,
)
from utils.stage_d_static_cache import (
    TRAINING_PACKAGE_KEYS, StaticDRCache, cpu_cache_payload, frozen_branch_hash,
    make_identity, renderer_contract, state_sha256, write_manifest,
    OWNERSHIP_CACHE_SCHEMA,
)
from utils.semantic_repair import (
    anti_veil_config, anti_veil_loss, smooth_ramp, spatial_frequency_energy,
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
SEMANTIC_OUTPUT_NAME = "stage_d_tihubird_c03r8_semantic_repair_v3"
SEMANTIC_NODES = (15025, 15100, 15500, 16000)
SEMANTIC_ENDPOINT = 16000
OWNERSHIP_OUTPUT_NAME = "stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4"
OWNERSHIP_ARM_NAMES = {
    "random_strict_inside": "arm_a_random_strict_inside",
    "transferred_d_inside": "arm_b_transferred_d_inside",
}
OWNERSHIP_NODES = (15000, 15100, 15250, 15500)
OWNERSHIP_ENDPOINT = 15500
OWNERSHIP_T_LONG_OUTPUT_NAME = (
    "stage_d_tihubird_c03r8_cuboid_path_ownership_tlong_g15500_g20000_v1"
)
OWNERSHIP_T_LONG_SOURCE_SHA256 = (
    "eff2135e19b48dd68f3661cbf23825d79deb0beaf29338887c0d5e8146127321"
)
OWNERSHIP_T_LONG_NODES = (15500, 16000, 17500, 20000)
OWNERSHIP_T_LONG_ENDPOINT = 20000
OWNERSHIP_T_LONG_GUARD_WINDOW = 100
OWNERSHIP_T_LONG_SATURATION_LIMIT = 0.50
OWNERSHIP_T_LONG_BLACK_LIMIT = 0.10
OWNERSHIP_T_LONG_CAPPED_FRACTION_LIMIT = 0.10
OWNERSHIP_T_LONG_MIN_SCALE_FACTOR = 0.02
TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME = (
    "stage_d_tihubird_c03r8_cuboid_path_ownership_trecover16000_preflight50_v3"
)
TSCALE_RECOVERY_LONG_OUTPUT_NAME = (
    "stage_d_tihubird_c03r8_cuboid_path_ownership_trecover16050_g20000_v3"
)
TSCALE_RECOVERY_SOURCE_SHA256 = (
    "52d1368dfb2a729240265e27f7696b0d230522af3fae795049c70932a673158e"
)
TSCALE_RECOVERY_PREFLIGHT_NODES = (16000, 16001, 16010, 16025, 16050)
TSCALE_RECOVERY_LONG_NODES = (16050,) + tuple(range(16250, 20001, 250))
TSCALE_RECOVERY_PREFLIGHT_ENDPOINT = 16050
TSCALE_RECOVERY_LONG_ENDPOINT = 20000
TSCALE_RECOVERY_SCALE_FACTOR_LIMIT = 0.90
TSCALE_RECOVERY_RAW_ACTIVE_ATOL = 2e-6


def _tscale_recovery_mode(opt):
    return bool(
        getattr(opt, "stage_d_tscale_recovery_preflight", False)
        or getattr(opt, "stage_d_tscale_recovery_long", False)
    )


def _cached_mode(opt):
    return bool(
        opt.stage_d_cached_twarmup or opt.stage_d_semantic_repair_pilot
        or opt.stage_d_ownership_pilot
        or getattr(opt, "stage_d_ownership_t_long", False)
        or _tscale_recovery_mode(opt)
    )


def _requires_cuboid_space(opt):
    """Stage D modes whose renderer/initialization contract requires the release cuboid."""
    return bool(
        opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
        or getattr(opt, "stage_d_ownership_t_long", False)
        or _tscale_recovery_mode(opt)
    )


def _required_nodes(opt):
    if getattr(opt, "stage_d_tscale_recovery_preflight", False):
        return TSCALE_RECOVERY_PREFLIGHT_NODES
    if getattr(opt, "stage_d_tscale_recovery_long", False):
        return TSCALE_RECOVERY_LONG_NODES
    if opt.stage_d_ownership_t_long:
        return OWNERSHIP_T_LONG_NODES
    if opt.stage_d_ownership_pilot:
        return OWNERSHIP_NODES
    return SEMANTIC_NODES if opt.stage_d_semantic_repair_pilot else FORMAL_NODES


def _telemetry_schema(opt):
    if _tscale_recovery_mode(opt):
        return "rtgs_stage_d_tscale_recovery_telemetry_v1"
    if opt.stage_d_ownership_t_long:
        return "rtgs_stage_d_ownership_t_long_telemetry_v1"
    if opt.stage_d_ownership_pilot:
        return "rtgs_stage_d_cuboid_path_ownership_telemetry_v4"
    if opt.stage_d_semantic_repair_pilot:
        return "rtgs_stage_d_semantic_repair_telemetry_v3"
    if opt.stage_d_cached_twarmup:
        return "rtgs_stage_d_cached_twarmup_telemetry_v1"
    if opt.stage_d_formal_onset:
        return "rtgs_stage_d_formal_telemetry_v2"
    return "rtgs_stage_d_smoke_telemetry_v1"


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
    recovery_preflight = bool(
        getattr(opt, "stage_d_tscale_recovery_preflight", False)
    )
    recovery_long = bool(getattr(opt, "stage_d_tscale_recovery_long", False))
    recovery = recovery_preflight or recovery_long
    recovery_start = 16001 if recovery_preflight else 16051
    recovery_endpoint = (
        TSCALE_RECOVERY_PREFLIGHT_ENDPOINT
        if recovery_preflight else TSCALE_RECOVERY_LONG_ENDPOINT
    )
    recovery_updates = 50 if recovery_preflight else 3950
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
        "stage_d_semantic_repair_pilot": bool(opt.stage_d_semantic_repair_pilot),
        "stage_d_ownership_pilot": bool(opt.stage_d_ownership_pilot),
        "stage_d_ownership_arm": str(opt.stage_d_ownership_arm),
        "stage_d_ownership_t_long": bool(opt.stage_d_ownership_t_long),
        "stage_d_tscale_recovery_preflight": recovery_preflight,
        "stage_d_tscale_recovery_long": recovery_long,
        "transparent_path_mode": str(dataset.transparent_path_mode),
        "transparent_direct_mode": str(dataset.transparent_direct_mode),
        "transparent_reflection_mode": str(dataset.transparent_reflection_mode),
        "cout_ownership_mode": str(dataset.cout_ownership_mode),
        "cached_t_warmup": (
            {
                "phase_a_global": [
                    recovery_start if recovery else (
                        15501 if opt.stage_d_ownership_t_long else 15001
                    ),
                    recovery_endpoint if recovery else (
                        OWNERSHIP_T_LONG_ENDPOINT if opt.stage_d_ownership_t_long else (
                    OWNERSHIP_ENDPOINT if opt.stage_d_ownership_pilot else (
                    SEMANTIC_ENDPOINT if opt.stage_d_semantic_repair_pilot
                    else CACHED_PHASE_A_END
                    )))
                ],
                "phase_a_updates": {
                    "diffuse": 0, "reflection": 0,
                    "transmittance": (
                        recovery_updates if recovery else (
                        4500 if opt.stage_d_ownership_t_long else (
                        500 if opt.stage_d_ownership_pilot else (
                        1000 if opt.stage_d_semantic_repair_pilot else 3000)
                    ))),
                },
                "phase_b_global": (
                    None if (
                        opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
                        or opt.stage_d_ownership_t_long or recovery
                    )
                    else [CACHED_PHASE_A_END + 1, 20000]
                ),
                "phase_b_mode": (
                    None if (
                        opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
                        or opt.stage_d_ownership_t_long or recovery
                    )
                    else "exact_uncached_joint_d_r_t"
                ),
                "reflection_local_semantics": (
                    "R-local remains 12000 for the entire ownership/semantic pilot; no Phase B"
                    if (
                        opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
                        or opt.stage_d_ownership_t_long or recovery
                    ) else
                    "R-local remains 12000 in Phase A and advances only for Phase-B optimizer updates"
                ),
                "cache_schema": (
                    OWNERSHIP_CACHE_SCHEMA
                    if (opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long or recovery)
                    else "rtgs_stage_d_static_dr_cache_v1"
                ),
                "cache_fp": "float32",
                "cache_contains_target_rgb": False,
            }
            if _cached_mode(opt) else None
        ),
        "semantic_repair": (
            {
                "schema": "rtgs_stage_d_semantic_repair_v3",
                "cuboid_space": getattr(dataset, "_semantic_cuboid_space_metadata", None),
                "r_transparent_filter": "outside_only",
                "cout_filter": "outside_only",
                "t_position_parameterization": "cuboid_inside_sigmoid_v1",
                "t_topology": {"densification": False, "pruning": False, "required_count": 4096},
                "anti_veil": anti_veil_config(opt),
                "pilot_global": [15001, SEMANTIC_ENDPOINT],
            }
            if opt.stage_d_semantic_repair_pilot else None
        ),
        "ownership_handoff": (
            {
                "schema": "rtgs_stage_d_cuboid_path_ownership_v4",
                "transparent_path_mode": dataset.transparent_path_mode,
                "transparent_direct_mode": dataset.transparent_direct_mode,
                "transparent_reflection_mode": dataset.transparent_reflection_mode,
                "cout_ownership_mode": dataset.cout_ownership_mode,
                "support_classification": "cuboid_local_finite_3sigma_v1",
                "support_sigma": 3.0,
                "t_topology": {
                    "densification": False, "pruning": False,
                    "required_count": 4096,
                    "position_parameterization": "cuboid_inside_support_sigmoid_v2",
                    "scaling_parameterization": (
                        "cuboid_support_projected_cap_v2"
                        if recovery else "cuboid_support_uniform_cap_v1"
                    ),
                },
                "arm": opt.stage_d_ownership_arm,
                "transfer_selection": {
                    "minimum_views": int(opt.transfer_min_views),
                    "minimum_total_weight": float(opt.transfer_min_total_weight),
                    "depth_margin": float(opt.transfer_depth_margin),
                    "opacity_scale": float(opt.transfer_opacity_scale),
                    "opacity_min": float(opt.transfer_opacity_min),
                    "opacity_max": float(opt.transfer_opacity_max),
                },
                "pilot_global": [15001, OWNERSHIP_ENDPOINT],
                "depth_enabled": False,
                "diagnostic_isolation_not_final_physics": True,
            }
            if (opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long or recovery)
            else None
        ),
        "ownership_t_long": (
            {
                "schema": "rtgs_stage_d_ownership_t_long_v1",
                "source_checkpoint_sha256": getattr(
                    dataset, "_stage_d_start_checkpoint_sha256", None,
                ),
                "global": [15501, OWNERSHIP_T_LONG_ENDPOINT],
                "t_local": [501, 5000],
                "updates": {"diffuse": 0, "reflection": 0, "transmittance": 4500},
                "review_nodes": list(OWNERSHIP_T_LONG_NODES),
                "depth_enabled": False,
                "guard": {
                    "window": OWNERSHIP_T_LONG_GUARD_WINDOW,
                    "saturation_fraction_limit": OWNERSHIP_T_LONG_SATURATION_LIMIT,
                    "high_ain_near_black_fraction_limit": OWNERSHIP_T_LONG_BLACK_LIMIT,
                    "capped_fraction_limit": OWNERSHIP_T_LONG_CAPPED_FRACTION_LIMIT,
                    "minimum_scale_factor": OWNERSHIP_T_LONG_MIN_SCALE_FACTOR,
                },
                "semantic_claim": False,
            }
            if opt.stage_d_ownership_t_long else None
        ),
        "tscale_recovery": (
            {
                "schema": "rtgs_stage_d_tscale_recovery_v1",
                "profile": "preflight50" if recovery_preflight else "long",
                "source_checkpoint_sha256": getattr(
                    dataset, "_stage_d_start_checkpoint_sha256", None,
                ),
                "global": [recovery_start, recovery_endpoint],
                "t_local": [recovery_start - 15000, recovery_endpoint - 15000],
                "updates": {"diffuse": 0, "reflection": 0,
                            "transmittance": recovery_updates},
                "review_nodes": list(
                    TSCALE_RECOVERY_PREFLIGHT_NODES
                    if recovery_preflight else TSCALE_RECOVERY_LONG_NODES
                ),
                "scaling_parameterization": "cuboid_support_projected_cap_v2",
                "projection": "raw_log_scale_equals_forward_active_scale_after_each_update",
                "affected_adam_state": "zero_exp_avg_exp_avg_sq_rows_only",
                "depth_enabled": False,
                "semantic_claim": False,
            }
            if recovery else None
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
            if (opt.stage_d_formal_onset or _cached_mode(opt)) else None
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
    allowed_t_init = {"random_bbox"}
    if opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long \
            or _tscale_recovery_mode(opt):
        allowed_t_init = {"random_strict_inside", "transferred_d_inside"}
    if dataset.transmittance_init_mode not in allowed_t_init:
        raise ValueError(f"Stage D T initialization must be one of {sorted(allowed_t_init)}")
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
    if _cached_mode(opt) and (opt.stage_d_smoke or opt.stage_d_formal_onset):
        raise ValueError("cached T warm-up is a distinct formal mode")
    modes = sum(bool(value) for value in (
        opt.stage_d_cached_twarmup, opt.stage_d_semantic_repair_pilot,
        opt.stage_d_ownership_pilot, opt.stage_d_ownership_t_long,
        getattr(opt, "stage_d_tscale_recovery_preflight", False),
        getattr(opt, "stage_d_tscale_recovery_long", False),
    ))
    if modes > 1:
        raise ValueError("Stage D cached/semantic/ownership modes are mutually exclusive")
    if getattr(opt, "stage_d_tscale_recovery_preflight", False):
        expected_phase_end = TSCALE_RECOVERY_PREFLIGHT_ENDPOINT
    elif getattr(opt, "stage_d_tscale_recovery_long", False):
        expected_phase_end = TSCALE_RECOVERY_LONG_ENDPOINT
    elif opt.stage_d_ownership_t_long:
        expected_phase_end = OWNERSHIP_T_LONG_ENDPOINT
    elif opt.stage_d_ownership_pilot:
        expected_phase_end = OWNERSHIP_ENDPOINT
    elif opt.stage_d_semantic_repair_pilot:
        expected_phase_end = SEMANTIC_ENDPOINT
    else:
        expected_phase_end = CACHED_PHASE_A_END
    if int(opt.stage_d_phase_a_end_iteration) != expected_phase_end:
        raise ValueError(f"cached T warm-up Phase A must end at global {expected_phase_end}")
    if opt.stage_d_cache_parity_atol <= 0 or opt.stage_d_cache_parity_mean_atol <= 0:
        raise ValueError("cached T warm-up parity tolerances must be positive")
    if opt.transparent_interface_margin_mode != "exclude":
        raise ValueError("semantic repair requires transparent_interface_margin_mode=exclude")
    if opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot:
        if opt.lambda_anti_veil_black <= 0 or opt.lambda_anti_veil_saturation <= 0:
            raise ValueError("semantic repair requires positive anti-veil weights")
        if int(opt.anti_veil_ramp_end) <= int(opt.anti_veil_ramp_start):
            raise ValueError("semantic repair anti-veil ramp is invalid")
    if opt.stage_d_ownership_pilot:
        required = {
            "path": dataset.transparent_path_mode == "cuboid_front_v1",
            "direct": dataset.transparent_direct_mode == "off",
            "reflection": dataset.transparent_reflection_mode == "off",
            "cout": dataset.cout_ownership_mode == "support_safe_outside",
            "arm": opt.stage_d_ownership_arm in OWNERSHIP_ARM_NAMES,
            "depth": int(opt.stage_d_depth_start_iteration) == 40000,
            "transfer_views": int(opt.transfer_min_views) >= 2,
            "transfer_weight": float(opt.transfer_min_total_weight) > 0,
            "transfer_margin": float(opt.transfer_depth_margin) > 0,
        }
        failures = [name for name, passed in required.items() if not passed]
        if failures:
            raise ValueError("ownership pilot configuration mismatch: " + ", ".join(failures))
    if opt.stage_d_ownership_t_long:
        required = {
            "path": dataset.transparent_path_mode == "cuboid_front_v1",
            "direct": dataset.transparent_direct_mode == "off",
            "reflection": dataset.transparent_reflection_mode == "off",
            "cout": dataset.cout_ownership_mode == "support_safe_outside",
            "init_identity": dataset.transmittance_init_mode == "transferred_d_inside",
            "reuse_cache": bool(opt.stage_d_reuse_static_cache),
            "depth": int(opt.stage_d_depth_start_iteration) == 40000,
        }
        failures = [name for name, passed in required.items() if not passed]
        if failures:
            raise ValueError(
                "ownership T-long configuration mismatch: " + ", ".join(failures)
            )
    if _tscale_recovery_mode(opt):
        required = {
            "path": dataset.transparent_path_mode == "cuboid_front_v1",
            "direct": dataset.transparent_direct_mode == "off",
            "reflection": dataset.transparent_reflection_mode == "off",
            "cout": dataset.cout_ownership_mode == "support_safe_outside",
            "init_identity": dataset.transmittance_init_mode == "transferred_d_inside",
            "reuse_cache": bool(opt.stage_d_reuse_static_cache),
            "depth": int(opt.stage_d_depth_start_iteration) == 40000,
        }
        failures = [name for name, passed in required.items() if not passed]
        if failures:
            raise ValueError(
                "T-scale recovery configuration mismatch: " + ", ".join(failures)
            )


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


def _validate_semantic_repair_contract(
    dataset, opt, release, source, fresh_from_stage_b,
    saving_iterations, checkpoint_iterations,
):
    if not opt.stage_d_semantic_repair_pilot:
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
        "endpoint": int(opt.iterations) == SEMANTIC_ENDPOINT,
        "phase_end": int(opt.stage_d_phase_a_end_iteration) == SEMANTIC_ENDPOINT,
        "depth_disabled": int(opt.stage_d_depth_start_iteration) == 40000,
        "lambda_spec": float(opt.lambda_spec) == 0.2,
        "specular_k0": float(opt.specular_k0) == 0.9,
        "resolution": int(dataset.resolution) == 8,
        "ray_chunk": int(dataset.ray_chunk_size) == 2048,
        "t_init": (
            dataset.transmittance_init_mode == "random_bbox"
            and int(dataset.transmittance_init_count) == 4096
            and int(dataset.transmittance_init_seed) == 20260703
        ),
        "output": Path(dataset.model_path).name == SEMANTIC_OUTPUT_NAME,
        "checkpoint_nodes": tuple(sorted(set(checkpoint_iterations))) == SEMANTIC_NODES,
        "ply_nodes": tuple(sorted(set(saving_iterations))) == SEMANTIC_NODES,
        "margin_mode": dataset._semantic_cuboid_space_metadata.get(
            "transparent_interface_margin_mode"
        ) == "exclude",
        "source_specular_contract": (
            source_config.get("lambda_spec") == 0.2
            and source_config.get("specular_k0") == 0.9
            and source_mask.get("manifest_file_sha256")
            == "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
            and current_mask.get("manifest_file_sha256")
            == source_mask.get("manifest_file_sha256")
            and current_mask.get("aggregate_sha256") == source_mask.get("aggregate_sha256")
        ),
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise ValueError("semantic repair pilot contract mismatch: " + ", ".join(failures))


def _validate_ownership_contract(
    dataset, opt, release, source, fresh_from_stage_b,
    saving_iterations, checkpoint_iterations,
):
    if not opt.stage_d_ownership_pilot:
        return
    arm_name = OWNERSHIP_ARM_NAMES.get(opt.stage_d_ownership_arm)
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
        "endpoint": int(opt.iterations) == OWNERSHIP_ENDPOINT,
        "phase_end": int(opt.stage_d_phase_a_end_iteration) == OWNERSHIP_ENDPOINT,
        "depth_disabled": int(opt.stage_d_depth_start_iteration) == 40000,
        "resolution": int(dataset.resolution) == 8,
        "ray_chunk": int(dataset.ray_chunk_size) == 2048,
        "t_count_seed": int(dataset.transmittance_init_count) == 4096
        and int(dataset.transmittance_init_seed) == 20260703,
        "arm_init_match": dataset.transmittance_init_mode == opt.stage_d_ownership_arm,
        "output": arm_name is not None and Path(dataset.model_path).name == arm_name
        and Path(dataset.model_path).parent.name == OWNERSHIP_OUTPUT_NAME,
        "checkpoint_nodes": tuple(sorted(set(checkpoint_iterations))) == OWNERSHIP_NODES,
        "ply_nodes": tuple(sorted(set(saving_iterations))) == OWNERSHIP_NODES,
        "path": dataset.transparent_path_mode == "cuboid_front_v1",
        "direct_off": dataset.transparent_direct_mode == "off",
        "reflection_off": dataset.transparent_reflection_mode == "off",
        "cout_safe": dataset.cout_ownership_mode == "support_safe_outside",
        "cuboid_space": (
            getattr(dataset, "_semantic_cuboid_space_metadata", {}).get("schema")
            == "rtgs_cuboid_space_v1"
        ),
        "cache_path": Path(dataset.stage_d_static_cache_path).name == "cuboid_front_cache_v4",
        "source_specular_contract": (
            source_config.get("lambda_spec") == 0.2
            and source_config.get("specular_k0") == 0.9
            and current_mask.get("manifest_file_sha256")
            == source_mask.get("manifest_file_sha256")
            == "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
            and current_mask.get("aggregate_sha256") == source_mask.get("aggregate_sha256")
        ),
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise ValueError("cuboid path ownership pilot contract mismatch: " + ", ".join(failures))


def _validate_ownership_t_long_contract(
    dataset, opt, release, source, fresh_from_stage_b, saved_config,
    global_iteration, reflection_iteration, transmittance_iteration,
    saving_iterations, checkpoint_iterations,
):
    if not opt.stage_d_ownership_t_long:
        return
    prior = (saved_config or {}).get("ownership_handoff", {})
    expected_cache = Path(
        "output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4/"
        "cuboid_front_cache_v4"
    ).resolve()
    required = {
        "stage_d_resume": not fresh_from_stage_b,
        "start_sha256": getattr(dataset, "_stage_d_start_checkpoint_sha256", None)
        == OWNERSHIP_T_LONG_SOURCE_SHA256,
        "start_global": int(global_iteration) == 15500,
        "start_r_local": int(reflection_iteration) == 12000,
        "start_t_local": int(transmittance_iteration) == 500,
        "source_sha256": source.get("sha256") == FORMAL_SOURCE_SHA256,
        "release_id": release.manifest.get("geometry_release_id") == FORMAL_RELEASE_ID,
        "release_sha256": release.validation.get("aggregate_sha256") == FORMAL_RELEASE_SHA256,
        "endpoint": int(opt.iterations) == OWNERSHIP_T_LONG_ENDPOINT,
        "phase_end": int(opt.stage_d_phase_a_end_iteration) == OWNERSHIP_T_LONG_ENDPOINT,
        "depth_off": int(opt.stage_d_depth_start_iteration) == 40000,
        "resolution": int(dataset.resolution) == 8,
        "ray_chunk": int(dataset.ray_chunk_size) == 2048,
        "output": Path(dataset.model_path).name == OWNERSHIP_T_LONG_OUTPUT_NAME,
        "cache": Path(dataset.stage_d_static_cache_path).resolve() == expected_cache,
        "checkpoint_nodes": tuple(sorted(set(checkpoint_iterations)))
        == OWNERSHIP_T_LONG_NODES,
        "ply_nodes": tuple(sorted(set(saving_iterations))) == OWNERSHIP_T_LONG_NODES,
        "prior_path": prior.get("transparent_path_mode") == "cuboid_front_v1",
        "prior_direct": prior.get("transparent_direct_mode") == "off",
        "prior_reflection": prior.get("transparent_reflection_mode") == "off",
        "prior_cout": prior.get("cout_ownership_mode") == "support_safe_outside",
        "prior_arm": prior.get("arm") == "transferred_d_inside",
        "prior_scale": prior.get("t_topology", {}).get("scaling_parameterization")
        == "cuboid_support_uniform_cap_v1",
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise ValueError(
            "ownership T-long contract mismatch: " + ", ".join(failures)
        )


def _validate_tscale_recovery_contract(
    dataset, opt, release, source, fresh_from_stage_b, saved_config,
    global_iteration, reflection_iteration, transmittance_iteration,
    saving_iterations, checkpoint_iterations,
):
    if not _tscale_recovery_mode(opt):
        return
    preflight = bool(opt.stage_d_tscale_recovery_preflight)
    prior = (saved_config or {}).get("ownership_handoff", {})
    prior_recovery = (saved_config or {}).get("tscale_recovery")
    expected_cache = Path(
        "output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4/"
        "cuboid_front_cache_v4"
    ).resolve()
    expected_start = 16000 if preflight else 16050
    expected_t_local = expected_start - 15000
    expected_endpoint = (
        TSCALE_RECOVERY_PREFLIGHT_ENDPOINT if preflight else TSCALE_RECOVERY_LONG_ENDPOINT
    )
    expected_output = (
        TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME if preflight else TSCALE_RECOVERY_LONG_OUTPUT_NAME
    )
    expected_nodes = (
        TSCALE_RECOVERY_PREFLIGHT_NODES if preflight else TSCALE_RECOVERY_LONG_NODES
    )
    source_path = Path(dataset._stage_d_start_checkpoint_path).resolve() \
        if getattr(dataset, "_stage_d_start_checkpoint_path", "") else None
    preflight_source = Path(
        "output", TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME, "chkpnt16050.pth"
    ).resolve()
    required = {
        "stage_d_resume": not fresh_from_stage_b,
        "start_hash": (
            getattr(dataset, "_stage_d_start_checkpoint_sha256", None)
            == TSCALE_RECOVERY_SOURCE_SHA256
            if preflight else True
        ),
        "long_source_path": preflight or source_path == preflight_source,
        "start_global": int(global_iteration) == expected_start,
        "start_r_local": int(reflection_iteration) == 12000,
        "start_t_local": int(transmittance_iteration) == expected_t_local,
        "source_sha256": source.get("sha256") == FORMAL_SOURCE_SHA256,
        "release_id": release.manifest.get("geometry_release_id") == FORMAL_RELEASE_ID,
        "release_sha256": release.validation.get("aggregate_sha256") == FORMAL_RELEASE_SHA256,
        "endpoint": int(opt.iterations) == expected_endpoint,
        "phase_end": int(opt.stage_d_phase_a_end_iteration) == expected_endpoint,
        "depth_off": int(opt.stage_d_depth_start_iteration) == 40000,
        "resolution": int(dataset.resolution) == 8,
        "ray_chunk": int(dataset.ray_chunk_size) == 2048,
        "output": Path(dataset.model_path).name == expected_output,
        "cache": Path(dataset.stage_d_static_cache_path).resolve() == expected_cache,
        "checkpoint_nodes": tuple(sorted(set(checkpoint_iterations))) == expected_nodes,
        "ply_nodes": tuple(sorted(set(saving_iterations))) == expected_nodes,
        "prior_path": prior.get("transparent_path_mode") == "cuboid_front_v1",
        "prior_direct": prior.get("transparent_direct_mode") == "off",
        "prior_reflection": prior.get("transparent_reflection_mode") == "off",
        "prior_cout": prior.get("cout_ownership_mode") == "support_safe_outside",
        "prior_arm": prior.get("arm") == "transferred_d_inside",
        "prior_scale": prior.get("t_topology", {}).get("scaling_parameterization")
        in ("cuboid_support_uniform_cap_v1", "cuboid_support_projected_cap_v2"),
        "long_recovery_source": preflight or (
            isinstance(prior_recovery, dict)
            and prior_recovery.get("profile") == "preflight50"
            and prior_recovery.get("scaling_parameterization")
            == "cuboid_support_projected_cap_v2"
        ),
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise ValueError(
            "T-scale recovery contract mismatch: " + ", ".join(failures)
        )


def _phase_for_iteration(
    iteration, semantic_repair=False, ownership=False, ownership_t_long=False,
    tscale_recovery=False,
):
    if tscale_recovery:
        return "cuboid_path_ownership_tscale_recovery"
    if ownership_t_long:
        return "cuboid_path_ownership_t_long"
    if ownership:
        return "cuboid_path_ownership_t_only"
    if semantic_repair:
        return "semantic_repair_cached_t_only"
    return "cached_t_warmup" if int(iteration) <= CACHED_PHASE_A_END else "exact_joint"


def _transmittance_topology_update_allowed(opt, transmittance_iteration):
    """Keep the semantic pilot's fixed-cardinality T contract fail closed."""
    if (
        opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
        or opt.stage_d_ownership_t_long or _tscale_recovery_mode(opt)
    ):
        return False
    return (
        transmittance_iteration > opt.transmittance_densify_from_iter
        and transmittance_iteration < opt.transmittance_densify_until_iter
        and transmittance_iteration % opt.transmittance_densification_interval == 0
    )


def _masked_values(value, mask):
    return value.detach()[mask.expand_as(value)].float()


def _quantiles(value):
    if not value.numel():
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(value.mean()),
        "p50": float(torch.quantile(value, 0.50)),
        "p95": float(torch.quantile(value, 0.95)),
        "p99": float(torch.quantile(value, 0.99)),
    }


def _semantic_step_metrics(package, gt, transparent_mask):
    """Cheap per-step hard metrics; node audits retain the full spatial maps."""
    mask = transparent_mask.detach().permute(1, 2, 0) >= 0.5
    valid = mask & (package["transmittance_valid"] > 0.5)
    ain = _masked_values(package["inside_alpha"], valid)
    cin = _masked_values(package["inside_color"], valid)
    conditional = _masked_values(package["conditional_inside_color"], valid)
    gt_hwc = gt.detach().permute(1, 2, 0)
    conditional_luma = (
        conditional.reshape(-1, 3)
        @ conditional.new_tensor((0.2126, 0.7152, 0.0722))
        if conditional.numel() else conditional.new_zeros((0,))
    )
    high_black = (
        (ain.reshape(-1) >= 0.80) & (conditional_luma < 0.08)
        if ain.numel() else ain.new_zeros((0,), dtype=torch.bool)
    )
    energies = {}
    for name in (
        "diffuse_contribution", "reflection_contribution", "reflection_unfiltered", "reflection_inside",
        "reflection_interface", "reflection_outside", "reflection_final_filtered",
        "transmittance_contribution", "outside_unfiltered", "outside_inside",
        "outside_interface", "outside_outside", "outside_final_filtered",
        "inside_contribution", "cout_contribution", "diffuse_unfiltered",
        "reflection_strict_inside_safe", "reflection_interface_margin",
        "reflection_strict_outside_safe", "reflection_crossing_or_ambiguous",
        "outside_strict_inside_safe", "outside_interface_margin",
        "outside_strict_outside_safe", "outside_crossing_or_ambiguous",
    ):
        if name in package:
            values = _masked_values(package[name], mask)
            energies[name] = float(values.abs().mean()) if values.numel() else 0.0
    d_energy = energies.get("diffuse_contribution", 0.0)
    t_energy = energies.get("transmittance_contribution", 0.0)
    glass_error = _masked_values(torch.abs(package["final"] - gt_hwc), mask)
    frequency = {
        branch: spatial_frequency_energy(package[name], mask)
        for branch, name in (
            ("diffuse", "diffuse_contribution"),
            ("reflection", "reflection_contribution"),
            ("transmittance", "transmittance_contribution"),
        ) if name in package
    }
    path_metrics = {}
    if "front_origin_tnear_error" in package:
        origin_error = _masked_values(package["front_origin_tnear_error"], valid)
        normal_dot = _masked_values(package["front_normal_faceforward_dot"], valid)
        plane = _masked_values(package["front_plane_residual"], valid)
        back = _masked_values(package["frozen_back_distance_residual"], valid)
        path_metrics = {
            "origin_tnear_max_abs": float(origin_error.abs().max()) if origin_error.numel() else None,
            "origin_tnear_mean_abs": float(origin_error.abs().mean()) if origin_error.numel() else None,
            "normal_faceforward_min_dot": float(normal_dot.min()) if normal_dot.numel() else None,
            "front_plane_residual_max": float(plane.max()) if plane.numel() else None,
            "back_tfar_residual_max": float(back.max()) if back.numel() else None,
        }
    return {
        "ain": _quantiles(ain),
        "ain_saturation_fraction_ge_0_95": float((ain >= 0.95).float().mean())
        if ain.numel() else 0.0,
        "high_ain_near_black_conditional_fraction": float(high_black.float().mean())
        if high_black.numel() else 0.0,
        "cin_energy": float(cin.abs().mean()) if cin.numel() else 0.0,
        "conditional_inside_luminance": _quantiles(conditional_luma),
        "contribution_energy": energies,
        "d_over_t_energy": float(d_energy / max(t_energy, 1e-12)),
        "spatial_frequency_energy": frequency,
        "transparent_rgb_l1": float(glass_error.mean()) if glass_error.numel() else 0.0,
        "t_spatial_counts": dict(package.get("t_spatial_counts", {})),
        "t_support_scale_cap": dict(package.get("t_support_scale_cap", {})),
        "r_spatial": dict(package.get("semantic_r_stats", {})),
        "cout_spatial": dict(package.get("semantic_cout_stats", {})),
        "r_filter": dict(package.get("semantic_r_filter_stats", {})),
        "cout_filter": dict(package.get("semantic_cout_filter_stats", {})),
        "cuboid_front_path": path_metrics,
        "transparent_ownership_modes": {
            "path": package.get("transparent_path_mode"),
            "direct": package.get("transparent_direct_mode"),
            "reflection": package.get("transparent_reflection_mode"),
        },
    }


def _set_branch_trainable(model, enabled):
    for name in (
        "_xyz", "_rotation", "_scaling", "_opacity", "_base_color",
        "_roughness", "_f0", "_ks", "_exposure", "_color",
    ):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and value.is_floating_point():
            value.requires_grad_(bool(enabled))


def _ownership_long_guard_result(window):
    if not window:
        return {}, []
    means = {
        name: sum(row[name] for row in window) / len(window)
        for name in ("saturation", "black", "capped_fraction")
    }
    minimum_factor = min(row["minimum_scale_factor"] for row in window)
    summary = {
        "window": len(window), **means, "minimum_scale_factor": minimum_factor,
    }
    failures = []
    if means["saturation"] >= OWNERSHIP_T_LONG_SATURATION_LIMIT:
        failures.append("Ain saturation")
    if means["black"] >= OWNERSHIP_T_LONG_BLACK_LIMIT:
        failures.append("high-Ain near-black collapse")
    if means["capped_fraction"] >= OWNERSHIP_T_LONG_CAPPED_FRACTION_LIMIT:
        failures.append("support-scale cap coverage")
    if minimum_factor <= OWNERSHIP_T_LONG_MIN_SCALE_FACTOR:
        failures.append("support-scale minimum factor")
    return summary, failures


def _tscale_recovery_guard_result(window):
    if not window:
        return {}, []
    means = {
        name: sum(row[name] for row in window) / len(window)
        for name in (
            "saturation", "black", "capped_fraction",
            "projection_affected_fraction", "t_energy", "cin_energy",
        )
    }
    summary = {
        "window": len(window), **means,
        "minimum_forward_factor": min(row["minimum_forward_factor"] for row in window),
        "minimum_pre_projection_factor": min(
            row["minimum_pre_projection_factor"] for row in window
        ),
        "maximum_post_projection_raw_active_abs": max(
            row["post_projection_raw_active_abs"] for row in window
        ),
    }
    failures = []
    if means["saturation"] >= OWNERSHIP_T_LONG_SATURATION_LIMIT:
        failures.append("Ain saturation")
    if means["black"] >= OWNERSHIP_T_LONG_BLACK_LIMIT:
        failures.append("high-Ain near-black collapse")
    if means["capped_fraction"] > 0:
        failures.append("post-projection capped support")
    if summary["minimum_forward_factor"] < 1.0 - TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
        failures.append("forward raw/active scale mismatch")
    if summary["minimum_pre_projection_factor"] < TSCALE_RECOVERY_SCALE_FACTOR_LIMIT:
        failures.append("single-update raw scale escape")
    if summary["maximum_post_projection_raw_active_abs"] > TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
        failures.append("post-update raw/active scale mismatch")
    if means["projection_affected_fraction"] >= OWNERSHIP_T_LONG_CAPPED_FRACTION_LIMIT:
        failures.append("scale projection coverage")
    if means["t_energy"] < 0.05:
        failures.append("T contribution collapse")
    if means["cin_energy"] < 0.03:
        failures.append("Cin collapse")
    return summary, failures


@torch.no_grad()
def _select_transferred_d_candidates(static_cache, diffuse, cuboid, opt, count=4096):
    """Aggregate source-bound multi-view front-ray evidence deterministically."""
    surfel_count = int(diffuse.get_xyz.shape[0])
    view_count = torch.zeros((surfel_count,), dtype=torch.int32)
    weight_sum = torch.zeros((surfel_count,), dtype=torch.float64)
    hit_count = torch.zeros((surfel_count,), dtype=torch.int64)
    evidence_views = 0
    for stem in sorted(static_cache.entries):
        evidence = static_cache.load(stem, device="cpu").get("transfer_evidence")
        if not evidence:
            raise RuntimeError(f"ownership cache lacks transfer evidence: {stem}")
        ids = evidence["surfel_ids"].long()
        if ids.numel():
            if int(ids.min()) < 0 or int(ids.max()) >= surfel_count:
                raise RuntimeError(f"transfer evidence surfel ID is out of range: {stem}")
            view_count[ids] += 1
            weight_sum[ids] += evidence["weight_sum"].double()
            hit_count[ids] += evidence["hit_count"].long()
        evidence_views += 1
    support = cuboid.support_masks(
        diffuse.get_xyz.detach(), diffuse.get_rotation.detach(),
        diffuse.get_scaling.detach(), sigma=3.0,
    )["strict_inside_safe"].detach().cpu()
    eligible = support & (view_count >= int(opt.transfer_min_views)) \
        & (weight_sum >= float(opt.transfer_min_total_weight))
    eligible_ids = eligible.nonzero(as_tuple=False)[:, 0].tolist()
    eligible_ids.sort(key=lambda index: (
        -int(view_count[index]), -float(weight_sum[index]), -int(hit_count[index]), int(index)
    ))
    selected = torch.tensor(eligible_ids[: int(count)], dtype=torch.long, device=diffuse.get_xyz.device)
    metadata = {
        "schema": "rtgs_d_inside_transfer_selection_v1",
        "evidence_view_count": evidence_views,
        "minimum_distinct_views": int(opt.transfer_min_views),
        "minimum_total_alpha_weight": float(opt.transfer_min_total_weight),
        "depth_safe_interval": (
            f"[t_near+{float(opt.transfer_depth_margin)},"
            f"t_far-{float(opt.transfer_depth_margin)}]"
        ),
        "support_class": "strict_inside_safe",
        "support_sigma": 3.0,
        "eligible_count": len(eligible_ids),
        "selected_count": int(selected.numel()),
        "requested_count": int(count),
        "random_fill_count": int(count - selected.numel()),
        "ranking": "distinct_views_desc,total_weight_desc,hit_count_desc,surfel_id_asc",
        "fill_strategy": "random_strict_inside_same_seed_v1",
    }
    return selected, metadata


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
        if state.semantic_repair:
            make_semantic_repair_contact_sheet(iteration_directory, stems)
        else:
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
        anti_black = image.new_zeros(())
        anti_saturation = image.new_zeros(())
        anti_ramp = 0.0
        anti_soft_saturation = image.new_zeros(())
        anti_soft_high = image.new_zeros(())
        if opt.stage_d_semantic_repair_pilot:
            mask_hwc = (
                camera.specular_mask.permute(1, 2, 0)
                * package["transmittance_valid"]
            )
            anti = anti_veil_loss(
                package["inside_color"], package["inside_alpha"],
                gt.permute(1, 2, 0), mask_hwc,
                gt_luminance_threshold=opt.anti_veil_gt_luminance_threshold,
                high_alpha_threshold=opt.anti_veil_high_alpha_threshold,
                black_luminance_threshold=opt.anti_veil_black_luminance_threshold,
                saturation_alpha_threshold=opt.anti_veil_saturation_alpha_threshold,
                target_saturation_coverage=opt.anti_veil_target_saturation_coverage,
                gate_temperature=opt.anti_veil_gate_temperature,
                black_temperature=opt.anti_veil_black_temperature,
                coverage_temperature=opt.anti_veil_coverage_temperature,
                epsilon=opt.anti_veil_epsilon,
            )
            anti_black, anti_saturation = anti["black"], anti["saturation"]
            anti_soft_saturation = anti["soft_saturation_coverage"]
            anti_soft_high = anti["soft_high_alpha_coverage"]
            anti_ramp = smooth_ramp(
                int(iteration) - 15000,
                opt.anti_veil_ramp_start, opt.anti_veil_ramp_end,
            )
        loss = (
            rgb_loss + opt.lambda_norm * normal_loss + opt.lambda_mono * mono_loss
            + opt.lambda_perc * perceptual_loss + opt.lambda_spec * specular_loss
            + (opt.lambda_depth * depth_loss if depth_enabled else 0.0)
            + anti_ramp * (
                opt.lambda_anti_veil_black * anti_black
                + opt.lambda_anti_veil_saturation * anti_saturation
            )
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
        "anti_veil_black": anti_black,
        "anti_veil_saturation": anti_saturation,
        "anti_veil_ramp": float(anti_ramp),
        "anti_veil_soft_saturation_coverage": anti_soft_saturation,
        "anti_veil_soft_high_alpha_coverage": anti_soft_high,
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
        if (opt.stage_d_formal_onset or _cached_mode(opt))
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


@torch.no_grad()
def _tscale_state_metrics(transmittance):
    raw = torch.exp(transmittance._scaling.detach())
    active = transmittance.get_scaling.detach()
    ratio = active / raw.clamp_min(torch.finfo(raw.dtype).tiny)
    classes = transmittance.cuboid_space.classify_support(
        transmittance.get_xyz.detach(), transmittance.get_rotation.detach(),
        active, sigma=3.0,
    )
    difference = (active - raw).abs()
    return {
        "schema": "cuboid_support_projected_cap_v2",
        "parameterization": transmittance.scaling_parameterization,
        "raw_scale_max": float(raw.max()),
        "active_scale_max": float(active.max()),
        "minimum_factor": float(ratio.min()),
        "capped_count": int(
            ratio.amin(dim=-1).lt(1.0 - TSCALE_RECOVERY_RAW_ACTIVE_ATOL).sum()
        ),
        "raw_active_max_abs": float(difference.max()),
        "raw_active_mean_abs": float(difference.mean()),
        "strict_inside_safe_count": int((classes == SUPPORT_STRICT_INSIDE).sum()),
        "count": int(raw.shape[0]),
    }


def _write_tscale_checkpoint_audit(
    model_path, checkpoint_path, checkpoint, frozen_hashes,
):
    branch_hashes = {
        branch: state_sha256(checkpoint[branch])
        for branch in ("diffuse", "reflection", "transmittance")
    }
    node = int(checkpoint["global_iteration"])
    record = {
        "schema": "rtgs_stage_d_tscale_checkpoint_audit_v1",
        "global_iteration": node,
        "reflection_iteration": int(checkpoint["reflection_iteration"]),
        "transmittance_iteration": int(checkpoint["transmittance_iteration"]),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "branch_hashes": branch_hashes,
        "frozen_hashes": dict(frozen_hashes),
        "diffuse_frozen": branch_hashes["diffuse"] == frozen_hashes["diffuse"],
        "reflection_frozen": branch_hashes["reflection"] == frozen_hashes["reflection"],
        "transmittance_scaling_parameterization": checkpoint["transmittance"].get(
            "scaling_parameterization"
        ),
        "geometry_release_id": checkpoint["geometry_release_id"],
        "geometry_release_aggregate_sha256": checkpoint[
            "geometry_release_aggregate_sha256"
        ],
    }
    target = Path(model_path) / "checkpoint_audits" / f"iteration_{node:06d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(target, record)
    if not record["diffuse_frozen"] or not record["reflection_frozen"]:
        raise RuntimeError("T-scale checkpoint audit detected frozen D/R drift")
    return record


def _write_tscale_abort_hashes(
    model_path, iteration, diffuse, reflection, transmittance, failures,
):
    record = {
        "schema": "rtgs_stage_d_tscale_abort_hashes_v1",
        "global_iteration": int(iteration),
        "failures": list(failures),
        "branch_hashes": {
            "diffuse": frozen_branch_hash(diffuse),
            "reflection": frozen_branch_hash(reflection),
            "transmittance": frozen_branch_hash(transmittance),
        },
        "t_scale": _tscale_state_metrics(transmittance),
    }
    _atomic_json(Path(model_path) / "tscale_recovery_abort_hashes.json", record)
    return record


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
    semantic_repair=False,
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
            # Candidate/hit diagnostics are required for the fixed review pack.
            # Avoid diagnostic synchronizations on the other 102 cache views.
            return_ray_diagnostics=bool(stem in CACHED_STEMS),
        )
        cacheable = _cacheable_static_inputs(static_inputs)
        if stem not in CACHED_STEMS:
            if semantic_repair:
                cacheable.pop("semantic_cout_components", None)
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
            "Cout": package["outside_color"].detach().clone(),
            "Aout": package["outside_alpha"].detach().clone(),
            "Dout": package["outside_depth"].detach().clone(),
            "Ct": package["transmittance_color"].detach().clone(),
            "At": package["transmittance_alpha"].detach().clone(),
            **({
                "front_position": package["front_position"].detach().clone(),
                "front_normal": package["front_normal"].detach().clone(),
                "frozen_camera_direction": package["frozen_camera_direction"].detach().clone(),
                "D_direct": package["diffuse_contribution"].detach().clone(),
                "R_contribution": package["reflection_contribution"].detach().clone(),
                "Cout_contribution": package["cout_contribution"].detach().clone(),
            } if "front_position" in package else {}),
        },
        "losses": {
            name: payload[name].detach().reshape(1).clone()
            for name in (
                "loss", "l1", "ssim", "rgb", "normal", "mono",
                "perceptual", "specular", "depth",
                "anti_veil_black", "anti_veil_saturation",
                "anti_veil_soft_saturation_coverage",
                "anti_veil_soft_high_alpha_coverage",
            )
        },
        "gradients": gradients,
    }


def _optimizer_group_hashes(model, *, include_parameter=True):
    result = {}
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        payload = {"state": model.optimizer.state.get(parameter, {})}
        if include_parameter:
            payload["parameter"] = parameter
        result[group["name"]] = state_sha256(payload)
    return result


def _run_tscale_migration_parity(
    target, cameras, static_cache, state, pipe, background, opt, perceptual,
    diffuse, reflection, transmittance,
):
    """Migrate legacy capped scale while proving fixed-input forward parity."""
    by_stem = {Path(str(camera.image_name)).stem: camera for camera in cameras}
    stem = "000039"
    camera = by_stem[stem]
    static_inputs = static_cache.load(stem, training_only=True)
    _zero_stage_d_gradients(diffuse, reflection, transmittance)
    d_hash_before = frozen_branch_hash(diffuse)
    r_hash_before = frozen_branch_hash(reflection)
    optimizer_before = _optimizer_group_hashes(transmittance)
    active_before = transmittance.get_scaling.detach().clone()
    world_before = transmittance.get_xyz.detach().clone()
    pre_payload = _forward_backward_stage_d(
        camera, state, pipe, background, opt, perceptual, 16001,
        static_inputs=static_inputs,
    )
    pre = _parity_snapshot(pre_payload, _t_gradients(transmittance))
    _zero_stage_d_gradients(diffuse, reflection, transmittance)

    migration = transmittance.project_raw_scaling_to_active_(migrate_legacy=True)
    state.mark_transmittance_updated()
    active_after = transmittance.get_scaling.detach().clone()
    world_after = transmittance.get_xyz.detach().clone()
    post_payload = _forward_backward_stage_d(
        camera, state, pipe, background, opt, perceptual, 16001,
        static_inputs=static_inputs,
    )
    post = _parity_snapshot(post_payload, _t_gradients(transmittance))
    _zero_stage_d_gradients(diffuse, reflection, transmittance)
    optimizer_after = _optimizer_group_hashes(transmittance)
    d_hash_after = frozen_branch_hash(diffuse)
    r_hash_after = frozen_branch_hash(reflection)

    comparisons = {
        "active_scale": _absolute_difference(active_before, active_after),
        "world_position": _absolute_difference(world_before, world_after),
        "outputs": {
            name: _absolute_difference(pre["outputs"][name], post["outputs"][name])
            for name in pre["outputs"]
        },
        "losses": {
            name: _absolute_difference(pre["losses"][name], post["losses"][name])
            for name in pre["losses"]
        },
    }
    failures = []
    for name in ("active_scale", "world_position"):
        difference = comparisons[name]
        if difference["max_abs"] > TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
            failures.append(name)
    for group in ("outputs", "losses"):
        for name, difference in comparisons[group].items():
            if (
                difference["max_abs"] > float(opt.stage_d_cache_parity_atol)
                or difference["mean_abs"] > float(opt.stage_d_cache_parity_mean_atol)
            ):
                failures.append(f"{group}:{name}")
    unchanged_optimizer_groups = {
        name: optimizer_before[name] == optimizer_after[name]
        for name in optimizer_before if name != "scaling"
    }
    if not all(unchanged_optimizer_groups.values()):
        failures.append("non-scaling Adam state changed")
    if d_hash_before != d_hash_after or r_hash_before != r_hash_after:
        failures.append("D/R state changed during T scale migration")
    if migration["affected_count"] <= 0:
        failures.append("legacy checkpoint did not expose expected capped scales")
    if migration["raw_active_max_abs_after"] > TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
        failures.append("post-migration raw/active mismatch")

    report = {
        "schema": "rtgs_stage_d_tscale_migration_parity_v1",
        "status": "PASS" if not failures else "TSCALE_RECOVERY_PREFLIGHT_BLOCKED",
        "camera_stem": stem,
        "optimizer_updates": 0,
        "migration": migration,
        "comparisons": comparisons,
        "optimizer_group_hashes_before": optimizer_before,
        "optimizer_group_hashes_after": optimizer_after,
        "unchanged_non_scaling_optimizer_groups": unchanged_optimizer_groups,
        "diffuse_hash_before": d_hash_before,
        "diffuse_hash_after": d_hash_after,
        "reflection_hash_before": r_hash_before,
        "reflection_hash_after": r_hash_after,
        "failures": failures,
    }
    _atomic_json(Path(target), report)
    del static_inputs, pre_payload, post_payload, pre, post
    if failures:
        raise RuntimeError("T-scale migration parity failed: " + ", ".join(failures))
    return report


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
        static_inputs = static_cache.load(
            stem, training_only=not opt.stage_d_semantic_repair_pilot
        )
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
    gradient_probe = None
    if opt.stage_d_semantic_repair_pilot:
        stem = stems[0]
        camera = by_stem[stem]
        static_inputs = static_cache.load(stem)
        package = render_from_static_dr(
            state, background, static_inputs, return_ray_aux=False,
        )
        probe = anti_veil_loss(
            package["inside_color"], package["inside_alpha"],
            camera.original_image.cuda().permute(1, 2, 0),
            camera.specular_mask.permute(1, 2, 0) * package["transmittance_valid"],
            gt_luminance_threshold=opt.anti_veil_gt_luminance_threshold,
            high_alpha_threshold=opt.anti_veil_high_alpha_threshold,
            black_luminance_threshold=opt.anti_veil_black_luminance_threshold,
            saturation_alpha_threshold=opt.anti_veil_saturation_alpha_threshold,
            target_saturation_coverage=opt.anti_veil_target_saturation_coverage,
            gate_temperature=opt.anti_veil_gate_temperature,
            black_temperature=opt.anti_veil_black_temperature,
            coverage_temperature=opt.anti_veil_coverage_temperature,
            epsilon=opt.anti_veil_epsilon,
        )
        gradient_probe = {}
        for loss_name in ("black", "saturation"):
            gradients = torch.autograd.grad(
                probe[loss_name],
                (transmittance._opacity, transmittance._color),
                retain_graph=True, allow_unused=True,
            )
            gradient_probe[loss_name] = {
                "loss": float(probe[loss_name]),
                "t_opacity_gradient_nonzero": bool(
                    gradients[0] is not None and torch.any(gradients[0] != 0)
                ),
                "t_color_gradient_nonzero": bool(
                    gradients[1] is not None and torch.any(gradients[1] != 0)
                ),
                "diffuse_gradient_leak": any(
                    value.grad is not None for value in (
                        diffuse._xyz, diffuse._opacity, diffuse._base_color,
                    )
                ),
                "reflection_gradient_leak": any(
                    value.grad is not None for value in (
                        reflection._xyz, reflection._opacity, reflection._color,
                    )
                ),
            }
        required_gradient_contract = (
            gradient_probe["black"]["t_opacity_gradient_nonzero"]
            and gradient_probe["black"]["t_color_gradient_nonzero"]
            and gradient_probe["saturation"]["t_opacity_gradient_nonzero"]
            and not any(
                row["diffuse_gradient_leak"] or row["reflection_gradient_leak"]
                for row in gradient_probe.values()
            )
        )
        if not required_gradient_contract:
            failures.append("anti_veil_gradient_contract")
        _zero_stage_d_gradients(diffuse, reflection, transmittance)
        del static_inputs, package, probe
    report = {
        "schema": "rtgs_stage_d_cache_parity_v1",
        "fixed_camera_stems": list(CACHED_STEMS),
        "deterministic_random_camera_stem": random_stem,
        "tolerances": {
            "max_absolute": float(opt.stage_d_cache_parity_atol),
            "mean_absolute": float(opt.stage_d_cache_parity_mean_atol),
        },
        "rows": rows, "failures": failures,
        "anti_veil_gradient_probe": gradient_probe,
        "status": "PASS" if not failures else (
            "TSCALE_RECOVERY_PREFLIGHT_BLOCKED"
            if _tscale_recovery_mode(opt) else (
            "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED"
            if opt.stage_d_ownership_pilot else (
            "SEMANTIC_REPAIR_PILOT_BLOCKED"
            if opt.stage_d_semantic_repair_pilot else "CACHED_T_WARMUP_BLOCKED")
            )
        ),
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
    dataset._stage_d_start_checkpoint_path = str(Path(start_checkpoint).resolve())
    dataset._stage_d_start_checkpoint_sha256 = sha256_file(start_checkpoint)
    release = GeometryRelease(Path(dataset.geometry_release_manifest))
    semantic_cuboid = None
    if _requires_cuboid_space(opt):
        semantic_cuboid = CuboidSpace.from_metadata(
            release.root / "mesh_metadata.json",
            interface_margin=opt.transparent_interface_margin,
            epsilon=1e-6, device="cuda", dtype=torch.float32,
        )
        dataset._semantic_cuboid_space_metadata = semantic_cuboid.metadata()
    else:
        dataset._semantic_cuboid_space_metadata = None
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
    saved_config = None
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
            dataset.transmittance_init_seed,
            transmittance_cuboid_space=semantic_cuboid,
            transmittance_init_mode=dataset.transmittance_init_mode,
            map_location="cuda",
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
    _validate_semantic_repair_contract(
        dataset, opt, release, source, fresh_from_stage_b,
        saving_iterations, checkpoint_iterations,
    )
    _validate_ownership_contract(
        dataset, opt, release, source, fresh_from_stage_b,
        saving_iterations, checkpoint_iterations,
    )
    _validate_ownership_t_long_contract(
        dataset, opt, release, source, fresh_from_stage_b, saved_config,
        global_iteration, reflection_iteration, transmittance_iteration,
        saving_iterations, checkpoint_iterations,
    )
    _validate_tscale_recovery_contract(
        dataset, opt, release, source, fresh_from_stage_b, saved_config,
        global_iteration, reflection_iteration, transmittance_iteration,
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
    elif opt.stage_d_ownership_pilot:
        metadata = {
            "schema": "rtgs_stage_d_cuboid_path_ownership_arm_v4",
            "source": source, "config": config,
            "arm": opt.stage_d_ownership_arm,
            "actual_transmittance_initialization": dict(transmittance.initialization),
            "required_nodes_global": list(OWNERSHIP_NODES),
            "required_nodes_local": [0, 100, 250, 500],
            "review_stems": list(CACHED_STEMS),
            "pilot_global": [15001, OWNERSHIP_ENDPOINT],
            "mode": "cuboid_front_frozen_dr_ownership_isolation",
            "diffuse_optimizer_updates": 0, "reflection_optimizer_updates": 0,
            "transmittance_optimizer_updates": 500,
            "t_topology_updates_allowed": False, "expected_t_count": 4096,
            "cuboid_space": semantic_cuboid.metadata(),
            "transparent_path_mode": dataset.transparent_path_mode,
            "transparent_direct_mode": dataset.transparent_direct_mode,
            "transparent_reflection_mode": dataset.transparent_reflection_mode,
            "cout_ownership_mode": dataset.cout_ownership_mode,
            "future_depth_activation_global": 40000,
            "depth_enabled_during_run": False,
            "bird_roi_status": "OPERATOR_EVALUATION_ONLY",
        }
        _atomic_json(Path(scene.model_path, "ownership_arm_metadata.json"), metadata)
    elif opt.stage_d_ownership_t_long:
        metadata = {
            "schema": "rtgs_stage_d_ownership_t_long_run_v1",
            "source": source, "config": config,
            "start_checkpoint": str(Path(start_checkpoint).resolve()),
            "start_checkpoint_sha256": dataset._stage_d_start_checkpoint_sha256,
            "required_nodes_global": list(OWNERSHIP_T_LONG_NODES),
            "pilot_source": (
                "output/stage_d_tihubird_c03r8_cuboid_path_ownership_ab500_v4/"
                "ownership_ab_cpu_audit.json"
            ),
            "global": [15501, OWNERSHIP_T_LONG_ENDPOINT],
            "transmittance_local": [501, 5000],
            "mode": "cuboid_front_frozen_dr_ownership_t_long",
            "diffuse_optimizer_updates": 0, "reflection_optimizer_updates": 0,
            "transmittance_optimizer_updates": 4500,
            "t_topology_updates_allowed": False, "expected_t_count": 4096,
            "cuboid_space": semantic_cuboid.metadata(),
            "future_depth_activation_global": 40000,
            "depth_enabled_during_run": False,
            "bird_roi_status": "NO_INDEPENDENT_BIRD_ROI",
            "semantic_claim": False,
        }
        _atomic_json(Path(scene.model_path, "ownership_t_long_metadata.json"), metadata)
    elif _tscale_recovery_mode(opt):
        preflight = bool(opt.stage_d_tscale_recovery_preflight)
        metadata = {
            "schema": "rtgs_stage_d_tscale_recovery_run_v1",
            "profile": "preflight50" if preflight else "long",
            "source": source, "config": config,
            "start_checkpoint": str(Path(start_checkpoint).resolve()),
            "start_checkpoint_sha256": dataset._stage_d_start_checkpoint_sha256,
            "required_nodes_global": list(
                TSCALE_RECOVERY_PREFLIGHT_NODES
                if preflight else TSCALE_RECOVERY_LONG_NODES
            ),
            "global": [16001, 16050] if preflight else [16051, 20000],
            "transmittance_local": [1001, 1050] if preflight else [1051, 5000],
            "mode": "cuboid_front_frozen_dr_projected_tscale_recovery",
            "diffuse_optimizer_updates": 0,
            "reflection_optimizer_updates": 0,
            "transmittance_optimizer_updates": 50 if preflight else 3950,
            "t_topology_updates_allowed": False,
            "expected_t_count": 4096,
            "cuboid_space": semantic_cuboid.metadata(),
            "scale_repair": {
                "schema": "cuboid_support_projected_cap_v2",
                "raw_equals_active_after_every_optimizer_update": True,
                "affected_optimizer_state": ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
                "unaffected_optimizer_groups_preserved": [
                    "xyz", "color", "opacity", "rotation",
                ],
            },
            "future_depth_activation_global": 40000,
            "depth_enabled_during_run": False,
            "semantic_claim": False,
        }
        _atomic_json(Path(scene.model_path, "tscale_recovery_metadata.json"), metadata)
    elif opt.stage_d_semantic_repair_pilot:
        metadata = {
            "schema": "rtgs_stage_d_semantic_repair_pilot_v3",
            "source": source, "config": config,
            "actual_transmittance_initialization": dict(transmittance.initialization),
            "required_nodes": list(SEMANTIC_NODES), "review_stems": list(CACHED_STEMS),
            "pilot_global": [15001, SEMANTIC_ENDPOINT],
            "mode": "frozen_dr_cached_t_only_semantic_falsification",
            "diffuse_optimizer_updates": 0, "reflection_optimizer_updates": 0,
            "transmittance_optimizer_updates": 1000,
            "t_topology_updates_allowed": False,
            "expected_t_count": 4096,
            "cuboid_space": semantic_cuboid.metadata(),
            "anti_veil": anti_veil_config(opt),
            "future_depth_activation_global": 40000,
            "depth_enabled_during_run": False,
            "bird_roi_available": False,
        }
        _atomic_json(Path(scene.model_path, "semantic_repair_run_metadata.json"), metadata)
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
        cuboid_space=semantic_cuboid,
        semantic_repair=bool(opt.stage_d_semantic_repair_pilot),
        transparent_path_mode=dataset.transparent_path_mode,
        transparent_direct_mode=dataset.transparent_direct_mode,
        transparent_reflection_mode=dataset.transparent_reflection_mode,
        cout_ownership_mode=dataset.cout_ownership_mode,
        support_sigma=3.0,
        transfer_depth_margin=opt.transfer_depth_margin,
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
    if _cached_mode(opt):
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
            cache_schema=(
                OWNERSHIP_CACHE_SCHEMA
                if (
                    opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
                    or _tscale_recovery_mode(opt)
                )
                else None
            )
            or "rtgs_stage_d_static_dr_cache_v1",
        )
        cache_directory = (
            Path(dataset.stage_d_static_cache_path)
            if (
                opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
                or _tscale_recovery_mode(opt)
            )
            else Path(scene.model_path) / "static_dr_cache"
        )
        if (
            (
                opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
                or _tscale_recovery_mode(opt)
            )
            and opt.stage_d_reuse_static_cache
        ):
            static_cache = StaticDRCache(cache_directory, cache_identity, camera_identities)
        else:
            static_cache = _build_static_dr_cache(
                cache_directory, cameras, state, pipe, background,
                cache_identity, camera_identities,
                semantic_repair=(
                    opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
                ),
            )
        if opt.stage_d_ownership_pilot and opt.stage_d_ownership_arm == "transferred_d_inside":
            selected, selection_metadata = _select_transferred_d_candidates(
                static_cache, diffuse, semantic_cuboid, opt, count=4096,
            )
            transmittance.create_transferred_from_diffuse(
                diffuse, selected, semantic_cuboid, 4096,
                dataset.transmittance_init_seed,
                opacity_scale=opt.transfer_opacity_scale,
                opacity_min=opt.transfer_opacity_min,
                opacity_max=opt.transfer_opacity_max,
                selection_metadata=selection_metadata,
            )
            transmittance.initialization["branch"] = "transmittance"
            transmittance.training_setup(opt)
            state.mark_transmittance_updated()
            metadata["actual_transmittance_initialization"] = dict(
                transmittance.initialization
            )
        config["cached_t_warmup"].update({
            "cache_identity_sha256": cache_identity["identity_sha256"],
            "cache_aggregate_sha256": static_cache.aggregate_sha256,
            "mask_manifest_sha256": mask_manifest["manifest_file_sha256"],
            "renderer_config_sha256": cache_identity["renderer_config_sha256"],
            "phase_a_frozen_hash_before": frozen_hash_before,
            "cache_path": str(cache_directory.resolve()),
            "cache_reused": bool(
                (
                    opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
                    or _tscale_recovery_mode(opt)
                )
                and opt.stage_d_reuse_static_cache
            ),
        })
        if opt.stage_d_tscale_recovery_preflight:
            migration_report = _run_tscale_migration_parity(
                Path(scene.model_path) / "tscale_migration_parity.json", cameras,
                static_cache, state, pipe, background, opt, perceptual,
                diffuse, reflection, transmittance,
            )
            metadata["scale_migration"] = migration_report
        elif opt.stage_d_tscale_recovery_long:
            source_scale = _tscale_state_metrics(transmittance)
            if source_scale["parameterization"] \
                    != "cuboid_support_projected_cap_v2" \
                    or source_scale["capped_count"] \
                    or source_scale["raw_active_max_abs"] \
                    > TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
                raise RuntimeError(
                    "TSCALE_RECOVERY_LONG_BLOCKED: source raw/active scale mismatch"
                )
            metadata["scale_source_validation"] = source_scale
        _run_cache_parity(
            Path(scene.model_path) / "cache_parity_report.json", cameras,
            static_cache, state, pipe, background, opt, perceptual,
            diffuse, reflection, transmittance,
        )
        if not (
            opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
            or _tscale_recovery_mode(opt)
        ):
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
            verdict = (
                "TSCALE_RECOVERY_PREFLIGHT_BLOCKED"
                if _tscale_recovery_mode(opt) else (
                "OWNERSHIP_T_LONG_BLOCKED"
                if opt.stage_d_ownership_t_long else (
                "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED"
                if opt.stage_d_ownership_pilot else (
                "SEMANTIC_REPAIR_PILOT_BLOCKED"
                if opt.stage_d_semantic_repair_pilot else "CACHED_T_WARMUP_BLOCKED")))
            )
            raise RuntimeError(f"{verdict}: D/R changed during cache preflight")
        metadata["cache_identity"] = cache_identity
        metadata["cache_aggregate_sha256"] = static_cache.aggregate_sha256
        metadata["phase_a_frozen_hash_before"] = frozen_hash_before
        metadata_name = (
            "tscale_recovery_metadata.json" if _tscale_recovery_mode(opt) else (
            "ownership_t_long_metadata.json" if opt.stage_d_ownership_t_long else (
            "ownership_arm_metadata.json" if opt.stage_d_ownership_pilot else (
            "semantic_repair_run_metadata.json"
            if opt.stage_d_semantic_repair_pilot else "cached_twarmup_run_metadata.json")))
        )
        _atomic_json(Path(scene.model_path, metadata_name), metadata)
        restore_rng_state(restored_rng_state)
    viewpoints, camera_indices = restore_camera_deck(cameras, runtime_state)
    if opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long \
            or _tscale_recovery_mode(opt):
        initial_iteration = (
            global_iteration if _tscale_recovery_mode(opt)
            else (15500 if opt.stage_d_ownership_t_long else 15000)
        )
        initial_checkpoint = make_stage_d_checkpoint(
            diffuse, reflection, transmittance, initial_iteration,
            reflection_iteration, transmittance_iteration, source, config,
            release.manifest["geometry_release_id"],
            release.validation["aggregate_sha256"],
            make_camera_runtime_state(camera_indices, len(cameras)),
        )
        initial_checkpoint_path = os.path.join(
            scene.model_path, f"chkpnt{initial_iteration}.pth"
        )
        torch.save(initial_checkpoint, initial_checkpoint_path)
        if _tscale_recovery_mode(opt):
            initial_hash_audit = _write_tscale_checkpoint_audit(
                scene.model_path, initial_checkpoint_path,
                initial_checkpoint, frozen_hash_before,
            )
        del initial_checkpoint
        scene.save(initial_iteration)
        _render_formal_review_node(
            scene, state, pipe, background, release, initial_iteration,
            stems=CACHED_STEMS, static_cache=static_cache,
        )
        if _tscale_recovery_mode(opt):
            node_telemetry = {
                "schema": "rtgs_stage_d_tscale_recovery_node_telemetry_v1",
                "global_iteration": int(initial_iteration),
                "reflection_local_iteration": int(reflection_iteration),
                "transmittance_local_iteration": int(transmittance_iteration),
                "optimizer_updates_this_step": {
                    "diffuse": 0, "reflection": 0, "transmittance": 0,
                },
                "counts": {
                    "diffuse": int(diffuse.get_xyz.shape[0]),
                    "reflection": int(reflection.get_xyz.shape[0]),
                    "transmittance": int(transmittance.get_xyz.shape[0]),
                },
                "t_scale": _tscale_state_metrics(transmittance),
                "checkpoint_audit": initial_hash_audit,
                "formal_review_node": True,
            }
            node_directory = Path(scene.model_path) / "review_node_telemetry"
            node_directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                node_directory / f"iteration_{initial_iteration:06d}.json",
                node_telemetry,
            )
    progress = tqdm(range(global_iteration, opt.iterations), desc="Stage D training progress")
    telemetry_path = Path(
        opt.stage_d_telemetry_jsonl
        or os.path.join(scene.model_path, "stage_d_telemetry.jsonl")
    )
    if telemetry_path.exists() and fresh_from_stage_b:
        raise FileExistsError(f"refusing to append a fresh Stage D run to {telemetry_path}")
    last_record = None
    ownership_long_guard = []
    tscale_recovery_guard = []
    for iteration in range(global_iteration + 1, opt.iterations + 1):
        ownership_long_failures = []
        tscale_recovery_failures = []
        scale_projection = None
        phase = (
            _phase_for_iteration(
                iteration, opt.stage_d_semantic_repair_pilot,
                opt.stage_d_ownership_pilot,
                opt.stage_d_ownership_t_long,
                _tscale_recovery_mode(opt),
            )
            if _cached_mode(opt) else "exact_joint"
        )
        phase_a = phase in (
            "cached_t_warmup", "semantic_repair_cached_t_only",
            "cuboid_path_ownership_t_only", "cuboid_path_ownership_t_long",
            "cuboid_path_ownership_tscale_recovery",
        )
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
            static_cache.load(
                Path(str(camera.image_name)).stem,
                training_only=True,
            )
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
        anti_black = payload["anti_veil_black"]
        anti_saturation = payload["anti_veil_saturation"]
        anti_ramp = payload["anti_veil_ramp"]
        anti_soft_saturation = payload["anti_veil_soft_saturation_coverage"]
        anti_soft_high = payload["anti_veil_soft_high_alpha_coverage"]

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
            transmittance.optimizer.step()
            if _tscale_recovery_mode(opt):
                scale_projection = transmittance.project_raw_scaling_to_active_()
            transmittance.optimizer.zero_grad(set_to_none=True)
            if phase_a:
                state.mark_transmittance_updated()
            else:
                state.mark_parameters_updated()
            topology_called = False
            if _transmittance_topology_update_allowed(opt, transmittance_iteration):
                topology_called = True
                transmittance.densify_and_prune(
                    opt.transmittance_densify_grad_threshold,
                    opt.transmittance_min_opacity, scene.cameras_extent,
                    allow_unhit_prune=(
                        transmittance_iteration >= opt.transmittance_prune_unhit_after
                    ),
                )
            if (
                opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
                or opt.stage_d_ownership_t_long or _tscale_recovery_mode(opt)
            ):
                if int(transmittance.get_xyz.shape[0]) != 4096:
                    raise RuntimeError(
                        "Stage D fixed-topology pilot: T count changed from 4096"
                    )
                transmittance.assert_strictly_inside()
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
            if (
                (opt.stage_d_semantic_repair_pilot and iteration == SEMANTIC_ENDPOINT)
                or (opt.stage_d_ownership_pilot and iteration == OWNERSHIP_ENDPOINT)
                or (opt.stage_d_ownership_t_long and iteration == OWNERSHIP_T_LONG_ENDPOINT)
                or (
                    opt.stage_d_tscale_recovery_preflight
                    and iteration == TSCALE_RECOVERY_PREFLIGHT_ENDPOINT
                )
                or (
                    opt.stage_d_tscale_recovery_long
                    and iteration == TSCALE_RECOVERY_LONG_ENDPOINT
                )
            ):
                frozen_hash_after = {
                    "diffuse": frozen_branch_hash(diffuse),
                    "reflection": frozen_branch_hash(reflection),
                }
                if frozen_hash_after != frozen_hash_before:
                    raise RuntimeError(
                        "Stage D frozen-D/R pilot: D/R frozen state changed"
                    )
                config["cached_t_warmup"]["phase_a_frozen_hash_after"] = frozen_hash_after
                metadata["phase_a_frozen_hash_after"] = frozen_hash_after
                _atomic_json(
                    Path(
                        scene.model_path,
                        "ownership_t_long_metadata.json"
                        if opt.stage_d_ownership_t_long else (
                        "tscale_recovery_metadata.json"
                        if _tscale_recovery_mode(opt) else (
                        "ownership_arm_metadata.json" if opt.stage_d_ownership_pilot
                        else "semantic_repair_run_metadata.json")),
                    ), metadata
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
                "schema": _telemetry_schema(opt),
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
                    "l_anti_veil_black": float(anti_black),
                    "l_anti_veil_saturation": float(anti_saturation),
                    "anti_veil_ramp": float(anti_ramp),
                    "anti_veil_soft_saturation_coverage": float(anti_soft_saturation),
                    "anti_veil_soft_high_alpha_coverage": float(anti_soft_high),
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
            if (
                opt.stage_d_semantic_repair_pilot or opt.stage_d_ownership_pilot
                or opt.stage_d_ownership_t_long or _tscale_recovery_mode(opt)
            ):
                last_record["semantic_metrics"] = _semantic_step_metrics(
                    package, gt, camera.specular_mask
                )
            if opt.stage_d_ownership_t_long:
                metrics = last_record["semantic_metrics"]
                cap = metrics.get("t_support_scale_cap", {})
                ownership_long_guard.append({
                    "saturation": metrics["ain_saturation_fraction_ge_0_95"],
                    "black": metrics["high_ain_near_black_conditional_fraction"],
                    "capped_fraction": float(cap.get("capped_count", 0)) / 4096.0,
                    "minimum_scale_factor": float(cap.get("minimum_factor", 1.0)),
                })
                ownership_long_guard = ownership_long_guard[-OWNERSHIP_T_LONG_GUARD_WINDOW:]
                if (
                    len(ownership_long_guard) == OWNERSHIP_T_LONG_GUARD_WINDOW
                    and transmittance_iteration >= 1000
                ):
                    guard_summary, failures = _ownership_long_guard_result(
                        ownership_long_guard
                    )
                    last_record["ownership_long_guard"] = guard_summary
                    if failures:
                        ownership_long_failures = failures
                        last_record["ownership_long_guard"]["failures"] = failures
            if _tscale_recovery_mode(opt):
                metrics = last_record["semantic_metrics"]
                cap = metrics.get("t_support_scale_cap", {})
                energy = metrics["contribution_energy"]
                last_record["t_scale_projection"] = scale_projection
                tscale_recovery_guard.append({
                    "saturation": metrics["ain_saturation_fraction_ge_0_95"],
                    "black": metrics["high_ain_near_black_conditional_fraction"],
                    "capped_fraction": float(cap.get("capped_count", 0)) / 4096.0,
                    "minimum_forward_factor": float(cap.get("minimum_factor", 1.0)),
                    "projection_affected_fraction": float(
                        scale_projection["affected_fraction"]
                    ),
                    "minimum_pre_projection_factor": float(
                        scale_projection["minimum_factor_before"]
                    ),
                    "post_projection_raw_active_abs": float(
                        scale_projection["raw_active_max_abs_after"]
                    ),
                    "t_energy": float(energy["transmittance_contribution"]),
                    "cin_energy": float(metrics["cin_energy"]),
                })
                tscale_recovery_guard = tscale_recovery_guard[-OWNERSHIP_T_LONG_GUARD_WINDOW:]
                guard_summary, failures = _tscale_recovery_guard_result(
                    tscale_recovery_guard
                )
                last_record["tscale_recovery_guard"] = guard_summary
                if failures:
                    tscale_recovery_failures = failures
                    last_record["tscale_recovery_guard"]["failures"] = failures

            checkpoint_due = iteration in checkpoint_iterations or iteration == opt.iterations
            if checkpoint_due:
                checkpoint = make_stage_d_checkpoint(
                    diffuse, reflection, transmittance, iteration,
                    reflection_iteration, transmittance_iteration, source, config,
                    release.manifest["geometry_release_id"],
                    release.validation["aggregate_sha256"],
                    make_camera_runtime_state(camera_indices, len(cameras)),
                )
                checkpoint_path = os.path.join(
                    scene.model_path, f"chkpnt{iteration}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                if _tscale_recovery_mode(opt):
                    last_record["checkpoint_audit"] = _write_tscale_checkpoint_audit(
                        scene.model_path, checkpoint_path, checkpoint,
                        frozen_hash_before,
                    )
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
            elif opt.stage_d_semantic_repair_pilot and iteration in SEMANTIC_NODES:
                _render_formal_review_node(
                    scene, state, pipe, background, release, iteration,
                    stems=CACHED_STEMS, static_cache=static_cache,
                )
                last_record["formal_review_node"] = True
            elif opt.stage_d_ownership_pilot and iteration in OWNERSHIP_NODES:
                _render_formal_review_node(
                    scene, state, pipe, background, release, iteration,
                    stems=CACHED_STEMS, static_cache=static_cache,
                )
                last_record["formal_review_node"] = True
            elif opt.stage_d_ownership_t_long and iteration in OWNERSHIP_T_LONG_NODES:
                _render_formal_review_node(
                    scene, state, pipe, background, release, iteration,
                    stems=CACHED_STEMS, static_cache=static_cache,
                )
                last_record["formal_review_node"] = True
            elif _tscale_recovery_mode(opt) and iteration in _required_nodes(opt):
                _render_formal_review_node(
                    scene, state, pipe, background, release, iteration,
                    stems=CACHED_STEMS, static_cache=static_cache,
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
        del anti_black, anti_saturation, anti_soft_saturation, anti_soft_high
        del reflection_aux, trans_aux, valid, depth_order
        device_free_before_release, _ = torch.cuda.mem_get_info()
        cache_released = _allocator_cache_under_pressure(device_free_before_release)
        if cache_released:
            torch.cuda.empty_cache()
        last_record["allocator_cache_released"] = bool(cache_released)
        last_record["device_free_before_cache_release_bytes"] = int(device_free_before_release)
        last_record["whole_step_wall_ms"] = float((time.perf_counter() - wall_start) * 1000.0)
        if _tscale_recovery_mode(opt) and iteration in _required_nodes(opt):
            node_directory = Path(scene.model_path) / "review_node_telemetry"
            node_directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                node_directory / f"iteration_{iteration:06d}.json", last_record,
            )
        _write_jsonl(telemetry_path, last_record)
        if ownership_long_failures:
            raise RuntimeError(
                "OWNERSHIP_T_LONG_BLOCKED: " + ", ".join(ownership_long_failures)
            )
        if tscale_recovery_failures:
            _write_tscale_abort_hashes(
                scene.model_path, iteration, diffuse, reflection, transmittance,
                tscale_recovery_failures,
            )
            verdict = (
                "TSCALE_RECOVERY_PREFLIGHT_BLOCKED"
                if opt.stage_d_tscale_recovery_preflight
                else "TSCALE_RECOVERY_LONG_BLOCKED"
            )
            raise RuntimeError(verdict + ": " + ", ".join(tscale_recovery_failures))

    if opt.stage_d_formal_onset or _cached_mode(opt):
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
            "STAGE_D_TSCALE_RECOVERY_PREFLIGHT_COMPLETED"
            if opt.stage_d_tscale_recovery_preflight else (
            "STAGE_D_TSCALE_RECOVERY_LONG_COMPLETED"
            if opt.stage_d_tscale_recovery_long else (
            "STAGE_D_OWNERSHIP_T_LONG_COMPLETED"
            if opt.stage_d_ownership_t_long else (
            "STAGE_D_CUBOID_PATH_OWNERSHIP_ARM_COMPLETED"
            if opt.stage_d_ownership_pilot else (
            "STAGE_D_SEMANTIC_REPAIR_PILOT_COMPLETED"
            if opt.stage_d_semantic_repair_pilot else (
                "STAGE_D_CACHED_TWARMUP_AND_JOINT_COMPLETED"
            if opt.stage_d_cached_twarmup else (
                "STAGE_D_FORMAL_ONSET_COMPLETED" if opt.stage_d_formal_onset
                else "STAGE_D_SMOKE_COMPLETED"
            ))))))
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
        "first_bounce": (
            "T from frozen cuboid front position + epsilon*frozen_camera_direction"
            if (
                opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
                or _tscale_recovery_mode(opt)
            )
            else "T from D position + epsilon*d_cam"
        ),
        "second_bounce": "D from frozen back_position + epsilon*d_cam",
        "phase_a_frozen_hash_before": frozen_hash_before,
        "phase_a_frozen_hash_after": frozen_hash_after,
        "phase_b_static_dr_cache_enabled": False if opt.stage_d_cached_twarmup else None,
        "semantic_repair_static_dr_cache_enabled": (
            True if opt.stage_d_semantic_repair_pilot else None
        ),
        "semantic_repair_t_topology_updates": (
            0 if opt.stage_d_semantic_repair_pilot else None
        ),
        "ownership_arm": opt.stage_d_ownership_arm if opt.stage_d_ownership_pilot else None,
        "ownership_t_topology_updates": (
            0 if (
                opt.stage_d_ownership_pilot or opt.stage_d_ownership_t_long
                or _tscale_recovery_mode(opt)
            ) else None
        ),
        "ownership_t_long": bool(opt.stage_d_ownership_t_long),
        "ownership_t_long_updates": 4500 if opt.stage_d_ownership_t_long else None,
        "tscale_recovery": bool(_tscale_recovery_mode(opt)),
        "tscale_recovery_updates": (
            50 if opt.stage_d_tscale_recovery_preflight else (
                3950 if opt.stage_d_tscale_recovery_long else None
            )
        ),
        "final_t_scale": (
            _tscale_state_metrics(transmittance)
            if _tscale_recovery_mode(opt) else None
        ),
    }
    summary_name = (
        "stage_d_tscale_recovery_summary.json" if _tscale_recovery_mode(opt) else (
        "stage_d_ownership_t_long_summary.json" if opt.stage_d_ownership_t_long else (
        "stage_d_ownership_arm_summary.json" if opt.stage_d_ownership_pilot else (
        "stage_d_semantic_repair_summary.json"
        if opt.stage_d_semantic_repair_pilot else (
        "stage_d_cached_twarmup_summary.json" if opt.stage_d_cached_twarmup else (
            "stage_d_formal_summary.json" if opt.stage_d_formal_onset
            else "stage_d_smoke_summary.json"
        )))))
    )
    with open(os.path.join(scene.model_path, summary_name), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    progress.close()
    print(summary["status"] + " " + json.dumps(summary["last_telemetry"], sort_keys=True))
