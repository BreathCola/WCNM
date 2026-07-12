#!/usr/bin/env python3
"""D-015 zero-update semantic-renderer repair ablation operator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer.reflection_renderer import StageBRenderState, render as render_stage_b
from gaussian_renderer.transmittance_renderer import (
    StageDRenderState, build_static_dr_inputs, render as render_stage_d,
)
from geometry.cuboid_space import CuboidSpace
from geometry.geometry_release import GeometryRelease, sha256_file, validate_geometry_release
from scene import Scene
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_d_state import (
    STAGE_D_FORMAT, initialize_stage_d_from_stage_b, load_stage_d_checkpoint,
)
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from stage_d_training import (
    FORMAL_RELEASE_SHA256, FORMAL_SOURCE_SHA256, FORMAL_STEMS,
)
from utils.semantic_renderer_ablation import (
    SCHEMA, apply_legacy_fallback, assert_outside_mask_bitwise_parity,
    black_pixel_attribution, mask_boundary_diagnostics, nested_tensor_hash,
    overbright_diagnostics, render_arm3_with_handoff, save_tensor_products,
    tensor_sha256, write_aggregate,
)


SOURCE_15000 = ROOT / "output/tier2_c03_r8_oneshot_v4_rstart_g03000_to_g15000/chkpnt15000.pth"
SOURCE_20000 = ROOT / "output/stage_d_tihubird_c03r8_cuboid_path_ownership_trecover16050_g20000_v3/chkpnt20000.pth"
MANIFEST = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
OUTPUT = ROOT / "output/stage_d_tihubird_c03r8_semantic_renderer_repair_zero_step_ablation_v1"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def tree_content_hash(directory: Path) -> dict:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rows.append({
                "path": path.relative_to(directory).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            })
    return {
        "root": str(directory.resolve()),
        "file_count": len(rows),
        "aggregate_sha256": tensor_sha256(
            torch.tensor(bytearray(json.dumps(rows, sort_keys=True).encode("utf-8")),
                         dtype=torch.uint8)
        ),
    }


def default_args(model_path: Path):
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)
    namespace = parser.parse_args([])
    dataset = model_params.extract(namespace)
    opt = opt_params.extract(namespace)
    pipe = pipe_params.extract(namespace)
    dataset.source_path = str(ROOT / "data/TiHuBird")
    dataset.model_path = str(model_path)
    dataset.images = "images"
    dataset.model_type = "surfel"
    dataset.stage = "stage_d"
    dataset.experiment = "TiHuBird D-015 semantic-renderer zero-step ablation"
    dataset.resolution = 8
    dataset.normal_priors = "diffrender_priors_candidates/axis_smoke/C03/normal"
    dataset.normal_prior_space = "camera"
    dataset.specular_masks = "specular_masks_reviewed_v1/manifest.json"
    dataset.geometry_release_manifest = str(MANIFEST)
    dataset.transmittance_init_mode = "random_strict_inside"
    dataset.transmittance_init_count = 4096
    dataset.transmittance_init_seed = 20260703
    dataset.transmittance_compose = "alpha_over"
    dataset.transparent_path_mode = "cuboid_front_v1"
    dataset.transparent_direct_mode = "off"
    dataset.transparent_reflection_mode = "off"
    dataset.cout_ownership_mode = "support_safe_outside"
    dataset.ray_background = "scene"
    dataset.ray_chunk_size = 2048
    opt.iterations = 15000
    opt.lambda_spec = 0.2
    opt.specular_k0 = 0.9
    opt.lambda_depth = 0.2
    opt.stage_d_depth_start_iteration = 40000
    opt.transparent_interface_margin = 0.05
    opt.transparent_interface_margin_mode = "exclude"
    return dataset, opt, pipe


def load_group(group: str, checkpoint: Path, group_output: Path):
    dataset, opt, pipe = default_args(group_output)
    diffuse = DiffuseSurfelModel(dataset.roughness_min, opt.optimizer_type)
    reflection = ReflectionSurfelModel()
    transmittance = TransmittanceSurfelModel()
    release = GeometryRelease(MANIFEST)
    cuboid = CuboidSpace.from_metadata(
        release.root / "mesh_metadata.json",
        interface_margin=opt.transparent_interface_margin,
        epsilon=1e-6, device="cuda", dtype=torch.float32,
    )
    dataset._semantic_cuboid_space_metadata = cuboid.metadata()
    scene = Scene(dataset, diffuse, shuffle=False, initialize_model=False, write_metadata=False)
    header = torch.load(checkpoint, map_location="cpu")
    if header.get("format") == "rtgs_stage_b":
        bbox_min, bbox_max = torch.tensor([-1, -1, -1], dtype=torch.float32, device="cuda"), \
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        global_it, refl_it, trans_it, source, _runtime = initialize_stage_d_from_stage_b(
            checkpoint, diffuse, reflection, transmittance, opt, opt, opt,
            bbox_min, bbox_max, dataset.transmittance_init_count,
            dataset.transmittance_init_seed,
            transmittance_cuboid_space=cuboid,
            transmittance_init_mode="random_strict_inside",
            map_location="cuda",
        )
    elif header.get("format") == STAGE_D_FORMAT:
        global_it, refl_it, trans_it, source, _config, _runtime = load_stage_d_checkpoint(
            checkpoint, diffuse, reflection, transmittance, opt, opt, opt,
            release.manifest["geometry_release_id"],
            release.validation["aggregate_sha256"], map_location="cuda",
        )
        transmittance.cuboid_space = cuboid
    else:
        raise ValueError(f"unsupported source checkpoint format for {group}: {header.get('format')}")
    return {
        "group": group, "dataset": dataset, "opt": opt, "pipe": pipe,
        "scene": scene, "release": release, "cuboid": cuboid,
        "diffuse": diffuse, "reflection": reflection, "transmittance": transmittance,
        "global_iteration": global_it, "reflection_iteration": refl_it,
        "transmittance_iteration": trans_it, "source": source,
    }


def make_state(context, reflection_mode: str):
    dataset = context["dataset"]
    return StageDRenderState(
        diffuse=context["diffuse"],
        reflection=context["reflection"],
        transmittance=context["transmittance"],
        geometry_release=context["release"],
        scene_radius=context["scene"].cameras_extent,
        ray_chunk_size=dataset.ray_chunk_size,
        ray_cutoff_sigma=dataset.ray_cutoff_sigma,
        ray_hit_threshold=dataset.ray_hit_threshold,
        ray_epsilon_scale=dataset.ray_epsilon_scale,
        material_alpha_threshold=dataset.material_alpha_threshold,
        roughness_min=context["diffuse"].roughness_min,
        roughness_remap=dataset.roughness_remap,
        ray_checkpoint_chunks=False,
        cuboid_space=context["cuboid"],
        semantic_repair=False,
        transparent_path_mode="cuboid_front_v1",
        transparent_direct_mode="off",
        transparent_reflection_mode=reflection_mode,
        cout_ownership_mode="support_safe_outside",
        support_sigma=3.0,
    )


def with_final_linear(package: dict, background: torch.Tensor) -> dict:
    result = dict(package)
    bg = background.reshape(1, 1, 3).to(package["final"])
    t = package.get("transmittance_contribution", torch.zeros_like(package["final"]))
    result["final_linear"] = (
        package["diffuse_contribution"] + package["reflection_contribution"]
        + t + (1.0 - package["alpha"]) * bg
    )
    result["final"] = result["final_linear"].clamp(0.0, 1.0)
    result["render"] = result["final"].permute(2, 0, 1)
    return result


def render_arms(context, camera, background):
    pipe = context["pipe"]
    state0 = make_state(context, "off")
    state1 = make_state(context, "support_safe_outside")
    arm0 = with_final_linear(render_stage_d(camera, state0, pipe, background, return_ray_diagnostics=True), background)
    arm1 = with_final_linear(render_stage_d(camera, state1, pipe, background, return_ray_diagnostics=True), background)
    legacy_state = StageBRenderState(
        diffuse=context["diffuse"], reflection=context["reflection"],
        scene_radius=context["scene"].cameras_extent, ray_chunk_size=2048,
        ray_checkpoint_chunks=False,
    )
    legacy = with_final_linear(render_stage_b(camera, legacy_state, pipe, background), background)
    arm2 = with_final_linear(apply_legacy_fallback(arm0, legacy, camera.specular_mask), background)
    static_inputs = build_static_dr_inputs(
        camera, state0, pipe, background,
        return_ray_aux=False, return_ray_diagnostics=True,
    )
    arm3 = with_final_linear(render_arm3_with_handoff(state0, background, static_inputs), background)
    return {"arm_0": arm0, "arm_1": arm1, "arm_2": arm2, "arm_3": arm3}


def camera_by_stem(scene):
    cameras = {Path(str(camera.image_name)).stem: camera for camera in scene.getTrainCameras()}
    cameras.update({Path(str(camera.image_name)).stem: camera for camera in scene.getTestCameras()})
    missing = sorted(set(FORMAL_STEMS) - set(cameras))
    if missing:
        raise RuntimeError(f"fixed-nine cameras are missing: {missing}")
    return cameras


def make_group_contact_sheets(group_output: Path):
    columns = (
        ("final.png", "final"),
        ("reflection_contribution.png", "R"),
        ("inside_color.png", "Cin"),
        ("inside_alpha.png", "Ain"),
        ("outside_color.png", "Cout"),
        ("fallback_pixel_mask.png", "fallback"),
        ("near_black.png", "black"),
        ("validity_disagreement.png", "validity"),
        ("cout_handoff_difference.png", "handoff diff"),
    )
    thumb = (220, 124)
    label_h = 24
    for arm in ("arm_0", "arm_1", "arm_2", "arm_3"):
        canvas = Image.new(
            "RGB",
            (thumb[0] * len(columns), (thumb[1] + label_h) * len(FORMAL_STEMS)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for row, stem in enumerate(FORMAL_STEMS):
            view = group_output / arm / stem
            for col, (filename, label) in enumerate(columns):
                path = view / filename
                if not path.is_file():
                    image = Image.new("RGB", thumb, "black")
                else:
                    image = Image.open(path).convert("RGB")
                    image.thumbnail(thumb, Image.Resampling.LANCZOS)
                x0 = col * thumb[0]
                y0 = row * (thumb[1] + label_h)
                x = x0 + (thumb[0] - image.width) // 2
                y = y0 + label_h + (thumb[1] - image.height) // 2
                canvas.paste(image, (x, y))
                draw.text((x0 + 3, y0 + 4), f"{stem} | {label}", fill="black")
        target = group_output / f"{arm}_contact_sheet.png"
        canvas.save(target)


def run_group(group: str, checkpoint: Path, output: Path):
    group_output = output / group
    group_output.mkdir(parents=True)
    context = load_group(group, checkpoint, group_output)
    background = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    before_hash = {
        "diffuse": nested_tensor_hash(context["diffuse"].capture()),
        "reflection": nested_tensor_hash(context["reflection"].capture()),
        "transmittance": nested_tensor_hash(context["transmittance"].capture()),
    }
    cameras = camera_by_stem(context["scene"])
    rows = []
    t_snapshot_hashes = {}
    for stem in FORMAL_STEMS:
        camera = cameras[stem]
        arms = render_arms(context, camera, background)
        t_snapshot_hashes[stem] = before_hash["transmittance"]
        arm0 = arms["arm_0"]
        hard = camera.specular_mask
        for arm, package in arms.items():
            black, black_maps = black_pixel_attribution(package, hard)
            boundary, boundary_maps = mask_boundary_diagnostics(
                hard, package["two_hit_valid"], package["final"]
            )
            over = overbright_diagnostics(package, arm0 if arm != "arm_0" else None)
            parity = assert_outside_mask_bitwise_parity(arm0, package, hard)
            view_dir = group_output / arm / stem
            save_tensor_products(package, view_dir, maps={**black_maps, **boundary_maps})
            record = {
                "schema": SCHEMA,
                "group": group,
                "stem": stem,
                "arm": arm,
                "black_attribution": black,
                "mask_boundary": boundary,
                "overbright": over,
                "outside_mask_parity": parity,
                "fallback_validation": package.get("fallback_validation"),
                "handoff_metadata": package.get("handoff_metadata"),
                "tensor_hashes": {
                    name: tensor_sha256(value)
                    for name, value in package.items() if torch.is_tensor(value)
                },
            }
            atomic_json(view_dir / "stats.json", record)
            rows.append(record)
            del package
        del arms
    after_hash = {
        "diffuse": nested_tensor_hash(context["diffuse"].capture()),
        "reflection": nested_tensor_hash(context["reflection"].capture()),
        "transmittance": nested_tensor_hash(context["transmittance"].capture()),
    }
    if before_hash != after_hash:
        raise RuntimeError(f"{group} model state changed during zero-update ablation")
    make_group_contact_sheets(group_output)
    metadata = {
        "schema": SCHEMA,
        "group": group,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "global_iteration": context["global_iteration"],
        "reflection_iteration": context["reflection_iteration"],
        "transmittance_iteration": context["transmittance_iteration"],
        "model_hash_before": before_hash,
        "model_hash_after": after_hash,
        "zero_optimizer_update_proof": {
            "backward_called": False,
            "optimizer_step_called": False,
            "scheduler_step_called": False,
            "state_hash_unchanged": before_hash == after_hash,
        },
        "shared_t_snapshot_hashes": t_snapshot_hashes,
    }
    atomic_json(group_output / "group_metadata.json", metadata)
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required to run the real fixed-nine operator")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to run the real operator without --execute")
    if not torch.cuda.is_available():
        raise RuntimeError("D-015 fixed-nine renderer operator requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite D-015 output: {output}")
    output.mkdir(parents=True)
    record_path = output / "operator_record.json"
    release_before = validate_geometry_release(MANIFEST)
    if release_before["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        raise RuntimeError("Stage-C release aggregate mismatch")
    if sha256_file(SOURCE_15000) != FORMAL_SOURCE_SHA256:
        raise RuntimeError("Branch-A global-15000 source SHA mismatch")
    release_tree_before = tree_content_hash(ROOT / "output/stage_c_geometry_release_v1")
    record = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "output": str(output),
        "groups": {
            "fresh_15000": str(SOURCE_15000),
            "trained_20000": str(SOURCE_20000),
        },
        "release_aggregate_before": release_before["aggregate_sha256"],
        "release_tree_before": release_tree_before,
        "optimizer_updates": 0,
        "checkpoints_written": 0,
        "ply_written": 0,
    }
    atomic_json(record_path, record)
    rows = []
    metadata = {}
    try:
        for group, checkpoint in (
            ("trained_20000", SOURCE_20000),
            ("fresh_15000", SOURCE_15000),
        ):
            group_rows, group_metadata = run_group(group, checkpoint, output)
            rows.extend(group_rows)
            metadata[group] = group_metadata
        reports = write_aggregate(rows, output)
        release_after = validate_geometry_release(MANIFEST)
        release_tree_after = tree_content_hash(ROOT / "output/stage_c_geometry_release_v1")
        forbidden = [
            path.as_posix() for path in output.rglob("*")
            if path.name.startswith("chkpnt") or path.suffix.lower() == ".ply"
        ]
        if forbidden:
            raise RuntimeError(f"D-015 wrote forbidden resumable artifacts: {forbidden}")
        record.update({
            "status": "AWAITING_USER_REVIEW",
            "verdict": "AWAITING_USER_REVIEW",
            "group_metadata": metadata,
            "reports": reports,
            "release_aggregate_after": release_after["aggregate_sha256"],
            "release_tree_after": release_tree_after,
            "release_unchanged": release_tree_before == release_tree_after,
            "forbidden_resumable_artifacts": forbidden,
        })
    except Exception as exc:
        record.update({
            "status": "BLOCKED",
            "verdict": "BLOCKED",
            "operator_error": f"{type(exc).__name__}: {exc}",
        })
        atomic_json(record_path, record)
        raise
    atomic_json(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
