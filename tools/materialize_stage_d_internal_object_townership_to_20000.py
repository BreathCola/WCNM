#!/usr/bin/env python3
"""Materialize D-016 15,500->20,000 post-hoc review artifacts.

This tool is intentionally post-run only.  It loads the real 15,500 source
checkpoint and the real continuation checkpoints, renders fixed-nine review
products, and writes a derived review directory with immutable provenance.  It
does not replay initialization, run an optimizer, call backward, resume
training, or save new training checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from geometry.cuboid_space import CuboidSpace
from geometry.geometry_release import GeometryRelease, sha256_file
from gaussian_renderer.transmittance_renderer import StageDRenderState
from gaussian_renderer.transmittance_renderer import render_from_static_dr
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_d_scene import StageDScene
from scene.stage_d_state import STAGE_D_FORMAT, restore_stage_d_checkpoint
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from stage_d_training import (
    FORMAL_RELEASE_ID,
    FORMAL_RELEASE_SHA256,
    FORMAL_STEMS,
    INTERNAL_OBJECT_TO_20000_ENDPOINT,
    INTERNAL_OBJECT_TO_20000_NODES,
    INTERNAL_OBJECT_TO_20000_OUTPUT_NAME,
    INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
    _release_camera_identities,
)
from utils.transmittance_debug import (
    make_semantic_repair_contact_sheet,
    save_transmittance_debug_maps,
)
from utils.general_utils import safe_state
from utils.internal_object_mask import (
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    REVIEWED_ROLE,
    object_domain_metrics,
    object_occupancy_domains,
    validate_internal_object_mask_set,
)
from utils.specular_mask import validate_specular_mask_set
from utils.stage_d_static_cache import StaticDRCache, state_sha256


POSTHOC_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_posthoc_review_v1"
PLAN_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_operator_plan_v1"
METADATA_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_v1"
OUTPUT = ROOT / "output" / INTERNAL_OBJECT_TO_20000_OUTPUT_NAME
PILOT_OUTPUT = ROOT / "output/stage_d_tihubird_c03r8_internal_object_townership_pilot_v1"
SOURCE = PILOT_OUTPUT / "chkpnt15500.pth"
RELEASE = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
INTERNAL_MASK = ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3/manifest.json"
GLASS_MASK = "specular_masks_reviewed_v1/manifest.json"
REVIEW_NODES = (15500,) + tuple(INTERNAL_OBJECT_TO_20000_NODES)
REQUIRED_CHECKPOINT_NODES = tuple(INTERNAL_OBJECT_TO_20000_NODES)
FLOAT_METRICS_NAME = "float_metrics.json"
IMMUTABLE_RELATIVE_FILES = (
    "stage_d_telemetry.jsonl",
    "internal_object_townership_to_20000_metadata.json",
) + tuple(f"chkpnt{node}.pth" for node in REQUIRED_CHECKPOINT_NODES)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_hash(root: Path, *, exclude_posthoc: bool = True) -> dict:
    digest = hashlib.sha256()
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if exclude_posthoc and rel.startswith("posthoc_review/"):
            continue
        file_hash = sha256_file(path)
        entries.append({"relative_path": rel, "sha256": file_hash, "size_bytes": path.stat().st_size})
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(entries), "sha256": digest.hexdigest(), "entries": entries}


def _immutable_hashes(output: Path, operator_plan: Path | None) -> dict[str, str]:
    files = [output / rel for rel in IMMUTABLE_RELATIVE_FILES]
    files += [SOURCE, RELEASE, INTERNAL_MASK, output / "cache_parity_report.json"]
    files += sorted((output / "point_cloud").glob("*/iteration_*/point_cloud.ply"))
    if operator_plan is not None:
        files.append(operator_plan)
    result = {}
    for path in files:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha256_file(path)
    return result


def _extract_train_args(plan: dict) -> tuple[object, object, object]:
    command = list(plan["command"])
    train_index = next(i for i, item in enumerate(command) if str(item).endswith("train.py"))
    argv = command[train_index + 1:]
    parser = ArgumentParser(description="D-016 to-20000 posthoc train-arg parser")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--diffuse_init_checkpoint", type=str, default=None)
    args = parser.parse_args(argv)
    return lp.extract(args), op.extract(args), pp.extract(args)


def _checkpoint_path(output: Path, node: int) -> Path:
    if node == 15500:
        return SOURCE
    return output / f"chkpnt{node}.pth"


def validate_preconditions(output: Path, posthoc: Path) -> dict:
    if output.name != INTERNAL_OBJECT_TO_20000_OUTPUT_NAME:
        raise ValueError("output path is not the D-016 to-20000 continuation output")
    if not output.is_dir():
        raise FileNotFoundError(output)
    if posthoc.exists():
        raise FileExistsError(f"refusing existing posthoc output: {posthoc}")
    if sha256_file(SOURCE) != INTERNAL_OBJECT_TO_20000_SOURCE_SHA256:
        raise ValueError("15,500 source checkpoint SHA-256 mismatch")
    metadata = _json(output / "internal_object_townership_to_20000_metadata.json")
    if metadata.get("schema") != METADATA_SCHEMA:
        raise ValueError("to-20000 metadata schema mismatch")
    if metadata.get("global") != [15501, INTERNAL_OBJECT_TO_20000_ENDPOINT]:
        raise ValueError("to-20000 metadata global range mismatch")
    if metadata.get("transmittance_local") != [501, 5000]:
        raise ValueError("to-20000 metadata T-local range mismatch")
    if metadata.get("diffuse_optimizer_updates") != 0 or metadata.get("reflection_optimizer_updates") != 0:
        raise ValueError("to-20000 metadata claims D/R updates")
    if metadata.get("transmittance_optimizer_updates") != 4500:
        raise ValueError("to-20000 metadata T update count mismatch")
    if metadata.get("expected_t_count") != 4096:
        raise ValueError("to-20000 metadata T count mismatch")
    if metadata.get("t_reinitialization") is not False:
        raise ValueError("to-20000 metadata must record no T reinitialization")
    if metadata.get("transferred_d_selection_rerun") is not False:
        raise ValueError("to-20000 metadata must record no transferred-D selection rerun")
    if metadata.get("random_fill_rerun") is not False:
        raise ValueError("to-20000 metadata must record no random fill rerun")
    for node in REQUIRED_CHECKPOINT_NODES:
        if not (output / f"chkpnt{node}.pth").is_file():
            raise FileNotFoundError(output / f"chkpnt{node}.pth")
    return metadata


def validate_plan(plan: dict, output: Path) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("operator plan schema mismatch")
    if plan.get("output") != str(output):
        raise ValueError("operator plan output path mismatch")
    if plan.get("source_sha256") != INTERNAL_OBJECT_TO_20000_SOURCE_SHA256:
        raise ValueError("operator plan source hash mismatch")
    if plan.get("release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise ValueError("operator plan Stage-C aggregate mismatch")
    if plan.get("internal_object_role") != REVIEWED_ROLE:
        raise ValueError("operator plan internal-object role mismatch")
    if plan.get("internal_object_human_status") != "accepted":
        raise ValueError("operator plan internal-object status mismatch")
    if plan.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        raise ValueError("operator plan internal-object semantic version mismatch")
    if plan.get("iterations", {}).get("end_inclusive") != 20000:
        raise ValueError("operator plan endpoint mismatch")
    if list(plan.get("nodes", [])) != list(INTERNAL_OBJECT_TO_20000_NODES):
        raise ValueError("operator plan nodes mismatch")


def _build_scene(dataset, opt, pipe, model_path: Path, metadata: dict):
    dataset.model_path = str(model_path)
    dataset._stage_d_start_checkpoint_path = str(SOURCE.resolve())
    dataset._stage_d_start_checkpoint_sha256 = sha256_file(SOURCE)
    release = GeometryRelease(Path(dataset.geometry_release_manifest))
    if release.manifest.get("geometry_release_id") != FORMAL_RELEASE_ID:
        raise ValueError("Stage-C release ID mismatch")
    if release.validation.get("aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise ValueError("Stage-C release aggregate mismatch")
    semantic_cuboid = CuboidSpace.from_metadata(
        release.root / "mesh_metadata.json",
        interface_margin=opt.transparent_interface_margin,
        epsilon=1e-6,
        device="cuda",
        dtype=torch.float32,
    )
    dataset._semantic_cuboid_space_metadata = semantic_cuboid.metadata()
    dataset._validated_specular_mask_manifest = validate_specular_mask_set(
        dataset.source_path, dataset.images, dataset.specular_masks or GLASS_MASK,
    )
    dataset._validated_internal_object_mask_manifest = validate_internal_object_mask_set(
        dataset.source_path, dataset.images, dataset.internal_object_masks,
    )
    diffuse = DiffuseSurfelModel(dataset.roughness_min, opt.optimizer_type)
    reflection = ReflectionSurfelModel()
    transmittance = TransmittanceSurfelModel()
    scene = StageDScene(dataset, diffuse, reflection, transmittance, shuffle=False)
    state = StageDRenderState(
        diffuse=diffuse,
        reflection=reflection,
        transmittance=transmittance,
        geometry_release=release,
        scene_radius=scene.cameras_extent,
        ray_chunk_size=512,
        ray_cutoff_sigma=dataset.ray_cutoff_sigma,
        ray_hit_threshold=dataset.ray_hit_threshold,
        ray_epsilon_scale=dataset.ray_epsilon_scale,
        material_alpha_threshold=dataset.material_alpha_threshold,
        roughness_min=diffuse.roughness_min,
        roughness_remap=dataset.roughness_remap,
        ray_checkpoint_chunks=False,
        cuboid_space=semantic_cuboid,
        semantic_repair=True,
        transparent_path_mode=dataset.transparent_path_mode,
        transparent_direct_mode=dataset.transparent_direct_mode,
        transparent_reflection_mode=dataset.transparent_reflection_mode,
        cout_ownership_mode=dataset.cout_ownership_mode,
        support_sigma=3.0,
        transfer_depth_margin=opt.transfer_depth_margin,
    )
    cache_identity = metadata["cache_identity"]
    static_cache = StaticDRCache(
        Path(metadata["config"]["cached_t_warmup"]["cache_path"]),
        cache_identity,
        _release_camera_identities(release, scene.getTrainCameras()),
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    return scene, state, background, release, static_cache


def _load_checkpoint_to_scene(scene, release, opt, checkpoint_path: Path, node: int) -> dict:
    before = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    if checkpoint.get("format") != STAGE_D_FORMAT:
        raise ValueError(f"checkpoint {node} is not rtgs_stage_d")
    expected = (node, 12000, node - 15000)
    actual = (
        checkpoint.get("global_iteration"),
        checkpoint.get("reflection_iteration"),
        checkpoint.get("transmittance_iteration"),
    )
    if actual != expected:
        raise ValueError(f"checkpoint {node} iteration tuple {actual} != {expected}")
    restore_stage_d_checkpoint(
        checkpoint,
        scene.diffuse,
        scene.reflection,
        scene.transmittance,
        opt,
        opt,
        opt,
        release.manifest["geometry_release_id"],
        release.validation["aggregate_sha256"],
        restore_rng=False,
    )
    after = sha256_file(checkpoint_path)
    if before != after:
        raise RuntimeError(f"checkpoint {node} changed while materializing")
    return checkpoint


def _mask_entry(mask_manifest: dict, stem: str) -> dict:
    for entry in mask_manifest["entries"]:
        if entry["stem"] == stem:
            return entry
    raise KeyError(stem)


def _mask_tensor(path: Path, *, device: str = "cuda", size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.reshape(image.height, image.width, 1).to(device=device)
    return (data.float() / 255.0).clamp(0, 1)


def _overlay_mask(base_path: Path, mask_path: Path, out: Path, color: tuple[int, int, int]) -> None:
    base = Image.open(base_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(base.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", base.size, color)
    overlay = Image.blend(base, tint, 0.45)
    base.paste(overlay, mask=mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.save(out)


def _save_domain_visual(domains: dict[str, torch.Tensor], out: Path) -> None:
    mpos = domains["Mpos"].detach().cpu().squeeze(-1)
    mignore = domains["Mignore"].detach().cpu().squeeze(-1)
    mneg = domains["Mneg"].detach().cpu().squeeze(-1)
    rgb = torch.zeros((3, mpos.shape[0], mpos.shape[1]))
    rgb[1] = mpos
    rgb[2] = mignore
    rgb[0] = mneg
    save_image(rgb.clamp(0, 1), str(out))


def _write_masks_and_float_metrics(view: Path, stem: str, mask_manifest: dict, package: dict[str, torch.Tensor]) -> dict:
    entry = _mask_entry(mask_manifest, stem)
    root = INTERNAL_MASK.parent
    bird_path = root / entry["bird_mask_path"]
    base_path = root / entry["internal_base_mask_path"]
    union_path = root / entry["internal_object_union_mask_path"]
    glass_path = root / entry["glass_hard_mask_path"]
    final_path = view / "final.png"
    _overlay_mask(final_path, bird_path, view / "bird_mask_overlay.png", (255, 220, 0))
    _overlay_mask(final_path, base_path, view / "internal_base_mask_overlay.png", (0, 180, 255))
    _overlay_mask(final_path, union_path, view / "union_mask_overlay.png", (255, 80, 80))
    h, w = package["inside_alpha"].shape[:2]
    mask_size = (int(w), int(h))
    union = _mask_tensor(union_path, size=mask_size)
    glass = _mask_tensor(glass_path, size=mask_size)
    valid = package["two_hit_valid"].detach()
    ignore = torch.zeros_like(union)
    domains = object_occupancy_domains(union, glass, valid, ignore, erode_px=3, dilate_px=3)
    _save_domain_visual(domains, view / "mpos_mignore_mneg.png")
    masks = {
        "bird": _mask_tensor(bird_path, size=mask_size),
        "internal_base": _mask_tensor(base_path, size=mask_size),
        "union": union,
    }
    metrics = object_domain_metrics(package, domains, masks, alpha_floor=0.35, erode_px=3, dilate_px=3)
    metrics.update({
        "schema": "rtgs_stage_d_internal_object_townership_to_20000_float_metrics_v1",
        "stem": stem,
        "source": "float tensors rendered during materialization; masks are formal reviewed v3",
        "mask_alignment": {
            "mode": "explicit_nearest_resize_to_render_resolution",
            "render_height": int(h),
            "render_width": int(w),
        },
    })
    _atomic_json(view / FLOAT_METRICS_NAME, metrics)
    return metrics


def _make_overview(posthoc: Path, filename: str, source_png: str, nodes: tuple[int, ...], stems: tuple[str, ...]) -> str:
    thumb = (180, 100)
    label_height = 24
    canvas = Image.new("RGB", (thumb[0] * len(stems), (thumb[1] + label_height) * len(nodes)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, node in enumerate(nodes):
        for col, stem in enumerate(stems):
            path = posthoc / "debug" / f"iteration_{node:06d}" / stem / source_png
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = col * thumb[0] + (thumb[0] - image.width) // 2
            y0 = row * (thumb[1] + label_height)
            y = y0 + label_height + (thumb[1] - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((col * thumb[0] + 3, y0 + 4), f"{node} {stem}", fill="black")
    target = posthoc / filename
    canvas.save(target)
    return filename


@torch.no_grad()
def _render_review_node(
    scene,
    state,
    pipe,
    background,
    release,
    static_cache,
    node: int,
    mask_manifest: dict,
) -> dict:
    cameras = {
        Path(str(camera.image_name)).stem: camera
        for camera in (scene.getTestCameras() or scene.getTrainCameras())
    }
    missing = sorted(set(FORMAL_STEMS) - set(cameras))
    if missing:
        raise ValueError(f"formal review cameras are missing: {missing}")
    iteration_directory = Path(scene.model_path) / "debug" / f"iteration_{node:06d}"
    prior_checkpoint_mode = state.ray_checkpoint_chunks
    prior_chunk_size = state.ray_chunk_size
    state.ray_checkpoint_chunks = False
    state.ray_chunk_size = 512
    node_metrics = {}
    try:
        for stem in FORMAL_STEMS:
            camera = cameras[stem]
            static_inputs = static_cache.load(stem)
            debug = render_from_static_dr(
                state,
                background,
                static_inputs,
                return_ray_aux=False,
                return_ray_diagnostics=True,
            )
            directory = iteration_directory / stem
            save_transmittance_debug_maps(
                debug,
                camera.original_image.cuda(),
                str(directory),
                camera.specular_mask,
                camera.specular_mask_sha256,
                release.manifest["geometry_release_id"],
                release.validation["aggregate_sha256"],
            )
            node_metrics[stem] = _write_masks_and_float_metrics(directory, stem, mask_manifest, debug)
            del debug, static_inputs
        make_semantic_repair_contact_sheet(str(iteration_directory), FORMAL_STEMS)
    finally:
        state.ray_checkpoint_chunks = prior_checkpoint_mode
        state.ray_chunk_size = prior_chunk_size
    return node_metrics


def materialize(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    posthoc = output / "posthoc_review"
    metadata = validate_preconditions(output, posthoc)
    operator_plan = args.operator_plan.resolve() if args.operator_plan else None
    plan = _json(operator_plan) if operator_plan else _json(Path("/tmp/d016_to_20000_plan.json"))
    validate_plan(plan, output)
    mask_manifest = _json(INTERNAL_MASK)
    if mask_manifest.get("artifact_role") != REVIEWED_ROLE:
        raise ValueError("formal internal-object mask role mismatch")
    if mask_manifest.get("human_status") != "accepted":
        raise ValueError("formal internal-object human status mismatch")
    if mask_manifest.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        raise ValueError("formal internal-object semantic version mismatch")

    before_hashes = _immutable_hashes(output, operator_plan)
    raw_tree = _tree_hash(output, exclude_posthoc=True)
    tmp = output / f".posthoc_review.tmp.{os.getpid()}"
    if tmp.exists():
        raise FileExistsError(tmp)
    tmp.mkdir(parents=True)
    try:
        dataset, opt, pipe = _extract_train_args(plan)
        safe_state(True)
        scene, state, background, release, static_cache = _build_scene(
            dataset, opt, pipe, tmp, metadata,
        )
        checkpoint_hashes = {}
        checkpoint_branch_hashes = {}
        float_metric_index = {}
        for node in REVIEW_NODES:
            checkpoint_path = _checkpoint_path(output, node)
            checkpoint = _load_checkpoint_to_scene(scene, release, opt, checkpoint_path, node)
            checkpoint_hashes[str(node)] = sha256_file(checkpoint_path)
            checkpoint_branch_hashes[str(node)] = {
                "diffuse": state_sha256(checkpoint["diffuse"]),
                "reflection": state_sha256(checkpoint["reflection"]),
                "transmittance": state_sha256(checkpoint["transmittance"]),
            }
            node_metrics = _render_review_node(
                scene, state, pipe, background, release, static_cache, node, mask_manifest,
            )
            float_metric_index[str(node)] = node_metrics

        overviews = {
            "final": _make_overview(posthoc=tmp, filename="overview_final_15500_20000.png", source_png="final.png", nodes=REVIEW_NODES, stems=FORMAL_STEMS),
            "cin": _make_overview(posthoc=tmp, filename="overview_cin_15500_20000.png", source_png="inside_color.png", nodes=REVIEW_NODES, stems=FORMAL_STEMS),
            "ain": _make_overview(posthoc=tmp, filename="overview_ain_15500_20000.png", source_png="inside_alpha.png", nodes=REVIEW_NODES, stems=FORMAL_STEMS),
        }
        after_hashes = _immutable_hashes(output, operator_plan)
        if before_hashes != after_hashes:
            raise RuntimeError("immutable source/output/release/mask files changed during materialization")
        new_files = [
            str(Path("posthoc_review") / path.relative_to(tmp))
            for path in sorted(p for p in tmp.rglob("*") if p.is_file())
        ]
        manifest = {
            "schema": POSTHOC_SCHEMA,
            "output": str(output),
            "source_checkpoint": str(SOURCE.resolve()),
            "source_checkpoint_sha256": INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
            "node_source_contract": {
                "15500": "loaded from immutable pilot chkpnt15500.pth",
                "16000_20000": "loaded from continuation checkpoints only",
                "deterministic_fresh_replay": False,
                "t_reinitialization": False,
                "transferred_d_selection_rerun": False,
                "random_fill_rerun": False,
            },
            "review_nodes": list(REVIEW_NODES),
            "fixed_nine_stems": list(FORMAL_STEMS),
            "checkpoint_sha256": checkpoint_hashes,
            "checkpoint_branch_hashes": checkpoint_branch_hashes,
            "posthoc_review_artifacts": {
                "debug_root": "posthoc_review/debug",
                "overview_final": "posthoc_review/overview_final_15500_20000.png",
                "overview_cin": "posthoc_review/overview_cin_15500_20000.png",
                "overview_ain": "posthoc_review/overview_ain_15500_20000.png",
                "overviews": overviews,
                "per_view_float_metrics": "posthoc_review/debug/iteration_*/<stem>/float_metrics.json",
            },
            "float_metric_index": float_metric_index,
            "raw_output_tree_sha256_before_materialization": raw_tree["sha256"],
            "raw_output_tree_file_count_before_materialization": raw_tree["file_count"],
            "immutable_files_before_after": {
                path: {"before": before_hashes[path], "after": after_hashes[path]}
                for path in sorted(before_hashes)
            },
            "newly_added_derived_files": new_files,
            "no_optimizer_execution": True,
            "no_backward": True,
            "no_training_resume": True,
            "no_checkpoint_write": True,
            "no_ply_mutation": True,
        }
        _atomic_json(tmp / "materialization_manifest.json", manifest)
        os.replace(tmp, posthoc)
        return manifest
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--operator-plan", type=Path, default=Path("/tmp/d016_to_20000_plan.json"))
    args = parser.parse_args()
    manifest = materialize(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
