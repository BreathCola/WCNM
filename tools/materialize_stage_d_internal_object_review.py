#!/usr/bin/env python3
"""Materialize post-hoc D-016 review artifacts without training updates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from geometry.cuboid_space import CuboidSpace
from geometry.geometry_release import GeometryRelease, sha256_file
from scene.diffuse_surfel_model import DiffuseSurfelModel
from scene.reflection_surfel_model import ReflectionSurfelModel
from scene.stage_d_scene import StageDScene
from scene.stage_d_state import (
    STAGE_D_FORMAT,
    initialize_stage_d_from_stage_b,
    restore_stage_d_checkpoint,
)
from scene.transmittance_surfel_model import TransmittanceSurfelModel
from stage_d_training import (
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    FORMAL_STEMS,
    INTERNAL_OBJECT_NODES,
    INTERNAL_OBJECT_OUTPUT_NAME,
    _config,
    _filter_transferred_candidates_by_internal_object_masks,
    _mesh_bounds,
    _release_camera_identities,
    _render_formal_review_node,
    _select_transferred_d_candidates,
)
from gaussian_renderer.transmittance_renderer import StageDRenderState
from utils.general_utils import safe_state
from utils.internal_object_mask import validate_internal_object_mask_set
from utils.specular_mask import validate_specular_mask_set
from utils.stage_d_static_cache import StaticDRCache, state_sha256


POSTHOC_SCHEMA = "rtgs_stage_d_internal_object_posthoc_review_v1"
TRAINING_NODES = (15100, 15250, 15500)
IMMUTABLE_RELATIVE_FILES = (
    "stage_d_telemetry.jsonl",
    "internal_object_townership_metadata.json",
    "chkpnt15100.pth",
    "chkpnt15250.pth",
    "chkpnt15500.pth",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_hash(root: Path, *, exclude_posthoc: bool = True) -> dict:
    digest = __import__("hashlib").sha256()
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


def _immutable_hashes(output: Path, operator_plan: Path, source: Path, release: Path, mask: Path) -> dict:
    files = [output / rel for rel in IMMUTABLE_RELATIVE_FILES]
    files += sorted((output / "point_cloud").glob("*/iteration_*/point_cloud.ply"))
    files += [operator_plan, source, release, mask]
    result = {}
    for path in files:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha256_file(path)
    return result


def _extract_train_args(plan: dict) -> tuple[object, object, object, str]:
    command = list(plan["command"])
    train_index = next(i for i, item in enumerate(command) if str(item).endswith("train.py"))
    argv = command[train_index + 1:]
    parser = ArgumentParser(description="D-016 posthoc materializer train-arg parser")
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
    return lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint


def _filter_metadata(metadata: dict) -> dict:
    init = metadata.get("actual_transmittance_initialization", {})
    selection = init.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("metadata lacks actual_transmittance_initialization.selection")
    filter_meta = selection.get("internal_object_filter")
    if not isinstance(filter_meta, dict):
        raise ValueError("metadata lacks actual_transmittance_initialization.selection.internal_object_filter")
    if filter_meta.get("schema") != "rtgs_stage_d_internal_object_transfer_filter_v3":
        raise ValueError("internal-object transfer filter schema mismatch")
    required = {
        "pre_filter_D_indices_sha256",
        "selected_D_indices_sha256",
        "rejected_D_indices_sha256",
        "random_fill_count",
        "pre_object_mask_candidate_count",
        "post_object_mask_candidate_count",
        "selected_transferred_count",
        "rejected_min_views_count",
        "rejected_support_ratio_count",
        "valid_projection_views_histogram",
        "visible_domain_views_histogram",
        "positive_object_views_histogram",
        "per_surfel_support_summary",
    }
    missing = sorted(required - set(filter_meta))
    if missing:
        raise ValueError(f"internal-object filter metadata missing {missing}")
    return filter_meta


def _build_scene(dataset, opt, pipe, model_path: Path, start_checkpoint: str, metadata: dict):
    dataset.model_path = str(model_path)
    dataset._stage_d_start_checkpoint_path = str(Path(start_checkpoint).resolve())
    dataset._stage_d_start_checkpoint_sha256 = sha256_file(Path(start_checkpoint))
    release = GeometryRelease(Path(dataset.geometry_release_manifest))
    semantic_cuboid = CuboidSpace.from_metadata(
        release.root / "mesh_metadata.json",
        interface_margin=opt.transparent_interface_margin,
        epsilon=1e-6,
        device="cuda",
        dtype=torch.float32,
    )
    dataset._semantic_cuboid_space_metadata = semantic_cuboid.metadata()
    dataset._validated_specular_mask_manifest = validate_specular_mask_set(
        dataset.source_path, dataset.images, dataset.specular_masks,
    )
    dataset._validated_internal_object_mask_manifest = validate_internal_object_mask_set(
        dataset.source_path, dataset.images, dataset.internal_object_masks,
    )
    diffuse = DiffuseSurfelModel(dataset.roughness_min, opt.optimizer_type)
    reflection = ReflectionSurfelModel()
    transmittance = TransmittanceSurfelModel()
    scene = StageDScene(dataset, diffuse, reflection, transmittance, shuffle=True)
    bbox_min, bbox_max = _mesh_bounds(
        release.root / release.manifest["glass_mesh_relative_path"], "cuda"
    )
    global_iteration, reflection_iteration, transmittance_iteration, source, runtime_state = (
        initialize_stage_d_from_stage_b(
            start_checkpoint,
            diffuse,
            reflection,
            transmittance,
            opt,
            opt,
            opt,
            bbox_min,
            bbox_max,
            dataset.transmittance_init_count,
            dataset.transmittance_init_seed,
            transmittance_cuboid_space=semantic_cuboid,
            transmittance_init_mode=dataset.transmittance_init_mode,
            map_location="cuda",
        )
    )
    if global_iteration != 15000 or reflection_iteration != 12000 or transmittance_iteration != 0:
        raise ValueError("unexpected source iteration identity for D-016 replay")
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    state = StageDRenderState(
        diffuse=diffuse,
        reflection=reflection,
        transmittance=transmittance,
        geometry_release=release,
        scene_radius=scene.cameras_extent,
        ray_chunk_size=dataset.ray_chunk_size,
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
    static_cache = StaticDRCache(
        Path(metadata["config"]["cached_t_warmup"]["cache_path"]),
        metadata["cache_identity"],
        _release_camera_identities(release, scene.getTrainCameras()),
    )
    selected, selection_metadata = _select_transferred_d_candidates(
        static_cache, diffuse, semantic_cuboid, opt, count=4096,
    )
    selected, object_filter_metadata = _filter_transferred_candidates_by_internal_object_masks(
        selected,
        diffuse,
        scene.getTrainCameras(),
        static_cache,
        opt,
        dataset._validated_internal_object_mask_manifest,
        count=4096,
    )
    selection_metadata["internal_object_filter"] = object_filter_metadata
    runtime_filter = _filter_metadata(metadata)
    for key, expected in runtime_filter.items():
        if object_filter_metadata.get(key) != expected:
            raise ValueError(f"replay identity mismatch for internal_object_filter.{key}")
    transmittance.create_transferred_from_diffuse(
        diffuse,
        selected,
        semantic_cuboid,
        4096,
        dataset.transmittance_init_seed,
        opacity_scale=opt.transfer_opacity_scale,
        opacity_min=opt.transfer_opacity_min,
        opacity_max=opt.transfer_opacity_max,
        selection_metadata=selection_metadata,
    )
    transmittance.initialization["branch"] = "transmittance"
    transmittance.training_setup(opt)
    return scene, state, background, release, static_cache, source


def _render_node_from_checkpoint(dataset, opt, pipe, scene, state, background, release, static_cache, checkpoint_path: Path, node: int) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    if checkpoint.get("format") != STAGE_D_FORMAT:
        raise ValueError(f"checkpoint {node} is not rtgs_stage_d")
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
    _render_formal_review_node(scene, state, pipe, background, release, node, stems=FORMAL_STEMS, static_cache=static_cache)
    scene.save(node)


def materialize(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    posthoc = output / "posthoc_review"
    if output.name != INTERNAL_OBJECT_OUTPUT_NAME:
        raise ValueError("output path is not the D-016 pilot output")
    if not output.is_dir():
        raise FileNotFoundError(output)
    if posthoc.exists():
        raise FileExistsError(f"refusing existing posthoc output: {posthoc}")
    metadata = _json(output / "internal_object_townership_metadata.json")
    if metadata.get("schema") != "rtgs_stage_d_internal_object_townership_pilot_v1":
        raise ValueError("pilot metadata schema mismatch")
    runtime_filter = _filter_metadata(metadata)
    plan = _json(args.operator_plan.resolve())
    if plan.get("execute") is not True:
        raise ValueError("operator plan must be execute=true")
    source = Path(plan["source"]).resolve()
    if sha256_file(source) != FORMAL_SOURCE_SHA256:
        raise ValueError("source checkpoint hash mismatch")
    if plan.get("release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise ValueError("Stage-C release hash mismatch")

    for node in TRAINING_NODES:
        if not (output / f"chkpnt{node}.pth").is_file():
            raise FileNotFoundError(output / f"chkpnt{node}.pth")

    before_hashes = _immutable_hashes(
        output,
        args.operator_plan.resolve(),
        source,
        ROOT / "geometry_releases/stage_c_geometry_release_v1.json",
        ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3/manifest.json",
    )
    raw_tree = _tree_hash(output, exclude_posthoc=True)
    tmp = output / f".posthoc_review.tmp.{os.getpid()}"
    if tmp.exists():
        raise FileExistsError(tmp)
    tmp.mkdir(parents=True)
    try:
        dataset, opt, pipe, start_checkpoint = _extract_train_args(plan)
        safe_state(True)
        scene, state, background, release, static_cache, source_identity = _build_scene(
            dataset, opt, pipe, tmp, start_checkpoint, metadata,
        )
        first_hash = state_sha256(scene.transmittance.capture())
        first_filter = dict(scene.transmittance.initialization["selection"]["internal_object_filter"])

        tmp_check = tmp / "_second_replay_check"
        tmp_check.mkdir(parents=True)
        dataset2, opt2, pipe2, start_checkpoint2 = _extract_train_args(plan)
        scene2, state2, _background2, _release2, _cache2, _source2 = _build_scene(
            dataset2, opt2, pipe2, tmp_check, start_checkpoint2, metadata,
        )
        second_hash = state_sha256(scene2.transmittance.capture())
        shutil.rmtree(tmp_check)
        if first_hash != second_hash:
            raise ValueError("deterministic replay T state hash mismatch")
        if first_filter != runtime_filter:
            raise ValueError("deterministic replay filter metadata mismatch")

        scene.save(15000)
        _render_formal_review_node(scene, state, pipe, background, release, 15000, stems=FORMAL_STEMS, static_cache=static_cache)
        replay_root = tmp / "initial_state_replay"
        replay_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            replay_root / "initial_state_replay.json",
            {
                "schema": "rtgs_stage_d_internal_object_initial_replay_v1",
                "initial_state_kind": "deterministic_zero_update_replay",
                "source_checkpoint_sha256": FORMAL_SOURCE_SHA256,
                "stage_c_aggregate_sha256": FORMAL_RELEASE_SHA256,
                "formal_internal_object_aggregate_sha256": plan["internal_object_aggregate_sha256"],
                "init_mode": "transferred_d_inside",
                "seed": 20260703,
                "transmittance_count": 4096,
                "optimizer_updates": 0,
                "scheduler_advance": 0,
                "densification_or_pruning": False,
                "first_replay_transmittance_state_sha256": first_hash,
                "second_replay_transmittance_state_sha256": second_hash,
                "replay_deterministic": True,
                "internal_object_filter": first_filter,
            },
        )
        (replay_root / "transmittance_state_sha256.txt").write_text(first_hash + "\n", encoding="utf-8")

        for node in TRAINING_NODES:
            _render_node_from_checkpoint(
                dataset,
                opt,
                pipe,
                scene,
                state,
                background,
                release,
                static_cache,
                output / f"chkpnt{node}.pth",
                node,
            )

        after_hashes = _immutable_hashes(
            output,
            args.operator_plan.resolve(),
            source,
            ROOT / "geometry_releases/stage_c_geometry_release_v1.json",
            ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3/manifest.json",
        )
        if before_hashes != after_hashes:
            raise RuntimeError("immutable pilot/source/release/mask files changed during materialization")
        new_files = [
            str(Path("posthoc_review") / path.relative_to(tmp))
            for path in sorted(p for p in tmp.rglob("*") if p.is_file())
        ]
        manifest = {
            "schema": POSTHOC_SCHEMA,
            "pilot_output": str(output),
            "source_execution_head": "f93b26aaf27d678f96dabafe14b3c0c58adff0ae",
            "actual_training_checkpoints": [str(output / f"chkpnt{node}.pth") for node in TRAINING_NODES],
            "posthoc_replayed_initial_state": {
                "initial_state_kind": "deterministic_zero_update_replay",
                "optimizer_updates": 0,
                "scheduler_advance": 0,
                "densification_or_pruning": False,
                "transmittance_count": 4096,
                "first_replay_transmittance_state_sha256": first_hash,
                "second_replay_transmittance_state_sha256": second_hash,
                "replay_deterministic": True,
                "internal_object_filter": first_filter,
            },
            "posthoc_review_artifacts": {
                "debug_root": "posthoc_review/debug",
                "nodes": list(INTERNAL_OBJECT_NODES),
                "fixed_nine_stems": list(FORMAL_STEMS),
                "initial_state_replay": "posthoc_review/initial_state_replay/initial_state_replay.json",
            },
            "raw_pilot_output_tree_sha256_before_materialization": raw_tree["sha256"],
            "raw_pilot_output_tree_file_count_before_materialization": raw_tree["file_count"],
            "immutable_files_before_after": {
                path: {"before": before_hashes[path], "after": after_hashes[path]}
                for path in sorted(before_hashes)
            },
            "newly_added_derived_files": new_files,
            "no_optimizer_execution_during_materialization": True,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-plan", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
