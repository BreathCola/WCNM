"""Fail-closed identity and storage helpers for Stage D Phase-A D/R caches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from geometry.geometry_release import sha256_file


CACHE_SCHEMA = "rtgs_stage_d_static_dr_cache_v1"
MANIFEST_SCHEMA = "rtgs_stage_d_static_dr_cache_manifest_v1"
OWNERSHIP_CACHE_SCHEMA = "rtgs_stage_d_cuboid_front_cache_v4"
OWNERSHIP_MANIFEST_SCHEMA = "rtgs_stage_d_cuboid_front_cache_manifest_v4"
FORBIDDEN_CACHE_KEYS = {"gt", "ground_truth", "target", "target_rgb", "original_image"}
TRAINING_PACKAGE_KEYS = {
    "alpha", "position", "normal", "surface_ks", "microfacet_F",
    "diffuse_contribution", "reflection_contribution",
    "semantic_r_stats", "semantic_r_filter_stats", "semantic_r_transparent_ray_count",
    "front_position", "front_normal", "frozen_camera_direction",
    "front_plane_residual", "front_face_index", "frozen_back_distance_residual",
    "cuboid_front_valid",
    "transparent_path_valid", "near_depth", "far_depth", "two_hit_valid",
    "transparent_mask_hard",
    "diffuse_unfiltered", "reflection_unfiltered",
    "transparent_path_mode", "transparent_direct_mode", "transparent_reflection_mode",
}


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_value(digest, value) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8")); digest.update(b"\0")
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for child in value:
            _hash_value(digest, child)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\0")
    else:
        raise TypeError(f"unsupported state-hash value: {type(value).__name__}")


def state_sha256(value) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def frozen_branch_hash(model) -> str:
    """Hash model, optimizer, topology/densification, and current scheduler LR state."""
    return state_sha256(model.capture())


def renderer_contract(dataset) -> dict:
    ownership = bool(getattr(dataset, "transparent_path_mode", "legacy_d_gbuffer") == "cuboid_front_v1")
    semantic = bool(getattr(dataset, "_semantic_cuboid_space_metadata", None))
    return {
        "schema": (
            "rtgs_stage_d_cuboid_front_renderer_contract_v4"
            if ownership else "rtgs_stage_d_renderer_contract_v1"
        ),
        "resolution": int(dataset.resolution),
        "ray_chunk_size": int(dataset.ray_chunk_size),
        "ray_cutoff_sigma": float(dataset.ray_cutoff_sigma),
        "ray_hit_threshold": float(dataset.ray_hit_threshold),
        "ray_epsilon_scale": float(dataset.ray_epsilon_scale),
        "material_alpha_threshold": float(dataset.material_alpha_threshold),
        "roughness_min": float(dataset.roughness_min),
        "roughness_remap": bool(dataset.roughness_remap),
        "ray_background": str(dataset.ray_background),
        "transmittance_compose": str(dataset.transmittance_compose),
        "reflection_domain": "all_valid_diffuse_surfaces",
        "transmittance_direction": "camera_incident_direction",
        "second_bounce_origin": "frozen_back_position_plus_epsilon_direction",
        "alpha_over": "Ct=Cin+(1-Ain)*Cout;At=Ain+(1-Ain)*Aout",
        "final_composition": "D_contribution+R_contribution+T_contribution+background",
        "bsdf_weight_mode": "brdf_times_cosine",
        "semantic_repair": semantic,
        "cuboid_space": getattr(dataset, "_semantic_cuboid_space_metadata", None),
        "r_transparent_spatial_filter": (
            getattr(dataset, "transparent_reflection_mode", "support_safe_outside")
            if ownership else ("outside_only" if semantic else None)
        ),
        "cout_spatial_filter": (
            getattr(dataset, "cout_ownership_mode", "support_safe_outside")
            if ownership else ("outside_only" if semantic else None)
        ),
        "transparent_path_mode": getattr(dataset, "transparent_path_mode", "legacy_d_gbuffer"),
        "transparent_direct_mode": getattr(dataset, "transparent_direct_mode", "legacy"),
        "transparent_reflection_mode": getattr(dataset, "transparent_reflection_mode", "legacy"),
        "cout_ownership_mode": getattr(dataset, "cout_ownership_mode", "legacy"),
        "support_classification": (
            "cuboid_local_finite_3sigma_v1" if ownership else None
        ),
    }


def make_identity(
    source_sha256, release_sha256, mask_manifest_sha256, renderer_config,
    cache_schema=CACHE_SCHEMA,
):
    identity = {
        "cache_schema_version": str(cache_schema),
        "source_checkpoint_sha256": str(source_sha256),
        "geometry_release_aggregate_sha256": str(release_sha256),
        "mask_manifest_sha256": str(mask_manifest_sha256),
        "renderer_config_sha256": canonical_sha256(renderer_config),
        "renderer_config": renderer_config,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _assert_no_targets(value, path="cache"):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_CACHE_KEYS:
                raise ValueError(f"static D/R cache must not contain target data: {path}.{key}")
            _assert_no_targets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_targets(child, f"{path}[{index}]")


def cpu_cache_payload(static_inputs: dict, identity: dict, stem: str, camera_identity: str):
    _assert_no_targets(static_inputs)

    def move(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            return tensor.float() if tensor.is_floating_point() else tensor
        if isinstance(value, dict):
            return {key: move(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(move(child) for child in value)
        return value

    return {
        "schema": identity["cache_schema_version"],
        "identity_sha256": identity["identity_sha256"],
        "camera_stem": str(stem),
        "camera_identity_sha256": str(camera_identity),
        "contains_target_rgb": False,
        "static_inputs": move(static_inputs),
    }


def write_manifest(directory: Path, identity: dict, entries: list[dict]) -> Path:
    directory = Path(directory)
    manifest = {
        "schema": (
            OWNERSHIP_MANIFEST_SCHEMA
            if identity.get("cache_schema_version") == OWNERSHIP_CACHE_SCHEMA
            else MANIFEST_SCHEMA
        ),
        "identity": identity,
        "entries": sorted(entries, key=lambda row: row["camera_stem"]),
    }
    manifest["aggregate_sha256"] = canonical_sha256([
        {key: row[key] for key in ("camera_stem", "camera_identity_sha256", "sha256")}
        for row in manifest["entries"]
    ])
    path = directory / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


class StaticDRCache:
    def __init__(self, directory: Path, expected_identity: dict, camera_identities: dict):
        self.directory = Path(directory)
        self.expected_identity = dict(expected_identity)
        self.camera_identities = dict(camera_identities)
        path = self.directory / "manifest.json"
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_manifest_schema = (
            OWNERSHIP_MANIFEST_SCHEMA
            if self.expected_identity.get("cache_schema_version") == OWNERSHIP_CACHE_SCHEMA
            else MANIFEST_SCHEMA
        )
        if self.manifest.get("schema") != expected_manifest_schema:
            raise ValueError("unsupported static D/R cache manifest schema")
        if self.manifest.get("identity") != self.expected_identity:
            raise ValueError("static D/R cache identity mismatch")
        rows = self.manifest.get("entries", [])
        self.entries = {row["camera_stem"]: row for row in rows}
        self._validated_stats = {}
        if len(self.entries) != len(rows) or set(self.entries) != set(self.camera_identities):
            raise ValueError("static D/R cache camera set mismatch")
        aggregate = canonical_sha256([
            {key: row[key] for key in ("camera_stem", "camera_identity_sha256", "sha256")}
            for row in sorted(rows, key=lambda item: item["camera_stem"])
        ])
        if aggregate != self.manifest.get("aggregate_sha256"):
            raise ValueError("static D/R cache aggregate mismatch")
        for stem, row in self.entries.items():
            if row.get("camera_identity_sha256") != self.camera_identities[stem]:
                raise ValueError(f"static D/R cache camera identity mismatch: {stem}")
            path = self.directory / row["relative_path"]
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                raise ValueError(f"static D/R cache file hash mismatch: {stem}")
            stat = path.stat()
            if int(row.get("size_bytes", stat.st_size)) != stat.st_size:
                raise ValueError(f"static D/R cache file size mismatch: {stem}")
            self._validated_stats[stem] = (stat.st_size, stat.st_mtime_ns)

    @property
    def aggregate_sha256(self):
        return self.manifest["aggregate_sha256"]

    def load(self, stem: str, device="cuda", training_only=False) -> dict:
        stem = str(stem)
        if stem not in self.entries:
            raise KeyError(f"camera is absent from static D/R cache: {stem}")
        row = self.entries[stem]
        path = self.directory / row["relative_path"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._validated_stats[stem]:
            raise ValueError(f"static D/R cache mutated after validation: {stem}")
        payload = torch.load(path, map_location="cpu")
        if (
            payload.get("schema") != self.expected_identity["cache_schema_version"]
            or payload.get("identity_sha256") != self.expected_identity["identity_sha256"]
            or payload.get("camera_stem") != stem
            or payload.get("camera_identity_sha256") != self.camera_identities[stem]
            or payload.get("contains_target_rgb") is not False
        ):
            raise ValueError(f"static D/R cache payload identity mismatch: {stem}")
        _assert_no_targets(payload)
        static_inputs = payload["static_inputs"]
        if training_only:
            static_inputs = dict(static_inputs)
            static_inputs.pop("semantic_cout_components", None)
            static_inputs.pop("transfer_evidence", None)
            package = static_inputs.get("package", {})
            static_inputs["package"] = {
                key: value for key, value in package.items()
                if key in TRAINING_PACKAGE_KEYS
            }

        def move(value):
            if torch.is_tensor(value):
                return value.to(device=device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(child) for child in value)
            return value

        return move(static_inputs)
