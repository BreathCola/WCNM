#!/usr/bin/env python3
"""Compare two completed Stage B pilots without renderer/CUDA access.

Only already-written PNGs and metadata are read.  Every paired visualization
uses one scale pooled across both runs.  If the physical 8-bit PNGs contain no
recoverable spatial signal, the affected comparison is reported unavailable
instead of being synthesized from summary statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _latest_debug(run: Path) -> Path:
    candidates = sorted((run / "debug").glob("iteration_*"))
    if not candidates:
        raise FileNotFoundError(f"no Stage B debug directory found under {run}")
    return candidates[-1]


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb(values: np.ndarray, path: Path) -> None:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded).save(path)


def _common_positive_scale(values: Iterable[np.ndarray]) -> float:
    flat = np.concatenate([value[np.isfinite(value)].reshape(-1) for value in values])
    if not flat.size:
        return 0.0
    scale = float(np.quantile(flat, 0.99))
    return scale if scale > 0.0 else float(flat.max(initial=0.0))


def _common_signed_scale(values: Iterable[np.ndarray]) -> float:
    return _common_positive_scale([np.abs(value) for value in values])


def _positive_display(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values / scale, 0.0, 1.0)


def _signed_display(values: np.ndarray, scale: float) -> np.ndarray:
    # Negative is blue, zero is mid-gray, positive is red.
    normalized = np.clip(values / scale, -1.0, 1.0)
    magnitude = np.mean(normalized, axis=-1)
    output = np.full((*magnitude.shape, 3), 0.5, dtype=np.float32)
    output[..., 0] += 0.5 * magnitude
    output[..., 2] -= 0.5 * magnitude
    return output


def _require_matching_shapes(named: Iterable[Tuple[str, np.ndarray]]) -> None:
    shapes = {name: value.shape for name, value in named}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"pilot debug image shapes do not match: {shapes}")


def compare_stage_b_pilots(run_a, run_b, output_dir) -> dict:
    run_a = Path(run_a).expanduser().resolve()
    run_b = Path(run_b).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for run in (run_a, run_b):
        if not run.is_dir():
            raise FileNotFoundError(run)
        if output == run or run in output.parents:
            raise ValueError("comparison output must not be inside either input run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"comparison output must be new or empty: {output}")

    debug_a, debug_b = _latest_debug(run_a), _latest_debug(run_b)
    before_a, before_b = _tree_hash(debug_a), _tree_hash(debug_b)
    gt_a, gt_b = _load_rgb(debug_a / "ground_truth.png"), _load_rgb(debug_b / "ground_truth.png")
    if not np.array_equal(gt_a, gt_b):
        raise ValueError("pilot fixed-view ground truths differ; refusing pixelwise comparison")
    ks_a, ks_b = _load_rgb(debug_a / "ks.png"), _load_rgb(debug_b / "ks.png")
    reflection_a = _load_rgb(debug_a / "reflection_contribution.png")
    reflection_b = _load_rgb(debug_b / "reflection_contribution.png")
    final_a, final_b = _load_rgb(debug_a / "final.png"), _load_rgb(debug_b / "final.png")
    _require_matching_shapes(
        (
            ("ground_truth", gt_a), ("ks_a", ks_a), ("ks_b", ks_b),
            ("reflection_a", reflection_a), ("reflection_b", reflection_b),
            ("final_a", final_a), ("final_b", final_b),
        )
    )
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "selected_debug_a": str(debug_a),
        "selected_debug_b": str(debug_b),
        "selected_debug_sha256_before": {"run_a": before_a, "run_b": before_b},
        "source_contract": {
            "renderer_invoked": False,
            "cuda_invoked": False,
            "physical_png_bit_depth": 8,
            "limitation": (
                "ks/reflection maps are existing 8-bit physical PNGs; values below one code value "
                "cannot be spatially recovered from reflection_metadata summary statistics"
            ),
        },
        "comparisons": {},
    }

    ks_scale = _common_positive_scale((ks_a, ks_b))
    if ks_scale <= 0.0:
        metadata["comparisons"]["ks"] = {"status": "unavailable", "reason": "both ks PNGs are zero"}
    else:
        _save_rgb(_positive_display(ks_a, ks_scale), output / "ks_run_a_common.png")
        _save_rgb(_positive_display(ks_b, ks_scale), output / "ks_run_b_common.png")
        metadata["comparisons"]["ks"] = {
            "status": "ok", "transform": "linear", "common_scale": ks_scale,
            "files": ["ks_run_a_common.png", "ks_run_b_common.png"],
        }

    delta_ks = ks_b - ks_a
    delta_ks_scale = _common_signed_scale((delta_ks,))
    if delta_ks_scale <= 0.0:
        metadata["comparisons"]["delta_ks"] = {
            "status": "unavailable", "reason": "quantized ks PNGs are identical"
        }
    else:
        _save_rgb(_signed_display(delta_ks, delta_ks_scale), output / "delta_ks_b_minus_a.png")
        metadata["comparisons"]["delta_ks"] = {
            "status": "ok", "transform": "signed_red_blue", "symmetric_scale": delta_ks_scale,
            "file": "delta_ks_b_minus_a.png",
        }

    reflection_scale = _common_positive_scale((reflection_a, reflection_b))
    if reflection_scale <= 0.0:
        metadata["comparisons"]["reflection_contribution"] = {
            "status": "unavailable",
            "reason": "both physical reflection_contribution PNGs contain no recoverable nonzero pixels",
        }
        metadata["comparisons"]["delta_reflection_contribution"] = {
            "status": "unavailable", "reason": "source physical PNGs contain no recoverable signal"
        }
    else:
        _save_rgb(
            _positive_display(reflection_a, reflection_scale),
            output / "reflection_contribution_run_a_common.png",
        )
        _save_rgb(
            _positive_display(reflection_b, reflection_scale),
            output / "reflection_contribution_run_b_common.png",
        )
        metadata["comparisons"]["reflection_contribution"] = {
            "status": "ok", "transform": "linear", "common_scale": reflection_scale,
            "files": [
                "reflection_contribution_run_a_common.png",
                "reflection_contribution_run_b_common.png",
            ],
        }
        delta_reflection = reflection_b - reflection_a
        delta_reflection_scale = _common_signed_scale((delta_reflection,))
        if delta_reflection_scale <= 0.0:
            metadata["comparisons"]["delta_reflection_contribution"] = {
                "status": "unavailable", "reason": "quantized physical PNGs are identical"
            }
        else:
            _save_rgb(
                _signed_display(delta_reflection, delta_reflection_scale),
                output / "delta_reflection_contribution_b_minus_a.png",
            )
            metadata["comparisons"]["delta_reflection_contribution"] = {
                "status": "ok", "transform": "signed_red_blue",
                "symmetric_scale": delta_reflection_scale,
                "file": "delta_reflection_contribution_b_minus_a.png",
            }

    residual_a, residual_b = np.abs(final_a - gt_a), np.abs(final_b - gt_b)
    residual_scale = _common_positive_scale((residual_a, residual_b))
    if residual_scale <= 0.0:
        metadata["comparisons"]["final_residual"] = {
            "status": "unavailable", "reason": "both final images exactly match ground truth"
        }
    else:
        _save_rgb(_positive_display(residual_a, residual_scale), output / "final_residual_run_a_common.png")
        _save_rgb(_positive_display(residual_b, residual_scale), output / "final_residual_run_b_common.png")
        metadata["comparisons"]["final_residual"] = {
            "status": "ok", "transform": "absolute_linear", "common_scale": residual_scale,
            "files": ["final_residual_run_a_common.png", "final_residual_run_b_common.png"],
        }
    delta_residual = residual_b - residual_a
    delta_residual_scale = _common_signed_scale((delta_residual,))
    if delta_residual_scale > 0.0:
        _save_rgb(
            _signed_display(delta_residual, delta_residual_scale),
            output / "delta_final_residual_b_minus_a.png",
        )
        metadata["comparisons"]["delta_final_residual"] = {
            "status": "ok", "transform": "signed_red_blue",
            "symmetric_scale": delta_residual_scale,
            "file": "delta_final_residual_b_minus_a.png",
        }
    else:
        metadata["comparisons"]["delta_final_residual"] = {
            "status": "unavailable", "reason": "quantized residual maps are identical"
        }

    after_a, after_b = _tree_hash(debug_a), _tree_hash(debug_b)
    if (before_a, before_b) != (after_a, after_b):
        raise RuntimeError("input debug directory changed during read-only comparison")
    metadata["selected_debug_sha256_after"] = {"run_a": after_a, "run_b": after_b}
    with (output / "comparison_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_a", required=True)
    parser.add_argument("--run_b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = compare_stage_b_pilots(args.run_a, args.run_b, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
