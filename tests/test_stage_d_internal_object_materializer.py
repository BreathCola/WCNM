import json
from argparse import Namespace

import pytest

import tools.materialize_stage_d_internal_object_review as materializer


FILTER_METADATA = {
    "schema": "rtgs_stage_d_internal_object_transfer_filter_v3",
    "pre_filter_D_indices_sha256": "pre",
    "selected_D_indices_sha256": "selected",
    "rejected_D_indices_sha256": "rejected",
    "random_fill_count": 2864,
    "pre_object_mask_candidate_count": 4096,
    "post_object_mask_candidate_count": 1232,
    "selected_transferred_count": 1232,
    "rejected_min_views_count": 10,
    "rejected_support_ratio_count": 20,
    "valid_projection_views_histogram": {"3": 1},
    "visible_domain_views_histogram": {"4": 2},
    "positive_object_views_histogram": {"5": 3},
    "per_surfel_support_summary": {"min": 0.6, "max": 1.0},
}


def _metadata(filter_meta=None):
    return {
        "actual_transmittance_initialization": {
            "selection": {
                "internal_object_filter": dict(filter_meta or FILTER_METADATA),
            }
        }
    }


def test_materializer_reads_exact_nested_filter_metadata_path():
    assert materializer._filter_metadata(_metadata()) == FILTER_METADATA


def test_materializer_rejects_legacy_filter_metadata_path():
    with pytest.raises(ValueError, match="actual_transmittance_initialization.selection"):
        materializer._filter_metadata({
            "actual_transmittance_initialization": {
                "internal_object_filter": dict(FILTER_METADATA),
            }
        })


def test_materializer_rejects_missing_filter_key():
    filter_meta = dict(FILTER_METADATA)
    filter_meta.pop("per_surfel_support_summary")

    with pytest.raises(ValueError, match="missing"):
        materializer._filter_metadata(_metadata(filter_meta))


def test_materializer_rejects_existing_posthoc_output_before_heavy_work(tmp_path):
    output = tmp_path / materializer.INTERNAL_OBJECT_OUTPUT_NAME
    output.mkdir()
    (output / "posthoc_review").mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"execute": True}), encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing existing posthoc output"):
        materializer.materialize(Namespace(output=output, operator_plan=plan))

