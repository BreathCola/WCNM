import numpy as np

from tools.audit_diffusion_renderer_normal_axes import (
    apply_axis_candidate,
    generate_axis_candidates,
    score_candidate,
)


def test_enumerates_all_48_signed_axis_permutations():
    candidates = generate_axis_candidates()
    assert len(candidates) == 48
    assert len({candidate.matrix for candidate in candidates}) == 48
    assert sum(candidate.right_handed for candidate in candidates) == 24
    assert {candidate.determinant for candidate in candidates} == {-1, 1}
    assert [candidate.candidate_id for candidate in candidates] == [f"C{index:02d}" for index in range(48)]


def test_candidate_mapping_matches_recorded_matrix_and_expression():
    candidate = generate_axis_candidates()[5]
    normal = np.array([[[0.2, -0.3, 0.9]]], dtype=np.float32)
    expected = normal @ np.asarray(candidate.matrix, dtype=np.float32).T
    np.testing.assert_allclose(apply_axis_candidate(normal, candidate), expected)
    assert candidate.expression.startswith("[x', y', z'] = [")


def test_synthetic_known_axis_transform_ranks_first():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(31, 43, 3)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
    known = generate_axis_candidates()[29]
    stable = apply_axis_candidate(raw, known)
    scores = {
        candidate.candidate_id: score_candidate([apply_axis_candidate(raw, candidate)], [stable])["rank_score"]
        for candidate in generate_axis_candidates()
    }
    assert max(scores, key=scores.get) == known.candidate_id
    assert scores[known.candidate_id] > 0.99999
