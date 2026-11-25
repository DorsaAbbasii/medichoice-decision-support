
import numpy as np
import pytest

from src.ahp import calculate_ahp, validate_matrix

from src.matrices import (
    PAIRWISE_MATRICES,
    ALTERNATIVES,
)

from src.criteria_weights import (
    CRITERIA_MATRIX,
    CRITERIA_IDS,
)

from src.scenarios import (
    SCENARIO_MATRICES,
    SCENARIO_PROBABILITIES,
)

from src.group_decision import (
    GROUP_NAMES,
    GROUP_COMPETENCE_MATRIX,
    GROUP_CRITERIA_MATRICES,
)

from src.aggregation import (
    linear_aggregation,
    maximin_aggregation,
    bottleneck_criteria,
)

from src.imprecise_probability import (
    idm_bounds,
    contamination_bounds,
    robust_expected_utility,
)

from sensitivity_analysis import switching_points


def get_weights(matrix):
    return calculate_ahp(matrix)["weights"]


def get_priority_matrix():
    return np.column_stack([
        get_weights(PAIRWISE_MATRICES[c])
        for c in CRITERIA_IDS
    ])


# ==========================================
# TEST 1: Validate all 17 AHP matrices
# ==========================================

ALL_MATRICES = {
    **PAIRWISE_MATRICES,
    "Baseline": CRITERIA_MATRIX,
    **SCENARIO_MATRICES,
    "Group Competence": GROUP_COMPETENCE_MATRIX,
}

for scenario, groups in GROUP_CRITERIA_MATRICES.items():
    for group, matrix in groups.items():
        ALL_MATRICES[f"{scenario} - {group}"] = matrix


@pytest.mark.parametrize(
    "name,matrix",
    list(ALL_MATRICES.items()),
    ids=list(ALL_MATRICES.keys()),
)
def test_all_ahp_matrices(name, matrix):

    assert validate_matrix(matrix)

    result = calculate_ahp(matrix)

    assert np.all(result["weights"] > 0)

    assert np.isclose(
        result["weights"].sum(), 1.0
    )

    assert result["cr"] <= 0.20

    assert result["consistent"]


# ==========================================
# TEST 2: Perfectly consistent AHP matrix
# ==========================================

def test_perfect_consistency():

    matrix = np.array([
        [1,   2,   4],
        [0.5, 1,   2],
        [0.25, 0.5, 1],
    ])

    result = calculate_ahp(matrix)

    expected = np.array([4, 2, 1]) / 7

    np.testing.assert_allclose(
        result["weights"], expected
    )

    assert np.isclose(result["cr"], 0)


# ==========================================
# TEST 3: Reject invalid matrices
# ==========================================

@pytest.mark.parametrize("matrix", [
    [[1, 2], [2, 1]],
    [[1, 0], [0, 1]],
    [[2, 1], [1, 1]],
])
def test_invalid_matrices(matrix):

    with pytest.raises(ValueError):
        validate_matrix(matrix)


# ==========================================
# TEST 4: Baseline global ranking
# ==========================================

def test_baseline_ranking():

    V = get_priority_matrix()
    C = get_weights(CRITERIA_MATRIX)

    scores = linear_aggregation(V, C)

    expected = [
        0.208038,
        0.328939,
        0.251865,
        0.211158,
    ]

    np.testing.assert_allclose(
        scores,
        expected,
        atol=1e-6,
        rtol=0,
    )

    assert ALTERNATIVES[np.argmax(scores)] == "B"

    assert np.isclose(scores.sum(), 1)


# ==========================================
# TEST 5: Risk and uncertainty
# ==========================================

def test_risk_and_wald():

    V = get_priority_matrix()

    insured = V @ get_weights(
        SCENARIO_MATRICES["Insured Event"]
    )

    no_event = V @ get_weights(
        SCENARIO_MATRICES["No Insured Event"]
    )

    U = np.column_stack([insured, no_event])

    expected_utility = U @ SCENARIO_PROBABILITIES
    wald = U.min(axis=1)

    assert np.isclose(
        SCENARIO_PROBABILITIES.sum(), 1
    )

    assert ALTERNATIVES[
        np.argmax(expected_utility)
    ] == "D"

    assert ALTERNATIVES[
        np.argmax(wald)
    ] == "B"

    np.testing.assert_allclose(
        expected_utility,
        [0.252545, 0.243828, 0.236800, 0.266826],
        atol=1e-6,
        rtol=0,
    )


# ==========================================
# TEST 6: Group decision making
# ==========================================

def test_group_decision():

    V = get_priority_matrix()

    q = get_weights(GROUP_COMPETENCE_MATRIX)

    np.testing.assert_allclose(
        q, [1/3, 2/3]
    )

    scores = []

    for group in GROUP_NAMES:

        C = get_weights(
            GROUP_CRITERIA_MATRICES[
                "Baseline"
            ][group]
        )

        scores.append(V @ C)

    combined = np.column_stack(scores) @ q

    np.testing.assert_allclose(
        combined,
        [0.193826, 0.367610, 0.262394, 0.176171],
        atol=1e-6,
        rtol=0,
    )

    assert np.isclose(combined.sum(), 1)

    assert ALTERNATIVES[
        np.argmax(combined)
    ] == "B"


# ==========================================
# TEST 7: Nonlinear maximin aggregation
# ==========================================

def test_maximin():

    V = get_priority_matrix()
    C = get_weights(CRITERIA_MATRIX)

    linear = linear_aggregation(V, C)

    maximin = maximin_aggregation(V, C)

    expected = np.min(V * C, axis=1)

    np.testing.assert_allclose(
        maximin, expected
    )

    assert ALTERNATIVES[
        np.argmax(linear)
    ] == "B"

    assert ALTERNATIVES[
        np.argmax(maximin)
    ] == "C"

    assert np.all(maximin <= linear)

    bottlenecks = bottleneck_criteria(V, C)

    for i, criterion in enumerate(bottlenecks):

        assert np.isclose(
            maximin[i],
            V[i, criterion] * C[criterion],
        )


# ==========================================
# TEST 8: IDM and contamination equivalence
# ==========================================

def test_idm_contamination_equivalence():

    lower, upper, epsilon = idm_bounds(
        n=2,
        N=100,
        s=2,
    )

    assert np.isclose(lower, 2 / 102)
    assert np.isclose(upper, 4 / 102)
    assert np.isclose(epsilon, 2 / 102)

    np.testing.assert_allclose(
        [lower, upper],
        contamination_bounds(0.02, epsilon),
    )

    assert np.isclose(
        upper - lower, epsilon
    )


# ==========================================
# TEST 9: IDM uncertainty decreases with N
# ==========================================

def test_idm_convergence():

    _, _, epsilon_small = idm_bounds(
        n=2, N=100, s=2
    )

    _, _, epsilon_large = idm_bounds(
        n=20, N=1000, s=2
    )

    assert epsilon_large < epsilon_small


# ==========================================
# TEST 10: Robust expected utility
# ==========================================

def test_robust_utility():

    V = get_priority_matrix()

    insured = V @ get_weights(
        SCENARIO_MATRICES["Insured Event"]
    )

    no_event = V @ get_weights(
        SCENARIO_MATRICES["No Insured Event"]
    )

    lower, upper, _ = idm_bounds(
        2, 100, 2
    )

    low_u, high_u = robust_expected_utility(
        insured,
        no_event,
        lower,
        upper,
    )

    assert np.all(low_u <= high_u)

    nominal = (
        0.02 * insured
        + 0.98 * no_event
    )

    assert np.all(low_u <= nominal)
    assert np.all(nominal <= high_u)

    assert ALTERNATIVES[
        np.argmax(low_u)
    ] == "D"


# ==========================================
# TEST 11: Probability switching point
# ==========================================

def test_probability_switching_point():

    V = get_priority_matrix()

    insured = V @ get_weights(
        SCENARIO_MATRICES["Insured Event"]
    )

    no_event = V @ get_weights(
        SCENARIO_MATRICES["No Insured Event"]
    )

    points = switching_points(
        no_event,
        insured,
        ALTERNATIVES,
    )

    assert len(points) == 1

    point = points[0]

    assert {
        point["Company 1"],
        point["Company 2"],
    } == {"B", "D"}

    assert point["Switching Point"] == pytest.approx(
        0.102185,
        abs=1e-5,
    )


# ==========================================
# TEST 12: Premium weight sensitivity
# ==========================================

def test_premium_switching_point():

    V = get_priority_matrix()
    C = get_weights(CRITERIA_MATRIX)

    index = CRITERIA_IDS.index("C5")

    mask = np.arange(len(C)) != index

    other_weights = C[mask] / (1 - C[index])

    start = V[:, mask] @ other_weights
    end = V[:, index]

    points = switching_points(
        start,
        end,
        ALTERNATIVES,
    )

    assert len(points) == 1

    assert points[0]["Switching Point"] == pytest.approx(
        0.477163,
        abs=1e-5,
    )
