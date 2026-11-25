
import numpy as np
from math import floor

GROUP_NAMES = [
    "Patients",
    "Healthcare Experts",
]

GROUP_COMPETENCE_MATRIX = np.array([
    [1, 1/2],
    [2, 1],
], dtype=float)

GROUP_IMPORTANCE_LEVELS = {
    "Baseline": {
        "Patients": [6, 3, 6, 4, 9, 5, 6],
        "Healthcare Experts": [2, 2, 5, 7, 3, 9, 8],
    },

    "Insured Event": {
        "Patients": [2, 1, 5, 8, 3, 9, 8],
        "Healthcare Experts": [1, 2, 5, 8, 2, 9, 9],
    },

    "No Insured Event": {
        "Patients": [6, 3, 5, 1, 9, 2, 2],
        "Healthcare Experts": [3, 2, 5, 3, 6, 7, 7],
    },
}


def build_comparison_matrix(levels):
    """
    Construct a reciprocal Saaty comparison matrix
    from predefined importance levels.
    """

    levels = np.asarray(levels, dtype=float)

    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("Invalid importance levels.")

    if np.any(levels < 1) or np.any(levels > 9):
        raise ValueError(
            "Importance levels must be between 1 and 9."
        )

    n = len(levels)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):

            a = levels[i]
            b = levels[j]

            ratio = max(a, b) / min(a, b)

            saaty_value = min(
                9,
                max(1, floor(ratio + 0.5))
            )

            if a >= b:
                matrix[i, j] = saaty_value
            else:
                matrix[i, j] = 1 / saaty_value

            matrix[j, i] = 1 / matrix[i, j]

    return matrix


GROUP_CRITERIA_MATRICES = {
    scenario: {
        group: build_comparison_matrix(levels)
        for group, levels in groups.items()
    }
    for scenario, groups
    in GROUP_IMPORTANCE_LEVELS.items()
}
