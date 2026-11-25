
import numpy as np

ALTERNATIVES = ["A", "B", "C", "D"]

CRITERIA = {
    "C1": "Ease of registration",
    "C2": "Company reputation",
    "C3": "Service quality",
    "C4": "Timeliness of payments",
    "C5": "Insurance premium",
    "C6": "Insurance coverage level",
    "C7": "Range of covered diseases",
}


def create_reciprocal_matrix(values):
    """Create a 4x4 reciprocal comparison matrix."""

    matrix = np.eye(4)

    pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    for (i, j), value in zip(pairs, values):
        matrix[i, j] = value
        matrix[j, i] = 1 / value

    return matrix


PAIRWISE_MATRICES = {
    # C1: Ease of registration
    "C1": create_reciprocal_matrix(
        [3, 2, 2, 1/2, 1/2, 2]
    ),

    # C2: Company reputation
    "C2": create_reciprocal_matrix(
        [1/3, 1/2, 1/2, 2, 2, 2]
    ),

    # C3: Service quality
    "C3": create_reciprocal_matrix(
        [1/2, 1/2, 2, 2, 3, 2]
    ),

    # C4: Timeliness of payments
    "C4": create_reciprocal_matrix(
        [1/3, 1/2, 1/2, 2, 2, 2]
    ),

    # C5: Insurance premium (lower is better)
    "C5": create_reciprocal_matrix(
        [2, 2, 1/2, 1/2, 1/3, 1/2]
    ),

    # C6: Insurance coverage level
    "C6": create_reciprocal_matrix(
        [1/3, 1/2, 2, 2, 4, 3]
    ),

    # C7: Range of covered diseases
    "C7": create_reciprocal_matrix(
        [1/3, 1/2, 2, 2, 3, 3]
    ),
}


if __name__ == "__main__":
    for key, matrix in PAIRWISE_MATRICES.items():
        print(f"\n{key}: {CRITERIA[key]}")
        print(np.round(matrix, 4))
