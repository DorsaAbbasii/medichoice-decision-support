
import numpy as np

RI_VALUES = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

CONSISTENCY_THRESHOLD = 0.20


def validate_matrix(matrix):
    """Validate a pairwise comparison matrix."""

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Matrix must contain finite values.")

    if np.any(matrix <= 0):
        raise ValueError("Matrix values must be positive.")

    if not np.allclose(np.diag(matrix), 1):
        raise ValueError("Diagonal values must equal 1.")

    if not np.allclose(matrix * matrix.T, 1):
        raise ValueError("Matrix must be reciprocal.")

    return True


def calculate_ahp(matrix):
    """Calculate AHP priorities and consistency."""

    matrix = np.asarray(matrix, dtype=float)

    validate_matrix(matrix)

    n = matrix.shape[0]

    # Step 1: Geometric mean of each row
    geometric_means = np.prod(matrix, axis=1) ** (1 / n)

    # Step 2: Normalize the weights
    weights = geometric_means / geometric_means.sum()

    # Step 3: Sum matrix columns
    column_sums = matrix.sum(axis=0)

    # Step 4: Estimate maximum eigenvalue
    lambda_max = np.dot(column_sums, weights)

    # Step 5: Consistency Index
    ci = max(0.0, (lambda_max - n) / (n - 1)) if n > 1 else 0.0

    # Step 6: Consistency Ratio
    ri = RI_VALUES[n]

    cr = ci / ri if ri > 0 else 0.0

    return {
        "geometric_means": geometric_means,
        "weights": weights,
        "lambda_max": lambda_max,
        "ci": ci,
        "ri": ri,
        "cr": cr,
        "consistent": cr <= CONSISTENCY_THRESHOLD,
    }
