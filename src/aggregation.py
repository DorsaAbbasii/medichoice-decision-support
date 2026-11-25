
import numpy as np


def validate_inputs(priority_matrix, criteria_weights):

    V = np.asarray(priority_matrix, dtype=float)
    C = np.asarray(criteria_weights, dtype=float)

    if V.ndim != 2 or C.ndim != 1:
        raise ValueError("Invalid input dimensions.")

    if V.shape[1] != len(C):
        raise ValueError("Incompatible dimensions.")

    if not np.all(np.isfinite(V)):
        raise ValueError("Invalid priority values.")

    if not np.all(np.isfinite(C)):
        raise ValueError("Invalid criteria weights.")

    if np.any(V < 0) or np.any(C < 0):
        raise ValueError("Negative weights detected.")

    if not np.isclose(C.sum(), 1.0):
        raise ValueError("Criteria weights must sum to 1.")

    if not np.allclose(V.sum(axis=0), 1.0):
        raise ValueError("Local priorities must sum to 1.")

    return V, C


def linear_aggregation(priority_matrix, criteria_weights):

    V, C = validate_inputs(
        priority_matrix,
        criteria_weights
    )

    return V @ C


def maximin_aggregation(priority_matrix, criteria_weights):

    V, C = validate_inputs(
        priority_matrix,
        criteria_weights
    )

    weighted_priorities = V * C

    return np.min(weighted_priorities, axis=1)


def bottleneck_criteria(priority_matrix, criteria_weights):

    V, C = validate_inputs(
        priority_matrix,
        criteria_weights
    )

    return np.argmin(V * C, axis=1)
