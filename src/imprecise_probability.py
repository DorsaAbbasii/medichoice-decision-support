
import numpy as np


def idm_bounds(n, N, s=2):

    if N <= 0:
        raise ValueError("N must be positive.")

    if not 0 <= n <= N:
        raise ValueError("Invalid event count.")

    if s <= 0:
        raise ValueError("s must be positive.")

    lower = n / (N + s)
    upper = (n + s) / (N + s)
    epsilon = s / (N + s)

    return lower, upper, epsilon


def contamination_bounds(probability, epsilon):

    if not 0 <= probability <= 1:
        raise ValueError("Invalid probability.")

    if not 0 <= epsilon <= 1:
        raise ValueError("Invalid epsilon.")

    lower = (1 - epsilon) * probability
    upper = lower + epsilon

    return lower, upper


def robust_expected_utility(
    insured_utility,
    no_event_utility,
    lower_probability,
    upper_probability,
):

    if not 0 <= lower_probability <= upper_probability <= 1:
        raise ValueError("Invalid probability interval.")

    insured = np.asarray(insured_utility, dtype=float)
    no_event = np.asarray(no_event_utility, dtype=float)

    if insured.shape != no_event.shape:
        raise ValueError("Utility dimensions must match.")

    utility_lower_endpoint = (
        lower_probability * insured
        + (1 - lower_probability) * no_event
    )

    utility_upper_endpoint = (
        upper_probability * insured
        + (1 - upper_probability) * no_event
    )

    lower_utility = np.minimum(
        utility_lower_endpoint,
        utility_upper_endpoint,
    )

    upper_utility = np.maximum(
        utility_lower_endpoint,
        utility_upper_endpoint,
    )

    return lower_utility, upper_utility
