
import numpy as np

SCENARIO_NAMES = [
    "Insured Event",
    "No Insured Event",
]

SCENARIO_PROBABILITIES = np.array([
    0.02,
    0.98,
])

# S1: Insured event occurs
INSURED_EVENT_MATRIX = np.array([
    [1,   1,   1/3, 1/6, 1/2, 1/7, 1/6],
    [1,   1,   1/3, 1/6, 1/2, 1/7, 1/6],
    [3,   3,   1,   1/2, 2,   1/2, 1/2],
    [6,   6,   2,   1,   3,   1,   1],
    [2,   2,   1/2, 1/3, 1,   1/4, 1/3],
    [7,   7,   2,   1,   4,   1,   1],
    [6,   6,   2,   1,   3,   1,   1],
], dtype=float)

# S2: No insured event occurs
NO_EVENT_MATRIX = np.array([
    [1,   2,   1,   4,   1/2, 4,   4],
    [1/2, 1,   1/2, 2,   1/5, 2,   2],
    [1,   2,   1,   3,   1/3, 3,   3],
    [1/4, 1/2, 1/3, 1,   1/9, 1,   1],
    [2,   5,   3,   9,   1,   9,   9],
    [1/4, 1/2, 1/3, 1,   1/9, 1,   1],
    [1/4, 1/2, 1/3, 1,   1/9, 1,   1],
], dtype=float)

SCENARIO_MATRICES = {
    "Insured Event": INSURED_EVENT_MATRIX,
    "No Insured Event": NO_EVENT_MATRIX,
}
