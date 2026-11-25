
import numpy as np

CRITERIA_IDS = [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
]

# Baseline criteria comparison matrix
CRITERIA_MATRIX = np.array([
    [1,   2,   1/2, 1/2, 1/3, 1/2, 1/2],
    [1/2, 1,   1/3, 1/3, 1/5, 1/4, 1/4],
    [2,   3,   1,   1,   1/2, 1/2, 1/2],
    [2,   3,   1,   1,   1/2, 1/2, 1/2],
    [3,   5,   2,   2,   1,   2,   2],
    [2,   4,   2,   2,   1/2, 1,   1],
    [2,   4,   2,   2,   1/2, 1,   1],
], dtype=float)
