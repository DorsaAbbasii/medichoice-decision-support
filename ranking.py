
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ahp import calculate_ahp
from src.matrices import (
    PAIRWISE_MATRICES,
    CRITERIA,
    ALTERNATIVES,
)
from src.criteria_weights import (
    CRITERIA_MATRIX,
    CRITERIA_IDS,
)


def main():

    print("=" * 65)
    print("MEDICHOICE - GLOBAL AHP RANKING")
    print("=" * 65)

    tables_dir = Path("results/tables")
    matrices_dir = Path("results/matrices")
    figures_dir = Path("results/figures")

    for directory in [
        tables_dir,
        matrices_dir,
        figures_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Calculate criteria weights
    criteria_result = calculate_ahp(CRITERIA_MATRIX)

    criteria_weights = criteria_result["weights"]

    print("\nCRITERIA COMPARISON MATRIX")

    matrix_df = pd.DataFrame(
        CRITERIA_MATRIX,
        index=CRITERIA_IDS,
        columns=CRITERIA_IDS,
    )

    print(matrix_df.round(4))

    print("\nCRITERIA WEIGHTS")

    criteria_df = pd.DataFrame({
        "Criterion": CRITERIA_IDS,
        "Description": [
            CRITERIA[c] for c in CRITERIA_IDS
        ],
        "Weight": criteria_weights,
    })

    print(criteria_df.round(6).to_string(index=False))

    print(
        f"\nLambda max: "
        f"{criteria_result['lambda_max']:.6f}"
    )
    print(f"CI: {criteria_result['ci']:.6f}")
    print(f"CR: {criteria_result['cr']:.2%}")

    if not criteria_result["consistent"]:
        raise ValueError(
            "Criteria matrix is inconsistent."
        )

    print("Consistency: PASS")

    # 2. Calculate local priority matrix
    local_results = []

    for criterion in CRITERIA_IDS:

        result = calculate_ahp(
            PAIRWISE_MATRICES[criterion]
        )

        if not result["consistent"]:
            raise ValueError(
                f"Inconsistent matrix: {criterion}"
            )

        local_results.append(result["weights"])

    priority_matrix = np.column_stack(
        local_results
    )

    priority_df = pd.DataFrame(
        priority_matrix,
        index=ALTERNATIVES,
        columns=CRITERIA_IDS,
    )

    print("\nLOCAL PRIORITY MATRIX")
    print(priority_df.round(6))

    # 3. Calculate global priorities
    global_scores = (
        priority_matrix @ criteria_weights
    )

    ranking_df = pd.DataFrame({
        "Company": ALTERNATIVES,
        "Score": global_scores,
    })

    ranking_df = ranking_df.sort_values(
        "Score",
        ascending=False,
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "Rank",
        range(1, len(ranking_df) + 1),
    )

    print("\nGLOBAL RANKING")
    print("-" * 65)

    print(ranking_df.round(6).to_string(index=False))

    print(
        f"\nBest alternative: Company "
        f"{ranking_df.iloc[0]['Company']}"
    )

    print(
        f"Total priority: "
        f"{global_scores.sum():.6f}"
    )

    # 4. Save results
    matrix_df.to_csv(
        matrices_dir / "criteria_matrix.csv"
    )

    priority_df.to_csv(
        tables_dir / "priority_matrix.csv"
    )

    criteria_df.to_csv(
        tables_dir / "criteria_weights.csv",
        index=False,
    )

    ranking_df.to_csv(
        tables_dir / "global_ranking.csv",
        index=False,
    )

    # 5. Criteria weights chart
    plt.figure(figsize=(11, 6))

    plt.bar(
        CRITERIA_IDS,
        criteria_weights,
        color="steelblue",
    )

    plt.title("AHP Criteria Weights")
    plt.xlabel("Criteria")
    plt.ylabel("Normalized Weight")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "criteria_weights.png",
        dpi=300,
    )

    plt.close()

    # 6. Global ranking chart
    plt.figure(figsize=(9, 5))

    plt.bar(
        ranking_df["Company"],
        ranking_df["Score"],
        color="seagreen",
    )

    plt.title("Medical Insurance Company Ranking")
    plt.xlabel("Insurance Company")
    plt.ylabel("Global Priority")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "global_ranking.png",
        dpi=300,
    )

    plt.close()

    print("\nAll results saved successfully.")


if __name__ == "__main__":
    main()
