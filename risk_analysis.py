
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ahp import calculate_ahp

from src.matrices import (
    PAIRWISE_MATRICES,
    ALTERNATIVES,
)

from src.criteria_weights import CRITERIA_IDS

from src.scenarios import (
    SCENARIO_NAMES,
    SCENARIO_MATRICES,
    SCENARIO_PROBABILITIES,
)


def main():

    print("=" * 65)
    print("MEDICHOICE - RISK AND UNCERTAINTY ANALYSIS")
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

    # 1. Calculate local alternative priorities
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

    priority_matrix = np.column_stack(local_results)

    scenario_weights = {}
    scenario_utilities = {}
    consistency_records = []

    # 2. Calculate weights for each scenario
    for name in SCENARIO_NAMES:

        matrix = SCENARIO_MATRICES[name]
        result = calculate_ahp(matrix)

        print(f"\nSCENARIO: {name}")
        print("-" * 65)

        matrix_df = pd.DataFrame(
            matrix,
            index=CRITERIA_IDS,
            columns=CRITERIA_IDS,
        )

        print(matrix_df.round(4))

        print("\nCriteria weights:")

        for criterion, weight in zip(
            CRITERIA_IDS,
            result["weights"],
        ):
            print(f"{criterion}: {weight:.6f}")

        print(
            f"\nLambda max: "
            f"{result['lambda_max']:.6f}"
        )
        print(f"CI: {result['ci']:.6f}")
        print(f"CR: {result['cr']:.2%}")

        if not result["consistent"]:
            raise ValueError(
                f"Inconsistent scenario: {name}"
            )

        print("Consistency: PASS")

        scenario_weights[name] = result["weights"]

        # Linear aggregation
        scenario_utilities[name] = (
            priority_matrix @ result["weights"]
        )

        consistency_records.append({
            "Scenario": name,
            "Lambda max": result["lambda_max"],
            "CI": result["ci"],
            "RI": result["ri"],
            "CR": result["cr"],
            "Status": "PASS",
        })

        filename = name.lower().replace(" ", "_")

        matrix_df.to_csv(
            matrices_dir / f"{filename}_matrix.csv"
        )

    # 3. Construct utility matrix
    utility_df = pd.DataFrame(
        scenario_utilities,
        index=ALTERNATIVES,
    )

    utility_df = utility_df[SCENARIO_NAMES]

    utility_matrix = utility_df.to_numpy()

    print("\nUTILITY MATRIX")
    print("-" * 65)
    print(utility_df.round(6))

    # 4. Validate probabilities
    probabilities = SCENARIO_PROBABILITIES

    if (
        np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError(
            "Invalid scenario probabilities."
        )

    # 5. Expected utility (risk)
    expected_utility = utility_matrix @ probabilities

    # 6. Wald criterion (uncertainty)
    wald_scores = utility_matrix.min(axis=1)

    # 7. Final comparison
    comparison = pd.DataFrame({
        "Company": ALTERNATIVES,
        "Insured Event": utility_matrix[:, 0],
        "No Insured Event": utility_matrix[:, 1],
        "Expected Utility": expected_utility,
        "Wald Score": wald_scores,
    })

    comparison["Risk Rank"] = (
        comparison["Expected Utility"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["Wald Rank"] = (
        comparison["Wald Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    print("\nDECISION COMPARISON")
    print("-" * 65)
    print(comparison.round(6).to_string(index=False))

    best_risk = ALTERNATIVES[
        np.argmax(expected_utility)
    ]

    best_wald = ALTERNATIVES[
        np.argmax(wald_scores)
    ]

    print(f"\nBest under risk: Company {best_risk}")
    print(
        f"Best under uncertainty: Company {best_wald}"
    )

    print(
        f"Total expected utility: "
        f"{expected_utility.sum():.6f}"
    )

    # 8. Save tables
    weights_df = pd.DataFrame(
        scenario_weights,
        index=CRITERIA_IDS,
    )

    weights_df.to_csv(
        tables_dir / "scenario_weights.csv"
    )

    utility_df.to_csv(
        tables_dir / "utility_matrix.csv"
    )

    comparison.to_csv(
        tables_dir / "risk_uncertainty_results.csv",
        index=False,
    )

    pd.DataFrame(consistency_records).to_csv(
        tables_dir / "scenario_consistency.csv",
        index=False,
    )

    # 9. Scenario criteria weights chart
    ax = weights_df.plot(
        kind="bar",
        figsize=(11, 6),
        rot=0,
    )

    ax.set_title("Criteria Weights by Scenario")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Normalized Weight")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        figures_dir / "scenario_weights.png",
        dpi=300,
    )

    plt.close()

    # 10. Utility comparison chart
    ax = utility_df.plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )

    ax.set_title("Insurance Company Utilities by Scenario")
    ax.set_xlabel("Insurance Company")
    ax.set_ylabel("Utility")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        figures_dir / "scenario_utilities.png",
        dpi=300,
    )

    plt.close()

    # 11. Risk and uncertainty comparison chart
    ax = comparison.set_index("Company")[
        ["Expected Utility", "Wald Score"]
    ].plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )

    ax.set_title("Decision Making under Risk and Uncertainty")
    ax.set_xlabel("Insurance Company")
    ax.set_ylabel("Decision Score")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        figures_dir / "risk_uncertainty_comparison.png",
        dpi=300,
    )

    plt.close()

    print("\nAll risk analysis results saved successfully.")


if __name__ == "__main__":
    main()
