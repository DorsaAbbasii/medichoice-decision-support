
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ahp import calculate_ahp

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


def checked_weights(matrix):

    result = calculate_ahp(matrix)

    if not result["consistent"]:
        raise ValueError("Inconsistent AHP matrix.")

    return result["weights"]


def main():

    print("=" * 70)
    print("MEDICHOICE - AGGREGATION ANALYSIS")
    print("Nonlinear Aggregation Analysis")
    print("=" * 70)

    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Local alternative priorities

    V = np.column_stack([
        checked_weights(PAIRWISE_MATRICES[c])
        for c in CRITERIA_IDS
    ])

    # 2. Individual criteria matrices

    individual_matrices = {
        "Baseline": CRITERIA_MATRIX,
        **SCENARIO_MATRICES,
    }

    # 3. Group competence weights

    group_weights = checked_weights(
        GROUP_COMPETENCE_MATRIX
    )

    records = []
    bottleneck_records = []
    winner_records = []

    def save_comparison(mode, scenario, linear, maximin):

        df = pd.DataFrame({
            "Company": ALTERNATIVES,
            "Linear Score": linear,
            "Maximin Score": maximin,
        })

        df["Linear Rank"] = (
            df["Linear Score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        df["Maximin Rank"] = (
            df["Maximin Score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        best_linear = ALTERNATIVES[np.argmax(linear)]
        best_maximin = ALTERNATIVES[np.argmax(maximin)]

        print("\n" + "=" * 70)
        print(f"{mode.upper()} - {scenario}")
        print("=" * 70)

        print(df.round(6).to_string(index=False))

        print(f"\nBest linear: Company {best_linear}")
        print(f"Best maximin: Company {best_maximin}")

        winner_records.append({
            "Mode": mode,
            "Scenario": scenario,
            "Linear Winner": best_linear,
            "Maximin Winner": best_maximin,
        })

        df.insert(0, "Scenario", scenario)
        df.insert(0, "Mode", mode)

        records.append(df)

    # 4. Individual and group analysis

    for mode in ["Individual", "Group"]:

        scenario_results = {}

        for scenario, matrix in individual_matrices.items():

            if mode == "Individual":

                C = checked_weights(matrix)

                linear = linear_aggregation(V, C)
                maximin = maximin_aggregation(V, C)

                bottlenecks = bottleneck_criteria(V, C)

                for company, index in zip(
                    ALTERNATIVES, bottlenecks
                ):
                    bottleneck_records.append({
                        "Mode": mode,
                        "Scenario": scenario,
                        "Group": "Individual",
                        "Company": company,
                        "Criterion": CRITERIA_IDS[index],
                    })

            else:

                group_linear = []
                group_maximin = []

                for group in GROUP_NAMES:

                    C = checked_weights(
                        GROUP_CRITERIA_MATRICES[
                            scenario
                        ][group]
                    )

                    group_linear.append(
                        linear_aggregation(V, C)
                    )

                    group_maximin.append(
                        maximin_aggregation(V, C)
                    )

                    bottlenecks = bottleneck_criteria(V, C)

                    for company, index in zip(
                        ALTERNATIVES, bottlenecks
                    ):
                        bottleneck_records.append({
                            "Mode": mode,
                            "Scenario": scenario,
                            "Group": group,
                            "Company": company,
                            "Criterion": CRITERIA_IDS[index],
                        })

                linear = (
                    np.column_stack(group_linear)
                    @ group_weights
                )

                maximin = (
                    np.column_stack(group_maximin)
                    @ group_weights
                )

            scenario_results[scenario] = (
                linear,
                maximin,
            )

            save_comparison(
                mode,
                scenario,
                linear,
                maximin,
            )

        # 5. Construct scenario utility matrices

        linear_utilities = np.column_stack([
            scenario_results["Insured Event"][0],
            scenario_results["No Insured Event"][0],
        ])

        maximin_utilities = np.column_stack([
            scenario_results["Insured Event"][1],
            scenario_results["No Insured Event"][1],
        ])

        # 6. Expected utility under risk

        linear_risk = (
            linear_utilities
            @ SCENARIO_PROBABILITIES
        )

        maximin_risk = (
            maximin_utilities
            @ SCENARIO_PROBABILITIES
        )

        save_comparison(
            mode,
            "Expected Utility",
            linear_risk,
            maximin_risk,
        )

        # 7. Wald criterion under uncertainty

        linear_wald = linear_utilities.min(axis=1)
        maximin_wald = maximin_utilities.min(axis=1)

        save_comparison(
            mode,
            "Wald Criterion",
            linear_wald,
            maximin_wald,
        )

    # 8. Save results

    comparison_df = pd.concat(
        records,
        ignore_index=True,
    )

    comparison_df.to_csv(
        tables_dir / "aggregation_comparison.csv",
        index=False,
    )

    pd.DataFrame(bottleneck_records).to_csv(
        tables_dir / "bottleneck_criteria.csv",
        index=False,
    )

    winners_df = pd.DataFrame(winner_records)

    winners_df.to_csv(
        tables_dir / "aggregation_winners.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL WINNERS SUMMARY")
    print("=" * 70)

    print(winners_df.to_string(index=False))

    # 9. Generate comparison figures
    # Separate axes because score scales differ.

    for mode in ["Individual", "Group"]:

        subset = comparison_df[
            (comparison_df["Mode"] == mode)
            & (comparison_df["Scenario"] == "Baseline")
        ]

        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5)
        )

        axes[0].bar(
            subset["Company"],
            subset["Linear Score"],
            color="steelblue",
        )

        axes[0].set_title("Linear Aggregation")
        axes[0].set_xlabel("Insurance Company")
        axes[0].set_ylabel("Linear Score")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(
            subset["Company"],
            subset["Maximin Score"],
            color="seagreen",
        )

        axes[1].set_title("Nonlinear Maximin")
        axes[1].set_xlabel("Insurance Company")
        axes[1].set_ylabel("Maximin Score")
        axes[1].grid(axis="y", alpha=0.25)

        fig.suptitle(f"{mode} Decision Comparison")

        plt.tight_layout()

        plt.savefig(
            figures_dir /
            f"aggregation_{mode.lower()}.png",
            dpi=300,
        )

        plt.close()

    print("\nAll aggregation results saved successfully.")


if __name__ == "__main__":
    main()
