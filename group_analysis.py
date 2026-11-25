
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

from src.scenarios import SCENARIO_PROBABILITIES

from src.group_decision import (
    GROUP_NAMES,
    GROUP_COMPETENCE_MATRIX,
    GROUP_CRITERIA_MATRICES,
)


def main():

    print("=" * 70)
    print("MEDICHOICE - GROUP DECISION MAKING")
    print("Group Decision Analysis")
    print("=" * 70)

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

    local_weights = []

    for criterion in CRITERIA_IDS:

        result = calculate_ahp(
            PAIRWISE_MATRICES[criterion]
        )

        if not result["consistent"]:
            raise ValueError(
                f"Inconsistent matrix: {criterion}"
            )

        local_weights.append(result["weights"])

    priority_matrix = np.column_stack(local_weights)

    # 2. Calculate group competence weights

    competence = calculate_ahp(
        GROUP_COMPETENCE_MATRIX
    )

    if not competence["consistent"]:
        raise ValueError(
            "Invalid group competence matrix."
        )

    group_weights = competence["weights"]

    competence_df = pd.DataFrame({
        "Group": GROUP_NAMES,
        "Weight": group_weights,
    })

    print("\nGROUP COMPETENCE WEIGHTS")
    print(competence_df.round(6).to_string(index=False))
    print(f"CR: {competence['cr']:.2%}")

    competence_df.to_csv(
        tables_dir / "group_competence_weights.csv",
        index=False,
    )

    pd.DataFrame(
        GROUP_COMPETENCE_MATRIX,
        index=GROUP_NAMES,
        columns=GROUP_NAMES,
    ).to_csv(
        matrices_dir / "group_competence_matrix.csv"
    )

    scenario_scores = {}

    consistency_records = [{
        "Scenario": "All",
        "Group": "Group competence",
        "Lambda max": competence["lambda_max"],
        "CI": competence["ci"],
        "RI": competence["ri"],
        "CR": competence["cr"],
        "Status": "PASS",
    }]

    criteria_records = []

    # 3. Evaluate each scenario

    for scenario, matrices in GROUP_CRITERIA_MATRICES.items():

        print("\n" + "=" * 70)
        print(f"SCENARIO: {scenario}")
        print("=" * 70)

        group_scores = []

        scenario_slug = scenario.lower().replace(" ", "_")

        for group in GROUP_NAMES:

            matrix = matrices[group]

            result = calculate_ahp(matrix)

            if not result["consistent"]:
                raise ValueError(
                    f"Inconsistent matrix: "
                    f"{scenario} / {group}"
                )

            weights = result["weights"]

            scores = priority_matrix @ weights

            group_scores.append(scores)

            print(f"\nGROUP: {group}")
            print("-" * 70)

            matrix_df = pd.DataFrame(
                matrix,
                index=CRITERIA_IDS,
                columns=CRITERIA_IDS,
            )

            print(matrix_df.round(4))

            print("\nCriteria weights:")

            for criterion, weight in zip(
                CRITERIA_IDS, weights
            ):
                print(f"{criterion}: {weight:.6f}")

                criteria_records.append({
                    "Scenario": scenario,
                    "Group": group,
                    "Criterion": criterion,
                    "Weight": weight,
                })

            print(f"\nCR: {result['cr']:.2%}")
            print("Consistency: PASS")

            print("\nAlternative scores:")

            for company, score in zip(
                ALTERNATIVES, scores
            ):
                print(
                    f"Company {company}: {score:.6f}"
                )

            consistency_records.append({
                "Scenario": scenario,
                "Group": group,
                "Lambda max": result["lambda_max"],
                "CI": result["ci"],
                "RI": result["ri"],
                "CR": result["cr"],
                "Status": "PASS",
            })

            group_slug = group.lower().replace(" ", "_")

            matrix_df.to_csv(
                matrices_dir /
                f"group_{scenario_slug}_{group_slug}.csv"
            )

        # 4. Aggregate group priorities

        score_matrix = np.column_stack(group_scores)

        combined_scores = score_matrix @ group_weights

        scenario_scores[scenario] = combined_scores

        scenario_df = pd.DataFrame(
            score_matrix,
            index=ALTERNATIVES,
            columns=GROUP_NAMES,
        )

        scenario_df["Combined Score"] = combined_scores

        scenario_df["Rank"] = (
            scenario_df["Combined Score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        scenario_df = scenario_df.sort_values("Rank")

        print("\nGROUP DECISION RESULTS")
        print(scenario_df.round(6))

        print(
            f"\nBest alternative: Company "
            f"{scenario_df.index[0]}"
        )

        scenario_df.to_csv(
            tables_dir /
            f"group_results_{scenario_slug}.csv"
        )

    # 5. Group decision under risk and uncertainty

    utility_matrix = np.column_stack([
        scenario_scores["Insured Event"],
        scenario_scores["No Insured Event"],
    ])

    expected_utility = (
        utility_matrix @ SCENARIO_PROBABILITIES
    )

    wald_scores = utility_matrix.min(axis=1)

    final_df = pd.DataFrame({
        "Company": ALTERNATIVES,
        "Baseline": scenario_scores["Baseline"],
        "Insured Event": utility_matrix[:, 0],
        "No Insured Event": utility_matrix[:, 1],
        "Expected Utility": expected_utility,
        "Wald Score": wald_scores,
    })

    final_df["Risk Rank"] = (
        final_df["Expected Utility"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    final_df["Wald Rank"] = (
        final_df["Wald Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    print("\n" + "=" * 70)
    print("FINAL GROUP DECISION SUMMARY")
    print("=" * 70)

    print(final_df.round(6).to_string(index=False))

    print(
        f"\nBest group decision under risk: Company "
        f"{ALTERNATIVES[np.argmax(expected_utility)]}"
    )

    print(
        f"Best group decision under uncertainty: Company "
        f"{ALTERNATIVES[np.argmax(wald_scores)]}"
    )

    # 6. Save results

    final_df.to_csv(
        tables_dir / "group_decision_summary.csv",
        index=False,
    )

    pd.DataFrame(consistency_records).to_csv(
        tables_dir / "group_consistency.csv",
        index=False,
    )

    pd.DataFrame(criteria_records).to_csv(
        tables_dir / "group_criteria_weights.csv",
        index=False,
    )

    # 7. Visualization

    ax = final_df.set_index("Company")[
        ["Baseline", "Expected Utility", "Wald Score"]
    ].plot(
        kind="bar",
        figsize=(11, 6),
        rot=0,
    )

    ax.set_title("Group Decision Making Results")
    ax.set_xlabel("Insurance Company")
    ax.set_ylabel("Decision Score")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        figures_dir / "group_decision_comparison.png",
        dpi=300,
    )

    plt.close()

    print("\nAll group decision results saved successfully.")


if __name__ == "__main__":
    main()
