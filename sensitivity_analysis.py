
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.imprecise_probability import (
    idm_bounds,
    contamination_bounds,
    robust_expected_utility,
)

TABLES = Path("results/tables")
FIGURES = Path("results/figures")


def plot_sensitivity(
    x,
    scores,
    companies,
    title,
    xlabel,
    filename,
    original_value=None,
):

    plt.figure(figsize=(10, 6))

    for i, company in enumerate(companies):

        plt.plot(
            x,
            scores[:, i],
            linewidth=2,
            label=f"Company {company}",
        )

    if original_value is not None:

        plt.axvline(
            original_value,
            linestyle="--",
            color="gray",
            label="Original value",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Decision Score")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300,
    )

    plt.close()


def switching_points(start, end, companies):

    slopes = end - start
    candidates = []

    for i in range(len(companies)):
        for j in range(i + 1, len(companies)):

            denominator = slopes[i] - slopes[j]

            if np.isclose(denominator, 0):
                continue

            p = (
                start[j] - start[i]
            ) / denominator

            if 0 < p < 1:

                scores = start + p * slopes

                if np.isclose(
                    scores[i], scores.max()
                ):

                    candidates.append({
                        "Company 1": companies[i],
                        "Company 2": companies[j],
                        "Switching Point": p,
                    })

    return candidates


def main():

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEDICHOICE - SENSITIVITY AND ROBUSTNESS")
    print("=" * 70)

    # Experiment 1: Probability sensitivity

    individual = pd.read_csv(
        TABLES / "utility_matrix.csv",
        index_col=0,
    )

    group = pd.read_csv(
        TABLES / "group_decision_summary.csv",
        index_col="Company",
    )

    probability_grid = np.linspace(0, 1, 1001)

    datasets = {
        "Individual": individual,
        "Group": group,
    }

    switch_records = []
    robust_records = []

    for mode, dataset in datasets.items():

        companies = dataset.index.to_list()

        insured = dataset["Insured Event"].to_numpy()

        no_event = dataset[
            "No Insured Event"
        ].to_numpy()

        scores = (
            probability_grid[:, None] * insured
            + (1 - probability_grid[:, None]) * no_event
        )

        df = pd.DataFrame(
            scores,
            columns=companies,
        )

        df.insert(
            0,
            "Probability",
            probability_grid,
        )

        df["Best Company"] = [
            companies[i]
            for i in np.argmax(scores, axis=1)
        ]

        slug = mode.lower()

        df.to_csv(
            TABLES /
            f"probability_sensitivity_{slug}.csv",
            index=False,
        )

        plot_sensitivity(
            probability_grid,
            scores,
            companies,
            f"{mode} Probability Sensitivity",
            "Probability of Insured Event",
            f"probability_sensitivity_{slug}.png",
            original_value=0.02,
        )

        print(f"\n{mode.upper()} PROBABILITY SENSITIVITY")

        switches = switching_points(
            no_event,
            insured,
            companies,
        )

        if not switches:
            print("No winner switching point.")

        for item in switches:

            item["Experiment"] = mode

            switch_records.append(item)

            print(
                f"Company {item['Company 1']} / "
                f"Company {item['Company 2']}: "
                f"{item['Switching Point']:.4%}"
            )

        # Experiment 3: IDM robust utility

        lower, upper, epsilon = idm_bounds(
            n=2,
            N=100,
            s=2,
        )

        contamination = contamination_bounds(
            0.02,
            epsilon,
        )

        if not np.allclose(
            [lower, upper],
            contamination,
        ):
            raise ValueError(
                "IDM and contamination bounds differ."
            )

        robust_lower, robust_upper = (
            robust_expected_utility(
                insured,
                no_event,
                lower,
                upper,
            )
        )

        print(f"\n{mode.upper()} IDM RESULTS")
        print(f"Lower probability: {lower:.6f}")
        print(f"Upper probability: {upper:.6f}")
        print(f"Epsilon: {epsilon:.6f}")

        robust_df = pd.DataFrame({
            "Company": companies,
            "Lower Utility": robust_lower,
            "Upper Utility": robust_upper,
        })

        robust_df["Robust Rank"] = (
            robust_df["Lower Utility"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        print(
            robust_df.round(6).to_string(index=False)
        )

        winner = companies[np.argmax(robust_lower)]

        print(f"Best robust company: {winner}")

        robust_df.insert(0, "Mode", mode)

        robust_records.append(robust_df)

    # Experiment 2: Premium weight sensitivity

    V_df = pd.read_csv(
        TABLES / "priority_matrix.csv",
        index_col=0,
    )

    criteria_df = pd.read_csv(
        TABLES / "criteria_weights.csv"
    ).set_index("Criterion")

    companies = V_df.index.to_list()

    criteria_ids = V_df.columns.to_list()

    weights = criteria_df.loc[
        criteria_ids, "Weight"
    ].to_numpy()

    V = V_df.to_numpy()

    premium_index = criteria_ids.index("C5")
    original_weight = weights[premium_index]

    mask = np.arange(len(weights)) != premium_index

    other_weights = (
        weights[mask] / (1 - original_weight)
    )

    other_scores = V[:, mask] @ other_weights

    premium_scores = V[:, premium_index]

    weight_grid = np.linspace(0, 1, 1001)

    scores = (
        (1 - weight_grid[:, None]) * other_scores
        + weight_grid[:, None] * premium_scores
    )

    df = pd.DataFrame(
        scores,
        columns=companies,
    )

    df.insert(
        0,
        "Premium Weight",
        weight_grid,
    )

    df["Best Company"] = [
        companies[i]
        for i in np.argmax(scores, axis=1)
    ]

    df.to_csv(
        TABLES / "premium_sensitivity.csv",
        index=False,
    )

    plot_sensitivity(
        weight_grid,
        scores,
        companies,
        "Insurance Premium Weight Sensitivity",
        "Weight of Criterion C5",
        "premium_sensitivity.png",
        original_value=original_weight,
    )

    print("\nPREMIUM WEIGHT SENSITIVITY")

    switches = switching_points(
        other_scores,
        premium_scores,
        companies,
    )

    for item in switches:

        item["Experiment"] = "Premium Weight"

        switch_records.append(item)

        print(
            f"Company {item['Company 1']} / "
            f"Company {item['Company 2']}: "
            f"{item['Switching Point']:.4%}"
        )

    # Save final results

    pd.concat(
        robust_records,
        ignore_index=True,
    ).to_csv(
        TABLES / "idm_robust_results.csv",
        index=False,
    )

    pd.DataFrame(switch_records).to_csv(
        TABLES / "decision_switching_points.csv",
        index=False,
    )

    print("\nAll sensitivity results saved successfully.")


if __name__ == "__main__":
    main()
