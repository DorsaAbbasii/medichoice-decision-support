
import pandas as pd

from src.matrices import (
    PAIRWISE_MATRICES,
    CRITERIA,
    ALTERNATIVES,
)

from src.ahp import calculate_ahp


def main():

    print("=" * 65)
    print("MEDICHOICE - MULTI-CRITERIA DECISION SUPPORT SYSTEM")
    print("Variant 14: Medical Insurance Company Selection")
    print("=" * 65)

    results = []

    for key, matrix in PAIRWISE_MATRICES.items():

        result = calculate_ahp(matrix)

        print(f"\n{key}: {CRITERIA[key]}")
        print("-" * 65)

        print("Pairwise Comparison Matrix:")
        print(pd.DataFrame(
            matrix,
            index=ALTERNATIVES,
            columns=ALTERNATIVES
        ).round(4))

        print("\nNormalized Weights:")

        for name, weight in zip(
            ALTERNATIVES, result["weights"]
        ):
            print(f"Company {name}: {weight:.6f}")

        print(f"\nLambda max: {result['lambda_max']:.6f}")
        print(f"CI: {result['ci']:.6f}")
        print(f"CR: {result['cr']:.2%}")

        status = (
            "PASS" if result["consistent"] else "FAIL"
        )

        print(f"Consistency: {status}")

        results.append({
            "Criterion": key,
            "Company A": result["weights"][0],
            "Company B": result["weights"][1],
            "Company C": result["weights"][2],
            "Company D": result["weights"][3],
            "CR": result["cr"],
            "Status": status,
        })

    summary = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print(summary.round(4).to_string(index=False))

    summary.to_csv(
        "results/tables/local_priorities.csv",
        index=False
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
