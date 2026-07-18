from pathlib import Path

import matplotlib.pyplot as plt


def plot_threshold_metrics(
    threshold_results,
    selected_threshold
):
    thresholds = [
        result["threshold"]
        for result in threshold_results
    ]

    precision_values = [
        result["precision"]
        for result in threshold_results
    ]

    recall_values = [
        result["recall"]
        for result in threshold_results
    ]

    f1_values = [
        result["f1"]
        for result in threshold_results
    ]

    output_dir = Path("reports") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        precision_values,
        marker="o",
        label="Precision"
    )

    plt.plot(
        thresholds,
        recall_values,
        marker="o",
        label="Recall"
    )

    plt.plot(
        thresholds,
        f1_values,
        marker="o",
        label="F1"
    )

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected threshold: {selected_threshold:.2f}"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Selection on Validation Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "threshold_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()