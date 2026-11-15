import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, save_path: str | Path
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(recall, precision, color="#1f77b4", lw=2.5, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cost_curve(
    df_cost: pd.DataFrame, optimal_threshold: float, title: str, save_path: str | Path
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(
        df_cost["threshold"],
        df_cost["total_cost"],
        color="#d62728",
        lw=2.5,
        label="Total Expected Cost",
    )
    plt.axvline(
        optimal_threshold,
        color="black",
        linestyle="--",
        label=f"Optimal p* = {optimal_threshold:.2f}",
    )
    plt.axvline(0.5, color="gray", linestyle=":", label="Default 0.50")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Total Financial Cost ($)")
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curve(
    prob_true: list, prob_pred: list, ece: float, title: str, save_path: str | Path
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(
        prob_pred, prob_true, marker="o", lw=2, color="#2ca02c", label=f"Model (ECE = {ece:.4f})"
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, title: str, save_path: str | Path) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0 (Neg)", "Pred 1 (Pos)"],
        yticklabels=["Actual 0 (Neg)", "Actual 1 (Pos)"],
    )
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
