"""
Visualization utilities for Spiking PointNet.

Provides:
  - plot_training_curves:  Loss and OA over epochs from CSV.
  - plot_confusion_matrix: Seaborn heatmap of predictions vs labels.
  - compare_models:        Side-by-side OA comparison of multiple runs.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(
    csv_path: Union[str, Path],
    save_dir: Union[str, Path] = "results",
    show: bool = False,
) -> None:
    """
    Plot training loss and OA curves from a metrics CSV file.

    Creates two subplots:
      Left:  Training loss and validation loss over epochs.
      Right: Training OA and validation OA over epochs.

    Args:
        csv_path: Path to the metrics CSV (output of Trainer).
        save_dir: Directory to save the plot PNG.
        show:     Call plt.show() after saving (for interactive use).
    """
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    run_name = csv_path.stem.replace("metrics_", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=14, fontweight='bold')

    # Loss plot
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:blue', linewidth=1.5)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:orange', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # OA plot
    ax = axes[1]
    ax.plot(df['epoch'], df['train_oa'] * 100, label='Train OA', color='tab:blue', linewidth=1.5)
    ax.plot(df['epoch'], df['val_oa'] * 100, label='Val OA', color='tab:orange', linewidth=1.5)
    if 'val_macc' in df.columns:
        ax.plot(
            df['epoch'], df['val_macc'] * 100,
            label='Val mAcc', color='tab:green', linewidth=1.5, linestyle='--'
        )
    best_oa = df['val_oa'].max() * 100
    best_epoch = df.loc[df['val_oa'].idxmax(), 'epoch']
    ax.axhline(best_oa, color='tab:orange', linestyle=':', alpha=0.6,
               label=f"Best OA={best_oa:.2f}% (ep {best_epoch})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / f"curves_{run_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {out_path}")

    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    preds: Union[np.ndarray, List[int]],
    labels: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    save_dir: Union[str, Path] = "results",
    run_name: str = "model",
    show: bool = False,
    normalize: bool = True,
) -> None:
    """
    Plot a confusion matrix heatmap using seaborn.

    Args:
        preds:        Predicted class indices [N].
        labels:       Ground truth class indices [N].
        class_names:  List of class name strings (length = num_classes).
        save_dir:     Directory to save the plot.
        run_name:     Name prefix for the saved file.
        show:         Call plt.show() after saving.
        normalize:    Normalize by row (true class totals) for recall-based view.
    """
    from sklearn.metrics import confusion_matrix

    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    num_classes = int(max(labels.max(), preds.max())) + 1

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_plot = cm.astype(float) / row_sums
        fmt = '.2f'
        title_suffix = " (normalized)"
    else:
        cm_plot = cm
        fmt = 'd'
        title_suffix = ""

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # For 40 classes, hide individual cell values and use smaller font
    annot = bool(num_classes <= 20)
    figsize = (20, 18) if num_classes > 20 else (14, 12)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot,
        annot=annot,
        fmt=fmt if annot else '',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        ax=ax,
        linewidths=0.3 if num_classes <= 20 else 0,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f"Confusion Matrix — {run_name}{title_suffix}", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8 if num_classes > 20 else 10)
    plt.yticks(rotation=0, fontsize=8 if num_classes > 20 else 10)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"confusion_{run_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {out_path}")

    if show:
        plt.show()
    plt.close()


def compare_models(
    csv_paths: List[Union[str, Path]],
    model_names: Optional[List[str]] = None,
    save_dir: Union[str, Path] = "results",
    show: bool = False,
) -> None:
    """
    Side-by-side comparison plot of multiple model training runs.

    Creates:
      Left:  Val OA over epochs for all models.
      Right: Bar chart of best val OA for each model.

    Args:
        csv_paths:   List of metric CSV file paths.
        model_names: Display names for each model (defaults to CSV stem).
        save_dir:    Directory to save the comparison plot.
        show:        Call plt.show() after saving.
    """
    csv_paths = [Path(p) for p in csv_paths]

    if model_names is None:
        model_names = [p.stem.replace("metrics_", "") for p in csv_paths]

    dataframes = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dataframes.append(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — ModelNet40", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10.colors

    # Val OA curves
    ax = axes[0]
    best_oas = []
    for i, (df, name) in enumerate(zip(dataframes, model_names)):
        color = colors[i % len(colors)]
        ax.plot(df['epoch'], df['val_oa'] * 100, label=name, color=color, linewidth=1.8)
        best_oa = df['val_oa'].max() * 100
        best_oas.append(best_oa)
        ax.axhline(best_oa, color=color, linestyle=':', alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val OA (%)')
    ax.set_title('Validation OA over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar chart of best OA
    ax = axes[1]
    bars = ax.bar(model_names, best_oas, color=colors[:len(model_names)])
    ax.set_ylabel('Best Val OA (%)')
    ax.set_title('Best Validation OA')
    ax.set_ylim(min(best_oas) * 0.95, max(best_oas) * 1.02)
    for bar, oa in zip(bars, best_oas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{oa:.2f}%",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    plt.xticks(rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved model comparison to {out_path}")

    if show:
        plt.show()
    plt.close()
