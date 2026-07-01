"""
Evaluation metrics for point cloud classification.

Provides:
  - overall_accuracy:      Overall accuracy (OA) = correct / total
  - mean_class_accuracy:   Per-class recall averaged over all classes (mAcc)
"""

import torch
import numpy as np
from typing import Union


def overall_accuracy(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute overall accuracy.

    OA = number_correct / total_samples

    Args:
        preds:  Predicted class indices [N].
        labels: Ground truth class indices [N].

    Returns:
        Float accuracy in [0, 1].
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    if len(preds) == 0:
        return 0.0

    return float(np.mean(preds == labels))


def mean_class_accuracy(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    num_classes: int = 40,
) -> float:
    """
    Compute mean class accuracy (mAcc).

    mAcc = (1/C) * sum_c [ correct_c / total_c ]

    Classes with zero samples are excluded from the average.

    Args:
        preds:       Predicted class indices [N].
        labels:      Ground truth class indices [N].
        num_classes: Number of classes (default 40 for ModelNet40).

    Returns:
        Float mean class accuracy in [0, 1].
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    class_accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue  # skip classes not present in this split
        class_acc = np.mean(preds[mask] == labels[mask])
        class_accs.append(class_acc)

    if len(class_accs) == 0:
        return 0.0

    return float(np.mean(class_accs))
