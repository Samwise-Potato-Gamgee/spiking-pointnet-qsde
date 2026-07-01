"""
Loss functions for Spiking PointNet training.

SpikingPointNetLoss:
    cross-entropy with label smoothing + T-Net orthogonality regularization.

The T-Net regularization encourages the predicted transform matrices to be
close to orthogonal (rotation / reflection), which stabilizes training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingPointNetLoss(nn.Module):
    """
    Combined loss: Cross-Entropy (with label smoothing) + T-Net regularization.

    Args:
        label_smoothing: Label smoothing coefficient (default 0.2).
        reg_weight:      Weight for T-Net orthogonality loss (default 1.0).
                         stn_regularization_loss already multiplies by 0.001 internally,
                         so reg_weight=1.0 matches the paper's exact formulation.
    """

    def __init__(
        self,
        label_smoothing: float = 0.2,
        reg_weight: float = 1.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reg_weight = reg_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        stn3_loss: torch.Tensor,
        stn64_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            logits:    Predicted class logits [B, num_classes].
            labels:    Ground truth class indices [B].
            stn3_loss:  Scalar regularization loss from input T-Net (already weighted).
            stn64_loss: Scalar regularization loss from feature T-Net (already weighted).

        Returns:
            Scalar total loss.
        """
        ce_loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        total = ce_loss + self.reg_weight * (stn3_loss + stn64_loss)

        return total

    def __repr__(self) -> str:
        return (
            f"SpikingPointNetLoss("
            f"label_smoothing={self.label_smoothing}, "
            f"reg_weight={self.reg_weight})"
        )
