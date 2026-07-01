"""
PointNet utility modules with spiking LIF neurons.

Provides:
  - SharedMLP:              Linear + BatchNorm1d + LIF
  - TNet:                   Input/Feature transform network using LIF throughout
  - stn_regularization_loss: Orthogonality regularization for T-Net matrices

All modules handle 4D input [B, T, N, C] by reshaping to [B*T, N, C] internally,
then restoring to [B, T, N, C] on output. This enables transparent multi-timestep
processing with standard 1D batch-norm semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional

from .lif_neuron import make_lif_neuron


class SharedMLP(nn.Module):
    """
    Shared MLP block: Linear -> BatchNorm1d -> LIF.

    Operates on point features of shape [B, N, C_in] or [B, T, N, C_in].
    The 'shared' refers to weights being applied identically across all N points
    (equivalent to a 1x1 convolution over the point dimension).

    Args:
        in_channels:  Input feature dimension.
        out_channels: Output feature dimension.
        tau:          LIF membrane decay factor.
        V_th:         LIF spike threshold.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tau: float = 0.25,
        V_th: float = 0.5,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = make_lif_neuron(tau=tau, V_th=V_th, step_mode='s')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] or [B, T, N, C]

        Returns:
            Spike tensor of same shape prefix with C replaced by out_channels.
        """
        original_shape = x.shape
        if x.dim() == 4:
            B, T, N, C = x.shape
            x = x.reshape(B * T, N, C)
            flat_4d = True
        else:
            B, N, C = x.shape
            flat_4d = False

        # Linear: [B*T, N, C] -> [B*T, N, out_channels]
        x = self.linear(x)

        # BatchNorm1d expects [B*T, out_channels, N] or [B*T*N, out_channels]
        # Use the [B*T*N, out_channels] approach for simplicity
        BT, N_, out_c = x.shape
        x = x.reshape(BT * N_, out_c)
        x = self.bn(x)
        x = x.reshape(BT, N_, out_c)

        # LIF activation
        x = self.lif(x)

        if flat_4d:
            x = x.reshape(B, T, N_, out_c)

        return x


class TNet(nn.Module):
    """
    Spatial Transformer Network (T-Net) for learning input/feature transformations.
    Uses LIF neurons throughout.

    For k=3: produces a 3x3 input transform matrix.
    For k=64: produces a 64x64 feature transform matrix.

    Architecture follows original PointNet T-Net:
        SharedMLP(k, 64) -> SharedMLP(64, 128) -> SharedMLP(128, 1024)
        Global max-pool -> [B, T, 1024]
        FC(1024, 512) + BN + LIF
        FC(512, 256) + BN + LIF
        FC(256, k*k)
        Reshape to [B, T, k, k] + identity bias

    Args:
        k:    Transform dimension (3 for input, 64 for feature).
        tau:  LIF decay factor.
        V_th: LIF threshold.
    """

    def __init__(self, k: int = 3, tau: float = 0.25, V_th: float = 0.5):
        super().__init__()
        self.k = k

        # Point-wise MLPs
        self.mlp1 = SharedMLP(k, 64, tau=tau, V_th=V_th)
        self.mlp2 = SharedMLP(64, 128, tau=tau, V_th=V_th)
        self.mlp3 = SharedMLP(128, 1024, tau=tau, V_th=V_th)

        # FC layers after global pooling
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.lif1 = make_lif_neuron(tau=tau, V_th=V_th, step_mode='s')

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.lif2 = make_lif_neuron(tau=tau, V_th=V_th, step_mode='s')

        self.fc3 = nn.Linear(256, k * k)

        # Initialize close to identity: tiny weight so transform ≈ I + ε
        nn.init.normal_(self.fc3.weight, std=1e-3)
        nn.init.zeros_(self.fc3.bias)
        # Identity matrix as bias
        identity = torch.eye(k).flatten()
        self.register_buffer('identity', identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, k] or [B, N, k]

        Returns:
            Transform matrices [B, T, k, k] or [B, k, k]
        """
        if x.dim() == 3:
            # [B, N, k] -> add T dim
            x = x.unsqueeze(1)
            squeeze_T = True
        else:
            squeeze_T = False

        B, T, N, k = x.shape

        # Shared MLPs
        x = self.mlp1(x)     # [B, T, N, 64]
        x = self.mlp2(x)     # [B, T, N, 128]
        x = self.mlp3(x)     # [B, T, N, 1024]

        # Mean pool over points: max of binary spikes over N≫1 always collapses to 1;
        # mean gives the firing rate (~0.25), which varies across channels and samples.
        x = x.mean(dim=2)  # [B, T, 1024]

        # Reshape for FC layers: process as [B*T, 1024]
        x = x.reshape(B * T, 1024)

        # FC layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.lif1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.lif2(x)

        x = self.fc3(x)   # [B*T, k*k]

        # Add identity bias
        x = x + self.identity.unsqueeze(0)

        # Reshape to transform matrix
        x = x.reshape(B, T, k, k)

        if squeeze_T:
            x = x.squeeze(1)  # [B, k, k]

        return x


def apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply a spatial transform matrix to point cloud features.

    Args:
        points:    [B, T, N, C] or [B, N, C]
        transform: [B, T, C, C] or [B, C, C]

    Returns:
        Transformed points of same shape as input.
    """
    if points.dim() == 3 and transform.dim() == 3:
        # [B, N, C] x [B, C, C] -> [B, N, C]
        return torch.bmm(points, transform)
    elif points.dim() == 4 and transform.dim() == 4:
        # [B, T, N, C] x [B, T, C, C] -> [B, T, N, C]
        B, T, N, C = points.shape
        p = points.reshape(B * T, N, C)
        t = transform.reshape(B * T, C, C)
        out = torch.bmm(p, t)
        return out.reshape(B, T, N, C)
    else:
        raise ValueError(
            f"Mismatched dims: points {points.shape}, transform {transform.shape}"
        )


def stn_regularization_loss(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute T-Net orthogonality regularization loss.

    L_reg = ||I - A*A^T||_F^2 * 0.001

    Args:
        mat: Transform matrix [B, k, k] or [B, T, k, k].

    Returns:
        Scalar loss value.
    """
    if mat.dim() == 4:
        B, T, k, _ = mat.shape
        mat = mat.reshape(B * T, k, k)
    else:
        B, k, _ = mat.shape

    # A * A^T
    aat = torch.bmm(mat, mat.transpose(1, 2))

    # Identity
    eye = torch.eye(k, device=mat.device, dtype=mat.dtype).unsqueeze(0)
    eye = eye.expand(aat.size(0), -1, -1)

    # Frobenius norm squared
    diff = eye - aat
    loss = (diff * diff).sum(dim=(1, 2)).mean()

    return loss * 0.001
