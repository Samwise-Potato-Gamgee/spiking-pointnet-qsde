"""
Spiking PointNet: full model combining T-Nets, Shared MLPs, and classifier.

Architecture (per NeurIPS 2023 paper):

    Input: [B, N, 3]
    After encoding: [B, T, N, 3]

    Per-timestep processing (batch-flattened as [B*T, N, C]):
        T-Net (input, k=3):  [B, T, N, 3]  -> [B, T, 3, 3]
        Apply transform:     [B, T, N, 3]  -> [B, T, N, 3]
        SharedMLP: 3 -> 64
        SharedMLP: 64 -> 64
        T-Net (feat, k=64):  [B, T, N, 64] -> [B, T, 64, 64]
        Apply transform:     [B, T, N, 64] -> [B, T, N, 64]
        SharedMLP: 64 -> 128
        SharedMLP: 128 -> 1024
        Global max pool:     [B, T, N, 1024] -> [B, T, 1024]
        Mean over T:         [B, T, 1024]    -> [B, 1024]

    Classifier:
        FC(1024, 512) + BN + LIF
        FC(512, 256) + BN + LIF + Dropout(0.3)
        FC(256, num_classes)     -- no LIF, raw logits

Forward returns:
    (logits [B, num_classes], stn3_loss scalar, stn64_loss scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from .lif_neuron import make_lif_neuron
from .pointnet_utils import SharedMLP, TNet, apply_transform, stn_regularization_loss


class SpikingPointNet(nn.Module):
    """
    Spiking PointNet for 3D point cloud classification.

    Args:
        num_classes: Number of output classes (default 40 for ModelNet40).
        tau:         LIF membrane decay factor (default 0.25).
        V_th:        LIF spike threshold (default 0.5).
        dropout:     Dropout probability in classifier (default 0.3).
    """

    def __init__(
        self,
        num_classes: int = 40,
        tau: float = 0.25,
        V_th: float = 0.5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- Feature extraction ---
        # Input T-Net (3x3 spatial transform)
        self.tnet3 = TNet(k=3, tau=tau, V_th=V_th)

        # First MLP block: 3 -> 64 -> 64
        self.mlp1a = SharedMLP(3, 64, tau=tau, V_th=V_th)
        self.mlp1b = SharedMLP(64, 64, tau=tau, V_th=V_th)

        # Feature T-Net (64x64 feature transform)
        self.tnet64 = TNet(k=64, tau=tau, V_th=V_th)

        # Second MLP block: 64 -> 128 -> 1024
        self.mlp2a = SharedMLP(64, 128, tau=tau, V_th=V_th)
        self.mlp2b = SharedMLP(128, 1024, tau=tau, V_th=V_th)

        # --- Classifier ---
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.lif1 = make_lif_neuron(tau=tau, V_th=V_th, step_mode='s')

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.lif2 = make_lif_neuron(tau=tau, V_th=V_th, step_mode='s')
        self.dropout = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Kaiming normal, preserving TNet fc3 init."""
        tnet_module_ids = set()
        for tnet in [self.tnet3, self.tnet64]:
            tnet_module_ids.update(id(m) for m in tnet.modules())

        for m in self.modules():
            if id(m) in tnet_module_ids:
                continue
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Point cloud encoded for T timesteps, shape [B, T, N, 3].
               For T=1 (training), shape is [B, 1, N, 3].

        Returns:
            logits:    [B, num_classes]
            stn3_loss: Scalar regularization loss for input T-Net.
            stn64_loss: Scalar regularization loss for feature T-Net.
        """
        # Reset all LIF membrane potentials at start of each forward call
        functional.reset_net(self)

        B, T, N, C = x.shape

        # --- Input T-Net ---
        # Produces [B, T, 3, 3] transform matrices
        trans3 = self.tnet3(x)          # [B, T, 3, 3]
        x = apply_transform(x, trans3)  # [B, T, N, 3]

        # --- MLP block 1 ---
        x = self.mlp1a(x)   # [B, T, N, 64]
        x = self.mlp1b(x)   # [B, T, N, 64]

        # --- Feature T-Net ---
        trans64 = self.tnet64(x)         # [B, T, 64, 64]
        x = apply_transform(x, trans64)  # [B, T, N, 64]

        # --- MLP block 2 ---
        x = self.mlp2a(x)   # [B, T, N, 128]
        x = self.mlp2b(x)   # [B, T, N, 1024]

        # --- Global mean pooling over points ---
        # max-pool of binary spikes over N=1024 points gives all-1s for every sample
        # (P(any of 1024 points fires) ≈ 1 at 30% rate), which collapses BN downstream.
        # Mean pool gives the per-channel firing rate (~0.3), which varies across samples.
        x = x.mean(dim=2)     # [B, T, 1024]  — keep T for temporal stepping

        # --- Step classifier LIF neurons across T timesteps ---
        # lif1 and lif2 accumulate membrane potential across t=0..T-1,
        # which is what makes this a real SNN (trained-less paradigm depends on this).
        spike_sum = None
        for t in range(T):
            h = self.fc1(x[:, t, :])   # [B, 512]
            h = self.bn1(h)
            h = self.lif1(h)            # accumulates membrane across t
            h = self.fc2(h)             # [B, 256]
            h = self.bn2(h)
            h = self.lif2(h)            # accumulates membrane across t
            h = self.dropout(h)
            if spike_sum is None:
                spike_sum = h
            else:
                spike_sum = spike_sum + h

        # Average spikes across T
        x = spike_sum / T               # [B, 256]
        logits = self.fc3(x)            # [B, num_classes]

        # Compute T-Net regularization losses
        stn3_loss = stn_regularization_loss(trans3)
        stn64_loss = stn_regularization_loss(trans64)

        return logits, stn3_loss, stn64_loss
