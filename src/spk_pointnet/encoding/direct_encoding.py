"""
Direct Encoding for Spiking PointNet (baseline).

Converts a static point cloud to a multi-timestep tensor by repeating the
input T times.  This is the "naive" encoding used in the original Spiking
PointNet paper.

For T=1 (training time in the trained-less paradigm) this is a no-op expansion.
For T=4 (inference time) the same point cloud is fed four times.
"""

import torch


def direct_encode(points: torch.Tensor, T: int) -> torch.Tensor:
    """
    Repeat the input point cloud T times to create a multi-timestep tensor.

    Args:
        points: Input point cloud [B, N, 3] (already normalized).
        T:      Number of timesteps to repeat.

    Returns:
        Encoded tensor of shape [B, T, N, 3].

    Example:
        >>> pts = torch.randn(4, 1024, 3)
        >>> enc = direct_encode(pts, T=4)
        >>> enc.shape
        torch.Size([4, 4, 1024, 3])
    """
    # points: [B, N, 3] -> unsqueeze -> [B, 1, N, 3] -> repeat -> [B, T, N, 3]
    return points.unsqueeze(1).repeat(1, T, 1, 1)
