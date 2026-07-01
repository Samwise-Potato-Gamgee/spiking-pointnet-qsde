"""
Queue-Driven Sampling Direct Encoding (Q-SDE).

Implements Algorithm 1 from the Spiking Point Transformer (AAAI 2025) paper.

Unlike direct encoding which repeats the same point cloud T times, Q-SDE
creates T diverse subsets of the point cloud using a FIFO queue mechanism:
  - Timestep 0: FPS-sample Ns points from all N points.
  - Timestep i: Drop the oldest Np points (dequeue), sample Np new points
    from the remaining unsampled set (FPS), and slide the window forward.

This gives each timestep a slightly different view of the point cloud,
effectively acting as a spatial data augmentation across time and reducing
redundancy in the SNN's temporal dimension.

Key parameters:
    Ns: Number of points per timestep (e.g. 512).
    T:  Number of timesteps (e.g. 4).
    Np: floor((N - Ns) / (T - 1)) -- points exchanged per step.

FPS is implemented manually to avoid torch_cluster dependency.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def fps_manual(pts: torch.Tensor, nsamples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) implemented in pure PyTorch.

    Iteratively selects the point farthest from the already-selected set.
    Runs on the same device as `pts`.

    Args:
        pts:      Input points [N, 3].
        nsamples: Number of points to select (must be <= N).

    Returns:
        Indices of selected points, shape [nsamples].

    Complexity: O(N * nsamples) -- acceptable for N<=2048, nsamples<=1024.
    """
    N = pts.shape[0]
    if nsamples >= N:
        # If requesting more than available, return all + wrap
        idx = torch.arange(N, device=pts.device)
        if nsamples > N:
            # Pad with random repeats
            extra = torch.randint(0, N, (nsamples - N,), device=pts.device)
            idx = torch.cat([idx, extra])
        return idx

    device = pts.device
    selected = torch.zeros(nsamples, dtype=torch.long, device=device)

    # Start from the point closest to origin (deterministic, reproducible)
    dist_to_origin = (pts ** 2).sum(dim=1)
    selected[0] = dist_to_origin.argmin()

    # Distance from each point to the nearest selected point
    distances = torch.full((N,), float('inf'), device=device)

    for i in range(1, nsamples):
        last = selected[i - 1]
        # Distance from all points to the last selected point
        diff = pts - pts[last].unsqueeze(0)   # [N, 3]
        dist = (diff ** 2).sum(dim=1)          # [N]
        # Update minimum distances
        distances = torch.minimum(distances, dist)
        # Select the farthest point
        selected[i] = distances.argmax()

    return selected


def _set_difference_indices(
    N: int,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Return indices of points NOT in mask (boolean mask of length N).

    Args:
        N:      Total number of points.
        mask:   Boolean tensor [N], True = already used.
        device: Target device.

    Returns:
        Indices of unmasked points [M] where M = N - mask.sum().
    """
    all_idx = torch.arange(N, device=device)
    return all_idx[~mask]


def qsde_single(
    pts: torch.Tensor,
    Ns: int,
    T: int,
) -> torch.Tensor:
    """
    Q-SDE for a single point cloud (Algorithm 1 from SPT paper).

    Args:
        pts: Input point cloud [N, 3].
        Ns:  Number of points per timestep output.
        T:   Number of timesteps.

    Returns:
        Multi-timestep point cloud [T, Ns, 3].
    """
    N = pts.shape[0]
    device = pts.device

    if T == 1:
        # Degenerate case: just FPS to Ns
        idx = fps_manual(pts, Ns)
        return pts[idx].unsqueeze(0)  # [1, Ns, 3]

    # Number of points exchanged per step
    Np = max(1, (N - Ns) // (T - 1))

    # Output container
    Pe = torch.zeros(T, Ns, 3, device=device, dtype=pts.dtype)

    # Timestep 0: FPS from all N points
    idx0 = fps_manual(pts, Ns)       # [Ns]
    Pe[0] = pts[idx0]                # [Ns, 3]

    # Track which original points have been "dequeued" (removed from pool)
    # We maintain a boolean mask: used_mask[i] = True means pts[i] was dequeued
    used_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # The current sliding window is stored as indices into the ORIGINAL pts array
    current_window_idx = idx0.clone()  # [Ns] indices into pts

    # Mark timestep 0 points as used so timestep 1 sees a reduced pool
    used_mask[idx0] = True

    for i in range(1, T):
        # Remaining = original pts NOT yet used (not dequeued)
        remaining_idx = _set_difference_indices(N, used_mask, device)

        if remaining_idx.numel() == 0:
            # Fallback: coverage -- reuse previous timestep
            Pe[i] = Pe[i - 1]
            continue

        # Dequeue: mark oldest Np points in current window as used
        dequeue_idx = current_window_idx[:Np]  # indices of points being dropped
        used_mask[dequeue_idx] = True

        # Keep last (Ns - Np) points from current window
        keep_idx = current_window_idx[Np:]     # [Ns - Np]

        # Sample Np new points from remaining pool via FPS
        remaining_pts = pts[remaining_idx]     # [M, 3]
        actual_Np = min(Np, remaining_pts.shape[0])

        if actual_Np > 0:
            new_local_idx = fps_manual(remaining_pts, actual_Np)   # [actual_Np]
            new_global_idx = remaining_idx[new_local_idx]           # global indices
        else:
            new_global_idx = torch.tensor([], dtype=torch.long, device=device)

        # Combine: keep + new
        if new_global_idx.numel() > 0:
            new_window_idx = torch.cat([keep_idx, new_global_idx], dim=0)
        else:
            new_window_idx = keep_idx

        # Pad if we ended up with fewer than Ns (shouldn't happen unless N is tiny)
        if new_window_idx.numel() < Ns:
            pad_count = Ns - new_window_idx.numel()
            pad_idx = torch.randint(0, N, (pad_count,), device=device)
            new_window_idx = torch.cat([new_window_idx, pad_idx], dim=0)
        elif new_window_idx.numel() > Ns:
            new_window_idx = new_window_idx[:Ns]

        Pe[i] = pts[new_window_idx]
        current_window_idx = new_window_idx

        # Dequeue from P (remove points used in PREVIOUS step)
        # (already done above via used_mask)

    return Pe  # [T, Ns, 3]


def qsde_encode(
    points_batch: torch.Tensor,
    Ns: int,
    T: int,
) -> torch.Tensor:
    """
    Batched Q-SDE encoding.

    Processes each sample in the batch independently (loop over batch).
    This is slower than a vectorized approach but correct and memory-efficient.

    Args:
        points_batch: Input point clouds [B, N, 3].
        Ns:           Number of output points per timestep.
        T:            Number of timesteps.

    Returns:
        Encoded tensor [B, T, Ns, 3].

    Example:
        >>> pts = torch.randn(4, 1024, 3)
        >>> enc = qsde_encode(pts, Ns=512, T=4)
        >>> enc.shape
        torch.Size([4, 4, 512, 3])
    """
    B, N, C = points_batch.shape
    device = points_batch.device
    dtype = points_batch.dtype

    out = torch.zeros(B, T, Ns, C, device=device, dtype=dtype)
    for b in range(B):
        out[b] = qsde_single(points_batch[b], Ns=Ns, T=T)

    return out
