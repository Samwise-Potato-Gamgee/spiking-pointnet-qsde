#!/usr/bin/env python3
"""
Evaluation script for Spiking PointNet.

Loads a checkpoint, runs the test set, computes OA + mAcc + SynOps,
prints a metrics table, and saves a confusion matrix.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/baseline_best.pth
    python scripts/evaluate.py --checkpoint checkpoints/qsde_512_best.pth --T_infer 4
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from spk_pointnet.models.spiking_pointnet import SpikingPointNet
from spk_pointnet.data.modelnet40 import get_dataloaders, MODELNET40_CLASSES
from spk_pointnet.encoding.direct_encoding import direct_encode
from spk_pointnet.encoding.qsde import qsde_encode
from spk_pointnet.utils.metrics import overall_accuracy, mean_class_accuracy
from spk_pointnet.utils.synops import SynOpsCounter
from spk_pointnet.utils.visualize import plot_confusion_matrix
from spk_pointnet.training.losses import SpikingPointNetLoss
from spikingjelly.activation_based import functional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Spiking PointNet checkpoint on ModelNet40 test set."
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help="Path to checkpoint .pth file."
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="Path to config YAML. If not provided, uses config saved in checkpoint."
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help="GPU index (default: 0, -1 for CPU)."
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help="Evaluation batch size (default: 16)."
    )
    parser.add_argument(
        '--T_infer', type=int, default=None,
        help="Override inference timesteps."
    )
    parser.add_argument(
        '--results_dir', type=str, default='results',
        help="Directory to save confusion matrix (default: results/)."
    )
    parser.add_argument(
        '--no_synops', action='store_true',
        help="Skip SynOps computation (faster)."
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load checkpoint and return (state_dict, config)."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def main():
    args = parse_args()

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, device)

    # Get config (from checkpoint or file)
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in ckpt and ckpt['config'] is not None:
        config = ckpt['config']
        print("Using config from checkpoint.")
    else:
        raise ValueError(
            "No config found. Provide --config or ensure checkpoint has config saved."
        )

    model_cfg = config.get('model', {})
    enc_cfg = config.get('encoding', {})
    data_cfg = config.get('data', {})

    # Determine inference timesteps
    T_infer = args.T_infer or model_cfg.get('T_infer', 4)
    encoding_type = enc_cfg.get('type', 'direct')
    Ns = enc_cfg.get('Ns', model_cfg.get('num_points', 1024))
    num_points = model_cfg.get('num_points', 1024)
    data_root = data_cfg.get('root', 'data/')

    print(f"\nEncoding: {encoding_type}, T_infer={T_infer}, Ns={Ns}")

    # Build model
    model = SpikingPointNet(
        num_classes=model_cfg.get('num_classes', 40),
        tau=model_cfg.get('tau', 0.25),
        V_th=model_cfg.get('V_th', 0.5),
        dropout=model_cfg.get('dropout', 0.3),
    )
    functional.set_step_mode(model, step_mode='m')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    trained_epoch = ckpt.get('epoch', '?')
    best_oa_ckpt = ckpt.get('best_oa', None)
    print(f"Checkpoint epoch: {trained_epoch}, stored best OA: {best_oa_ckpt}")

    # DataLoader (test only)
    _, test_loader = get_dataloaders(
        root=data_root,
        num_points=num_points,
        batch_size=args.batch_size,
        num_workers=4,
        download=False,
        pin_memory=device.type == 'cuda',
    )

    criterion = SpikingPointNetLoss()

    # Encoding function
    def encode(pts):
        if encoding_type == 'direct':
            return direct_encode(pts, T_infer)
        elif encoding_type == 'qsde':
            return qsde_encode(pts, Ns=Ns, T=T_infer)
        else:
            raise ValueError(f"Unknown encoding: {encoding_type}")

    # Evaluation loop
    all_preds = []
    all_labels = []
    total_loss = 0.0

    print(f"\nRunning evaluation on {len(test_loader.dataset)} test samples...")

    # SynOps counter (run on a single batch only for efficiency)
    synops_report = None
    synops_done = False

    with torch.no_grad():
        for i, (points, labels) in enumerate(test_loader):
            points = points.to(device)
            labels = labels.to(device)

            encoded = encode(points)

            # Compute SynOps on first batch
            if not args.no_synops and not synops_done:
                with SynOpsCounter(model) as counter:
                    logits, stn3, stn64 = model(encoded)
                synops_report = counter.get_report()
                synops_done = True
            else:
                logits, stn3, stn64 = model(encoded)

            loss = criterion(logits, labels, stn3, stn64)
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Metrics
    oa = overall_accuracy(all_preds, all_labels)
    macc = mean_class_accuracy(all_preds, all_labels, num_classes=40)
    avg_loss = total_loss / len(test_loader)

    # Print results table
    run_name = Path(args.checkpoint).stem
    print(f"\n{'='*55}")
    print(f"  Evaluation Results: {run_name}")
    print(f"{'='*55}")
    print(f"  Test samples:     {len(all_labels)}")
    print(f"  Encoding:         {encoding_type} (T_infer={T_infer})")
    print(f"  Test Loss:        {avg_loss:.4f}")
    print(f"  Overall Accuracy: {oa * 100:.2f}%")
    print(f"  Mean Class Acc:   {macc * 100:.2f}%")
    if synops_report:
        print(f"  SynOps (1 batch): {synops_report['total_synops']:.3e}")
        print(f"  Mean spike rate:  {synops_report['mean_spike_rate']:.4f}")
        print(f"  AC energy:        {synops_report['ac_energy_pj']:.2f} pJ")
        print(f"  ANN MAC energy:   {synops_report['mac_energy_pj']:.2f} pJ")
        print(f"  Energy ratio:     {synops_report['energy_ratio']:.1f}x")
    print(f"{'='*55}")

    # Save confusion matrix
    class_names = MODELNET40_CLASSES
    plot_confusion_matrix(
        preds=all_preds,
        labels=all_labels,
        class_names=class_names,
        save_dir=args.results_dir,
        run_name=run_name,
    )

    # Per-class breakdown
    print(f"\n  Per-class accuracy (sorted by accuracy):")
    class_accs = []
    for c in range(40):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == c).mean()
            class_accs.append((class_names[c] if c < len(class_names) else str(c), acc, mask.sum()))

    class_accs.sort(key=lambda x: x[1])
    print(f"  {'Class':<20} {'Acc':>8} {'N':>6}")
    print(f"  {'-'*36}")
    for name, acc, n in class_accs:
        print(f"  {name:<20} {acc*100:>7.1f}%  {n:>5}")


if __name__ == "__main__":
    main()
