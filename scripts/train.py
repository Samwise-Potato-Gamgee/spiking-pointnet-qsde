#!/usr/bin/env python3
"""
Training script for Spiking PointNet + Q-SDE.

Usage:
    # Baseline (direct encoding, T=1 train / T=4 infer):
    python scripts/train.py --config configs/baseline.yaml --run_name baseline_run1

    # Q-SDE variant:
    python scripts/train.py --config configs/qsde.yaml --run_name qsde_512_run1

    # Resume from checkpoint:
    python scripts/train.py --config configs/baseline.yaml --resume checkpoints/baseline_run1_last.pth

    # Reduce memory with gradient accumulation:
    python scripts/train.py --config configs/qsde.yaml --batch_size 8 --grad_accum 2
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from spk_pointnet.models.spiking_pointnet import SpikingPointNet
from spk_pointnet.data.modelnet40 import get_dataloaders
from spk_pointnet.training.trainer import Trainer
from spk_pointnet.training.losses import SpikingPointNetLoss
from spikingjelly.activation_based import functional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict) -> SpikingPointNet:
    """Construct SpikingPointNet from config."""
    model_cfg = config.get('model', {})
    model = SpikingPointNet(
        num_classes=model_cfg.get('num_classes', 40),
        tau=model_cfg.get('tau', 0.25),
        V_th=model_cfg.get('V_th', 0.5),
        dropout=model_cfg.get('dropout', 0.3),
    )
    # Set multi-step mode for SpikingJelly
    functional.set_step_mode(model, step_mode='m')
    return model


def build_optimizer(model: SpikingPointNet, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = config.get('optimizer', {})
    opt_type = opt_cfg.get('type', 'Adam')
    lr = opt_cfg.get('lr', 1e-3)
    weight_decay = opt_cfg.get('weight_decay', 1e-4)

    if opt_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Build learning rate scheduler from config."""
    sched_cfg = config.get('scheduler', {})
    sched_type = sched_cfg.get('type', 'CosineAnnealingLR')
    num_epochs = config.get('training', {}).get('epochs', 200)

    if sched_type == 'CosineAnnealingLR':
        T_max = sched_cfg.get('T_max', num_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=1e-5
        )
    elif sched_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get('step_size', 20),
            gamma=sched_cfg.get('gamma', 0.7),
        )
    elif sched_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_cfg.get('milestones', [100, 160]),
            gamma=sched_cfg.get('gamma', 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    return scheduler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Spiking PointNet (+ optional Q-SDE) on ModelNet40."
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to YAML config file (e.g., configs/baseline.yaml)."
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Path to checkpoint .pth file to resume training from."
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help="GPU index to use (default: 0). Use -1 for CPU."
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help="Override config batch_size."
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help="Override config epochs."
    )
    parser.add_argument(
        '--run_name', type=str, default=None,
        help="Name for this run (used in checkpoint and log filenames). "
             "Defaults to config file stem."
    )
    parser.add_argument(
        '--no_amp', action='store_true',
        help="Disable mixed-precision (AMP). Use for debugging only."
    )
    parser.add_argument(
        '--grad_accum', type=int, default=1,
        help="Gradient accumulation steps (default: 1). "
             "Effective batch = batch_size * grad_accum."
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        '--results_dir', type=str, default='results',
        help="Directory for CSV logs and plots (default: results/)."
    )
    parser.add_argument(
        '--ckpt_dir', type=str, default='checkpoints',
        help="Directory for model checkpoints (default: checkpoints/)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.no_amp:
        config.setdefault('training', {})['use_amp'] = False

    # Run name
    run_name = args.run_name or Path(args.config).stem

    # Seeds
    set_seed(args.seed)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available or --gpu -1)")

    # Extract config sections
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})

    num_epochs = train_cfg.get('epochs', 200)
    batch_size = train_cfg.get('batch_size', 16)
    use_amp = train_cfg.get('use_amp', True)
    label_smoothing = train_cfg.get('label_smoothing', 0.2)
    reg_weight = train_cfg.get('reg_weight', 0.001)
    num_workers = data_cfg.get('num_workers', 4)
    data_root = data_cfg.get('root', 'data/')
    aug_config = data_cfg.get('augmentation', {})
    num_points = model_cfg.get('num_points', 1024)

    print(f"\nRun: {run_name}")
    print(f"Config: {args.config}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, AMP: {use_amp}")
    print(f"Grad accum: {args.grad_accum} (effective batch: {batch_size * args.grad_accum})")

    # DataLoaders
    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(
        root=data_root,
        num_points=num_points,
        batch_size=batch_size,
        num_workers=num_workers,
        aug_config=aug_config,
        download=True,
        pin_memory=device.type == 'cuda',
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Loss
    criterion = SpikingPointNetLoss(
        label_smoothing=label_smoothing,
        reg_weight=reg_weight,
    )

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        run_name=run_name,
        use_amp=use_amp,
        grad_accum=args.grad_accum,
        use_wandb=args.use_wandb,
        results_dir=args.results_dir,
        ckpt_dir=args.ckpt_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.fit(num_epochs)


if __name__ == "__main__":
    main()
