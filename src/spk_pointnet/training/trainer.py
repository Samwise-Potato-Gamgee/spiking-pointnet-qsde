"""
Training loop for Spiking PointNet.

Features:
  - train_epoch() / val_epoch() methods
  - AMP (mixed precision) with GradScaler
  - Gradient accumulation
  - tqdm progress bars
  - CSV metric logging to results/
  - Best-model checkpointing by val OA
  - Optional Weights & Biases logging
"""

import os
import csv
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikingjelly.activation_based import functional

from ..encoding.direct_encoding import direct_encode
from ..encoding.qsde import qsde_encode
from ..utils.metrics import overall_accuracy, mean_class_accuracy


class Trainer:
    """
    Trainer for Spiking PointNet.

    Handles the trained-less paradigm:
        - Baseline (direct encoding): train with T=1, evaluate with T=T_infer
        - Q-SDE: train with T=T (paper trains with T=4 directly)

    Args:
        model:         SpikingPointNet instance.
        criterion:     SpikingPointNetLoss instance.
        optimizer:     PyTorch optimizer.
        scheduler:     LR scheduler.
        train_loader:  Training DataLoader.
        val_loader:    Validation/test DataLoader.
        config:        Full configuration dict (from yaml).
        device:        Torch device.
        run_name:      Name for logging (used in checkpoint and CSV filenames).
        use_amp:       Enable mixed-precision training.
        grad_accum:    Gradient accumulation steps.
        use_wandb:     Enable Weights & Biases logging.
        results_dir:   Directory for CSV and checkpoint saving.
        ckpt_dir:      Directory for model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        run_name: str = "experiment",
        use_amp: bool = True,
        grad_accum: int = 1,
        use_wandb: bool = False,
        results_dir: str = "results",
        ckpt_dir: str = "checkpoints",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.run_name = run_name
        self.use_amp = use_amp
        self.grad_accum = grad_accum
        self.use_wandb = use_wandb
        self.results_dir = Path(results_dir)
        self.ckpt_dir = Path(ckpt_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler(enabled=use_amp)
        self.best_oa = 0.0
        self.start_epoch = 0

        # Extract encoding parameters from config
        enc_cfg = config.get('encoding', {})
        self.encoding_type = enc_cfg.get('type', 'direct')
        self.Ns = enc_cfg.get('Ns', config.get('model', {}).get('num_points', 1024))

        model_cfg = config.get('model', {})
        self.T_train = model_cfg.get('T', 1)
        self.T_infer = model_cfg.get('T_infer', 4)

        # For Q-SDE, T_train is from encoding config
        if self.encoding_type == 'qsde':
            self.T_train = enc_cfg.get('T', self.T_train)
            self.T_infer = enc_cfg.get('T_infer', self.T_infer)

        # CSV log file
        self.csv_path = self.results_dir / f"metrics_{run_name}.csv"
        self._init_csv()

        # WandB
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="spiking-pointnet", name=run_name, config=config)
                self._wandb = wandb
            except ImportError:
                print("WARNING: wandb not installed. Disabling wandb logging.")
                self.use_wandb = False

    def _init_csv(self):
        """Initialize CSV log file with header."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_oa',
                'val_loss', 'val_oa', 'val_macc',
                'lr', 'epoch_time_s',
            ])

    def _encode(self, points: torch.Tensor, T: int) -> torch.Tensor:
        """
        Encode point cloud according to encoding strategy.

        Args:
            points: [B, N, 3]
            T:      Number of timesteps.

        Returns:
            [B, T, N_out, 3]  (N_out = N for direct, Ns for Q-SDE)
        """
        if self.encoding_type == 'direct':
            return direct_encode(points, T)
        elif self.encoding_type == 'qsde':
            return qsde_encode(points, Ns=self.Ns, T=T)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dict with 'loss' and 'oa' keys.
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [train]",
            leave=False,
            dynamic_ncols=True,
        )

        self.optimizer.zero_grad()
        accum_count = 0

        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device, non_blocking=True)   # [B, N, 3]
            labels = labels.to(self.device, non_blocking=True)   # [B]

            # Encode for training
            encoded = self._encode(points, self.T_train)          # [B, T, N, 3]

            with autocast(enabled=self.use_amp):
                logits, stn3_loss, stn64_loss = self.model(encoded)
                loss = self.criterion(logits, labels, stn3_loss, stn64_loss)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()
            accum_count += 1

            if accum_count == self.grad_accum:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                accum_count = 0

            # Track metrics
            total_loss += loss.item() * self.grad_accum
            preds = logits.argmax(dim=1).detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())

            # Update progress bar
            current_oa = overall_accuracy(preds, labels.cpu())
            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}", oa=f"{current_oa:.4f}")

        # Handle remaining accumulated gradients
        if accum_count > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        oa = overall_accuracy(all_preds, all_labels)
        avg_loss = total_loss / len(self.train_loader)

        return {'loss': avg_loss, 'oa': oa}

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one validation epoch.

        Uses T_infer timesteps (trained-less paradigm for baseline).

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dict with 'loss', 'oa', 'macc' keys.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [val]",
            leave=False,
            dynamic_ncols=True,
        )

        for points, labels in pbar:
            points = points.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Encode for inference (use T_infer)
            encoded = self._encode(points, self.T_infer)

            with autocast(enabled=self.use_amp):
                logits, stn3_loss, stn64_loss = self.model(encoded)
                loss = self.criterion(logits, labels, stn3_loss, stn64_loss)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        oa = overall_accuracy(all_preds, all_labels)
        macc = mean_class_accuracy(all_preds, all_labels)
        avg_loss = total_loss / len(self.val_loader)

        return {'loss': avg_loss, 'oa': oa, 'macc': macc}

    def _log_csv(self, epoch: int, train_m: Dict, val_m: Dict, lr: float, epoch_time: float):
        """Append one row to the CSV log."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_m['loss']:.6f}",
                f"{train_m['oa']:.6f}",
                f"{val_m['loss']:.6f}",
                f"{val_m['oa']:.6f}",
                f"{val_m['macc']:.6f}",
                f"{lr:.8f}",
                f"{epoch_time:.1f}",
            ])

    def save_checkpoint(self, epoch: int, val_oa: float, is_best: bool):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_oa': self.best_oa,
            'config': self.config,
        }
        last_path = self.ckpt_dir / f"{self.run_name}_last.pth"
        torch.save(ckpt, last_path)

        if is_best:
            best_path = self.ckpt_dir / f"{self.run_name}_best.pth"
            torch.save(ckpt, best_path)
            print(f"  [*] New best OA: {val_oa:.4f} -> saved to {best_path}")

    def load_checkpoint(self, ckpt_path: str):
        """
        Load a checkpoint to resume training.

        Args:
            ckpt_path: Path to the checkpoint .pth file.
        """
        print(f"Loading checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_oa = ckpt.get('best_oa', 0.0)
        self.start_epoch = ckpt.get('epoch', 0)
        print(f"  Resumed from epoch {self.start_epoch}, best OA = {self.best_oa:.4f}")

    def fit(self, num_epochs: int):
        """
        Full training loop.

        Args:
            num_epochs: Total number of epochs to train.
        """
        print(f"\n{'='*60}")
        print(f"  Run: {self.run_name}")
        print(f"  Encoding: {self.encoding_type}  T_train={self.T_train}  T_infer={self.T_infer}")
        print(f"  AMP: {self.use_amp}  GradAccum: {self.grad_accum}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, num_epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.val_epoch(epoch)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            # Print summary
            print(
                f"Epoch {epoch + 1:>3}/{num_epochs}  "
                f"train loss={train_metrics['loss']:.4f}  oa={train_metrics['oa']:.4f}  |  "
                f"val loss={val_metrics['loss']:.4f}  oa={val_metrics['oa']:.4f}  "
                f"macc={val_metrics['macc']:.4f}  |  "
                f"lr={lr:.6f}  time={epoch_time:.0f}s"
            )

            # Log to CSV
            self._log_csv(epoch, train_metrics, val_metrics, lr, epoch_time)

            # WandB
            if self.use_wandb:
                self._wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/oa': train_metrics['oa'],
                    'val/loss': val_metrics['loss'],
                    'val/oa': val_metrics['oa'],
                    'val/macc': val_metrics['macc'],
                    'lr': lr,
                })

            # Checkpoint
            is_best = val_metrics['oa'] > self.best_oa
            if is_best:
                self.best_oa = val_metrics['oa']
            self.save_checkpoint(epoch, val_metrics['oa'], is_best)

        print(f"\nTraining complete. Best val OA: {self.best_oa:.4f}")
        if self.use_wandb:
            self._wandb.finish()
