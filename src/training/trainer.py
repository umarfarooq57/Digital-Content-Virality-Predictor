"""
Training engine: full training loop, evaluation, early stopping,
and incremental learning with Elastic Weight Consolidation (EWC).
"""
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BATCH_SIZE, EARLY_STOPPING_PATIENCE, EWC_LAMBDA,
    INCREMENTAL_EPOCHS, INCREMENTAL_LR, LEARNING_RATE,
    MODELS_DIR, NUM_EPOCHS, TRAIN_SPLIT, VAL_SPLIT,
    WEIGHT_DECAY,
)
from src.model.virality_model import CombinedLoss, ViralityPredictor
from src.utils.helpers import get_device, get_logger, seed_everything

logger = get_logger("trainer")


# ═══════════════════════════════════════════════════════════════════════
#  DATASET CREATION
# ═══════════════════════════════════════════════════════════════════════
def create_dataloaders(
    data: dict,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split preprocessed data into train/val/test DataLoaders."""
    n = len(data["y_regression"])
    indices = np.random.permutation(n)

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _make_loader(idx, shuffle=False):
        ds = TensorDataset(
            torch.tensor(data["text_emb"][idx]),
            torch.tensor(data["img_emb"][idx]),
            torch.tensor(data["tabular"][idx]),
            torch.tensor(data["y_regression"][idx]),
            torch.tensor(data["y_classification"][idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader = _make_loader(val_idx)
    test_loader = _make_loader(test_idx)

    logger.info(f"Splits: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(
    model: ViralityPredictor,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    target_encoder=None,
) -> dict:
    """Evaluate model and return metrics dict."""
    model.eval()
    all_reg_pred, all_reg_true = [], []
    all_cls_pred, all_cls_true = [], []
    total_loss = 0.0
    n_batches = 0

    for text, img, tab, y_reg, y_cls in loader:
        text, img, tab = text.to(device), img.to(device), tab.to(device)
        y_reg, y_cls = y_reg.to(device), y_cls.to(device)

        reg_pred, cls_pred = model(text, img, tab)
        loss, _ = criterion(reg_pred, y_reg, cls_pred, y_cls)
        total_loss += loss.item()
        n_batches += 1

        all_reg_pred.append(reg_pred.cpu().numpy())
        all_reg_true.append(y_reg.cpu().numpy())
        all_cls_pred.append(cls_pred.argmax(dim=1).cpu().numpy())
        all_cls_true.append(y_cls.cpu().numpy())

    reg_pred = np.concatenate(all_reg_pred)
    reg_true = np.concatenate(all_reg_true)
    cls_pred = np.concatenate(all_cls_pred)
    cls_true = np.concatenate(all_cls_true)

    # If target_encoder available, convert to original scale for MAE/RMSE
    if target_encoder is not None:
        reg_pred_orig = target_encoder.inverse_transform_regression(reg_pred)
        reg_true_orig = target_encoder.inverse_transform_regression(reg_true)
        mae = mean_absolute_error(reg_true_orig, reg_pred_orig)
        rmse = np.sqrt(mean_squared_error(reg_true_orig, reg_pred_orig))
    else:
        mae = mean_absolute_error(reg_true, reg_pred)
        rmse = np.sqrt(mean_squared_error(reg_true, reg_pred))

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "r2": r2_score(reg_true, reg_pred),
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy_score(cls_true, cls_pred),
        "f1_weighted": f1_score(cls_true, cls_pred, average="weighted"),
        "f1_macro": f1_score(cls_true, cls_pred, average="macro"),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════
def train_model(
    model: ViralityPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    target_encoder=None,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Full training loop with early stopping.
    Returns training history dict.
    """
    model = model.to(device)
    criterion = CombinedLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "val_r2": [], "val_mae": [], "val_rmse": [],
        "val_accuracy": [], "val_f1": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info(f"Training for {num_epochs} epochs on {device}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for text, img, tab, y_reg, y_cls in train_loader:
            text, img, tab = text.to(device), img.to(device), tab.to(device)
            y_reg, y_cls = y_reg.to(device), y_cls.to(device)

            optimizer.zero_grad()
            reg_pred, cls_pred = model(text, img, tab)
            loss, loss_dict = criterion(reg_pred, y_reg, cls_pred, y_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device, target_encoder)
        elapsed = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_weighted"])

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"R²: {val_metrics['r2']:.4f} | "
            f"MAE: {val_metrics['mae']:.0f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_weighted']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model weights")

    # Save
    if save_path is None:
        save_path = MODELS_DIR / "virality_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "history": history,
    }, save_path)
    logger.info(f"Model saved to {save_path}")

    return history


# ═══════════════════════════════════════════════════════════════════════
#  INCREMENTAL LEARNING (EWC)
# ═══════════════════════════════════════════════════════════════════════
class EWCTrainer:
    """
    Elastic Weight Consolidation for incremental learning.
    Prevents catastrophic forgetting when fine-tuning on new data.
    """

    def __init__(
        self,
        model: ViralityPredictor,
        device: torch.device,
        ewc_lambda: float = EWC_LAMBDA,
    ):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.fisher_dict: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def compute_fisher(self, data_loader: DataLoader) -> None:
        """Compute Fisher Information Matrix diagonal from current data."""
        self.model.eval()
        self.fisher_dict = {}

        for name, param in self.model.named_parameters():
            self.fisher_dict[name] = torch.zeros_like(param)

        n_samples = 0
        criterion = CombinedLoss().to(self.device)

        for text, img, tab, y_reg, y_cls in data_loader:
            text, img, tab = text.to(self.device), img.to(self.device), tab.to(self.device)
            y_reg, y_cls = y_reg.to(self.device), y_cls.to(self.device)

            self.model.zero_grad()
            reg_pred, cls_pred = self.model(text, img, tab)
            loss, _ = criterion(reg_pred, y_reg, cls_pred, y_cls)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_dict[name] += param.grad.data ** 2

            n_samples += text.size(0)

        for name in self.fisher_dict:
            self.fisher_dict[name] /= max(n_samples, 1)

        # Store optimal params
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        logger.info("Fisher Information computed")

    def ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                diff = param - self.optimal_params[name]
                penalty += (self.fisher_dict[name] * diff ** 2).sum()
        return self.ewc_lambda * penalty

    def incremental_train(
        self,
        new_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = INCREMENTAL_EPOCHS,
        lr: float = INCREMENTAL_LR,
        target_encoder=None,
    ) -> dict:
        """Fine-tune on new data with EWC regularization."""
        self.model.to(self.device)
        criterion = CombinedLoss().to(self.device)

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(criterion.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY,
        )

        history = {"train_loss": [], "val_loss": [], "val_r2": [], "val_accuracy": []}

        logger.info(f"Incremental training for {num_epochs} epochs (EWC λ={self.ewc_lambda})")

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for text, img, tab, y_reg, y_cls in new_loader:
                text, img, tab = text.to(self.device), img.to(self.device), tab.to(self.device)
                y_reg, y_cls = y_reg.to(self.device), y_cls.to(self.device)

                optimizer.zero_grad()
                reg_pred, cls_pred = self.model(text, img, tab)
                loss, _ = criterion(reg_pred, y_reg, cls_pred, y_cls)

                # Add EWC penalty
                ewc_loss = self.ewc_penalty()
                total_loss = loss + ewc_loss
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            val_metrics = evaluate(self.model, val_loader, criterion, self.device, target_encoder)

            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_r2"].append(val_metrics["r2"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                f"[Incremental] Epoch {epoch}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | Val R²: {val_metrics['r2']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

        # Save updated model
        save_path = MODELS_DIR / "virality_model_incremental.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "fisher_dict": self.fisher_dict,
            "optimal_params": self.optimal_params,
        }, save_path)
        logger.info(f"Incremental model saved to {save_path}")

        return history


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: Full pipeline
# ═══════════════════════════════════════════════════════════════════════
def run_full_training(
    data: dict,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> tuple[ViralityPredictor, dict, dict]:
    """
    Convenience function: create model → train → evaluate on test set.
    Returns (model, history, test_metrics).
    """
    seed_everything(42)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(data, batch_size)

    # Build model
    tabular_dim = data["tabular"].shape[1]
    model = ViralityPredictor(tabular_dim=tabular_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    target_encoder = data.get("target_encoder")

    # Train
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=lr,
        target_encoder=target_encoder,
    )

    # Test evaluation
    criterion = CombinedLoss().to(device)
    test_metrics = evaluate(model, test_loader, criterion, device, target_encoder)

    logger.info("═══ TEST RESULTS ═══")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return model, history, test_metrics
