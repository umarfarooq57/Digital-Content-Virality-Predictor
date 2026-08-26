"""
Multi-Input PyTorch Model for Virality Prediction.

Architecture:
  ┌─────────┐   ┌─────────┐   ┌──────────┐
  │  Text   │   │  Image  │   │ Tabular  │
  │ Encoder │   │ Encoder │   │ Encoder  │
  └────┬────┘   └────┬────┘   └────┬─────┘
       │             │             │
       └──────┬──────┘─────────────┘
              │  Concatenate
        ┌─────┴─────┐
        │  Fusion   │
        │  Network  │
        └─────┬─────┘
              │
     ┌────────┴────────┐
     │                 │
  ┌──┴───┐        ┌───┴────┐
  │ Reg  │        │  Cls   │
  │ Head │        │  Head  │
  └──────┘        └────────┘
  (views)       (Low/Med/High)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM, HIDDEN_DIM,
    DROPOUT, NUM_VIRALITY_CLASSES,
)


class TextEncoder(nn.Module):
    """MLP encoder for text embeddings."""

    def __init__(self, input_dim: int = TEXT_EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """MLP encoder for image embeddings."""

    def __init__(self, input_dim: int = IMAGE_EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularEncoder(nn.Module):
    """MLP encoder for tabular features."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionNetwork(nn.Module):
    """Fuses multi-modal features and produces shared representation."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft attention weighting
        attn_weights = torch.sigmoid(self.attention(x))
        x = x * attn_weights
        return self.fusion(x)


class ViralityPredictor(nn.Module):
    """
    Multi-input model: text + image + tabular → views (regression) + virality class.
    """

    def __init__(
        self,
        text_dim: int = TEXT_EMBEDDING_DIM,
        image_dim: int = IMAGE_EMBEDDING_DIM,
        tabular_dim: int = 24,   # will be set dynamically
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_VIRALITY_CLASSES,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(text_dim, hidden_dim)
        self.image_encoder = ImageEncoder(image_dim, hidden_dim)
        self.tabular_encoder = TabularEncoder(tabular_dim, hidden_dim)

        # Each encoder outputs hidden_dim//2, so fusion input = 3 * (hidden_dim//2)
        fusion_input = 3 * (hidden_dim // 2)
        self.fusion = FusionNetwork(fusion_input, hidden_dim)

        # Task-specific heads
        fusion_out = hidden_dim // 2

        # Regression head (predict log-scaled views)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_out, fusion_out // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(fusion_out // 2, 1),
        )

        # Classification head (predict virality class)
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_out, fusion_out // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(fusion_out // 2, num_classes),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        img_emb: torch.Tensor,
        tab_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            reg_out: (batch, 1) — predicted scaled views
            cls_out: (batch, num_classes) — logits for virality class
        """
        t = self.text_encoder(text_emb)
        i = self.image_encoder(img_emb)
        tab = self.tabular_encoder(tab_features)

        fused = torch.cat([t, i, tab], dim=1)
        shared = self.fusion(fused)

        reg_out = self.regression_head(shared).squeeze(-1)
        cls_out = self.classification_head(shared)

        return reg_out, cls_out

    def get_feature_importance(
        self,
        text_emb: torch.Tensor,
        img_emb: torch.Tensor,
        tab_features: torch.Tensor,
    ) -> dict[str, float]:
        """
        Estimate feature importance via gradient-based attribution.
        """
        text_emb = text_emb.clone().detach().requires_grad_(True)
        img_emb = img_emb.clone().detach().requires_grad_(True)
        tab_features = tab_features.clone().detach().requires_grad_(True)

        reg_out, cls_out = self.forward(text_emb, img_emb, tab_features)

        # Use regression output for attribution
        reg_out.sum().backward()

        text_importance = text_emb.grad.abs().mean().item()
        image_importance = img_emb.grad.abs().mean().item()
        tabular_importance = tab_features.grad.abs().mean().item()

        total = text_importance + image_importance + tabular_importance + 1e-8
        return {
            "Text Features": text_importance / total,
            "Image Features": image_importance / total,
            "Tabular Features": tabular_importance / total,
        }


# ═══════════════════════════════════════════════════════════════════════
#  COMBINED LOSS
# ═══════════════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    """
    Multi-task loss: weighted sum of MSE (regression) + CrossEntropy (classification).
    Uses learnable task weights (uncertainty weighting).
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance for each task (Kendall et al., 2018)
        self.log_var_reg = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        reg_pred: torch.Tensor,
        reg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mse_loss = self.mse(reg_pred, reg_target)
        ce_loss = self.ce(cls_pred, cls_target)

        # Homoscedastic uncertainty weighting
        precision_reg = torch.exp(-self.log_var_reg)
        precision_cls = torch.exp(-self.log_var_cls)

        total = (
            precision_reg * mse_loss + self.log_var_reg
            + precision_cls * ce_loss + self.log_var_cls
        )

        loss_dict = {
            "total": total.item(),
            "mse": mse_loss.item(),
            "ce": ce_loss.item(),
            "w_reg": precision_reg.item(),
            "w_cls": precision_cls.item(),
        }
        return total, loss_dict
