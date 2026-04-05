"""
Feedforward Neural Network (FNN) for FraudShieldAI.

Architecture:
  Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear(1) → Sigmoid

Handles class imbalance via BCEWithLogitsLoss(pos_weight).
Includes:
  - Early stopping (monitors validation AUC-PR)
  - Cosine annealing LR scheduler
  - Sklearn-compatible wrapper (predict / predict_proba) for use in ensemble
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class FNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class FNNClassifier:
    """
    Sklearn-compatible wrapper around FNNModel.
    Supports predict_proba() for use in the ensemble.
    """

    def __init__(self, cfg: dict, input_dim: int):
        arch = cfg.get("architecture", {})
        train_cfg = cfg.get("training", {})

        self.hidden_layers: list[int] = arch.get("hidden_layers", [256, 128, 64])
        self.dropout: float = arch.get("dropout", 0.3)
        self.batch_norm: bool = arch.get("batch_norm", True)
        self.epochs: int = train_cfg.get("epochs", 50)
        self.batch_size: int = train_cfg.get("batch_size", 2048)
        self.lr: float = train_cfg.get("learning_rate", 0.001)
        self.weight_decay: float = train_cfg.get("weight_decay", 0.0001)
        self.pos_weight: float = train_cfg.get("pos_weight", 10.0)
        self.patience: int = train_cfg.get("early_stopping_patience", 7)
        self.input_dim = input_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[FNNModel] = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_auc_pr": []}

    def _build_model(self) -> FNNModel:
        return FNNModel(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FNNClassifier":
        # Hold out 10% for early stopping (stratified)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42
        )

        # Tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        self.model = self._build_model()
        pos_weight = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        best_val_auc = -np.inf
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            epoch_loss /= len(y_tr)

            # ---- Validate ----
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = average_precision_score(y_val, val_probs)

            self.history["train_loss"].append(epoch_loss)
            self.history["val_auc_pr"].append(val_auc)
            scheduler.step()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "[FNN] Epoch %3d/%d | Loss: %.4f | Val AUC-PR: %.4f | Best: %.4f",
                    epoch, self.epochs, epoch_loss, val_auc, best_val_auc,
                )

            if patience_counter >= self.patience:
                logger.info("[FNN] Early stopping at epoch %d", epoch)
                break

        # Restore best weights
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        logger.info("[FNN] Training complete. Best Val AUC-PR: %.4f", best_val_auc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [P(class=0), P(class=1)] array, sklearn-style."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
            },
            path,
        )
        logger.info("[FNN] Saved model → %s", path)

    @classmethod
    def load(cls, path: str | Path, cfg: dict) -> "FNNClassifier":
        checkpoint = torch.load(path, map_location="cpu")
        instance = cls(cfg=cfg, input_dim=checkpoint["input_dim"])
        instance.model = FNNModel(
            input_dim=checkpoint["input_dim"],
            hidden_layers=checkpoint["hidden_layers"],
            dropout=checkpoint["dropout"],
            batch_norm=checkpoint["batch_norm"],
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        return instance


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_fnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> FNNClassifier:
    """Instantiate and train the FNN classifier."""
    logger.info("[FNN] Input dim: %d | Device: %s", X_train.shape[1],
                "cuda" if torch.cuda.is_available() else "cpu")
    clf = FNNClassifier(cfg=cfg, input_dim=X_train.shape[1])
    clf.fit(X_train, y_train)
    return clf
