"""
LSTM model for FraudShieldAI.

Treats each cardholder's transaction history as a time-ordered sequence.
For each target transaction, the model looks at the previous N transactions
by the same card (padding with zeros if fewer exist) and predicts fraud.

Architecture:
  Sequence(N, features) → LSTM(hidden, layers) → last hidden state
                        → Linear → Sigmoid

Sklearn-compatible wrapper: predict / predict_proba.

Sequence construction:
  Because the Kaggle dataset provides individual rows (not pre-grouped),
  we build per-card sequences at training/inference time using
  build_sequences(), which sorts by card + timestamp and extracts
  a sliding window of length `seq_len` ending at each transaction.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    X: np.ndarray,
    card_indices: np.ndarray,
    seq_len: int = 10,
) -> np.ndarray:
    """
    Build fixed-length look-back sequences for each transaction.

    For transaction i belonging to card c, the sequence is the
    `seq_len` transactions immediately preceding i (same card),
    zero-padded on the left if fewer than seq_len prior transactions exist.

    Args:
        X:            Feature matrix (n_samples, n_features). Rows must be
                      sorted by (card, time) before calling this function.
        card_indices: Integer array of shape (n_samples,) encoding which
                      card each row belongs to (e.g. hashed cc_num).
        seq_len:      Number of prior transactions to include.

    Returns:
        sequences: np.ndarray of shape (n_samples, seq_len, n_features)
    """
    n, n_feat = X.shape
    sequences = np.zeros((n, seq_len, n_feat), dtype=np.float32)

    # Build a mapping card → sorted row indices
    unique_cards = np.unique(card_indices)
    card_to_rows: dict[int, list[int]] = {c: [] for c in unique_cards}
    for i, c in enumerate(card_indices):
        card_to_rows[c].append(i)

    for c, rows in card_to_rows.items():
        rows = sorted(rows)             # already sorted if X is sorted
        for pos, row_idx in enumerate(rows):
            start = max(0, pos - seq_len)
            history = rows[start:pos]   # up to seq_len prior rows
            if len(history) == 0:
                continue
            seq_data = X[history]       # shape (<=seq_len, n_feat)
            pad = seq_len - len(seq_data)
            sequences[row_idx] = np.vstack(
                [np.zeros((pad, n_feat), dtype=np.float32), seq_data]
            )

    return sequences


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, (h_n, _) = self.lstm(x)
        # Use the last time-step output
        last_out = out[:, -1, :]        # (batch, hidden_size)
        last_out = self.dropout(last_out)
        return self.fc(last_out).squeeze(1)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class LSTMClassifier:
    """
    Sklearn-compatible wrapper around LSTMModel.
    Expects 3-D input: (n_samples, seq_len, n_features).
    """

    def __init__(self, cfg: dict, input_dim: int, seq_len: int):
        arch = cfg.get("architecture", {})
        train_cfg = cfg.get("training", {})

        self.hidden_size: int = arch.get("hidden_size", 128)
        self.num_layers: int = arch.get("num_layers", 2)
        self.dropout: float = arch.get("dropout", 0.3)
        self.bidirectional: bool = arch.get("bidirectional", False)
        self.epochs: int = train_cfg.get("epochs", 40)
        self.batch_size: int = train_cfg.get("batch_size", 1024)
        self.lr: float = train_cfg.get("learning_rate", 0.001)
        self.weight_decay: float = train_cfg.get("weight_decay", 0.0001)
        self.pos_weight: float = train_cfg.get("pos_weight", 10.0)
        self.patience: int = train_cfg.get("early_stopping_patience", 7)
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMModel] = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_auc_pr": []}

    def _build_model(self) -> LSTMModel:
        return LSTMModel(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        """
        Args:
            X_seq: shape (n_samples, seq_len, n_features)
            y:     shape (n_samples,)
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_seq, y, test_size=0.1, stratify=y, random_state=42
        )

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_np = y_val

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
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
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            epoch_loss /= len(y_tr)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = average_precision_score(y_val_np, val_probs)

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
                    "[LSTM] Epoch %3d/%d | Loss: %.4f | Val AUC-PR: %.4f | Best: %.4f",
                    epoch, self.epochs, epoch_loss, val_auc, best_val_auc,
                )

            if patience_counter >= self.patience:
                logger.info("[LSTM] Early stopping at epoch %d", epoch)
                break

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        logger.info("[LSTM] Training complete. Best Val AUC-PR: %.4f", best_val_auc)
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Return [P(0), P(1)] array. Input shape: (n, seq_len, features)."""
        self.model.eval()
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_seq)[:, 1] >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
            },
            path,
        )
        logger.info("[LSTM] Saved model → %s", path)

    @classmethod
    def load(cls, path: str | Path, cfg: dict) -> "LSTMClassifier":
        checkpoint = torch.load(path, map_location="cpu")
        instance = cls(
            cfg=cfg,
            input_dim=checkpoint["input_dim"],
            seq_len=checkpoint["seq_len"],
        )
        instance.model = LSTMModel(
            input_dim=checkpoint["input_dim"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            bidirectional=checkpoint["bidirectional"],
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        return instance


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_lstm(
    X_seq: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> LSTMClassifier:
    """
    Train the LSTM classifier on pre-built sequences.

    Args:
        X_seq:   shape (n_samples, seq_len, n_features)
        y_train: shape (n_samples,)
        cfg:     models.yaml['lstm'] section
    """
    seq_len = X_seq.shape[1]
    input_dim = X_seq.shape[2]
    logger.info(
        "[LSTM] seq_len=%d | input_dim=%d | device=%s",
        seq_len, input_dim, "cuda" if torch.cuda.is_available() else "cpu",
    )
    clf = LSTMClassifier(cfg=cfg, input_dim=input_dim, seq_len=seq_len)
    clf.fit(X_seq, y_train)
    return clf
