"""YOLO-feature + LSTM learning-state model for DAiSEE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.daisee_state import DAISEE_LABEL_COLUMNS, DAiSEEPrediction, map_daisee_to_project_state


FEATURE_DIM = 48


@dataclass
class LSTMPrediction:
    state: str
    confidence: float
    levels: dict[str, int]
    probabilities: dict[str, list[float]]


class DAiSEEFeatureLSTM:
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_levels: int = 4,
        dropout: float = 0.25,
        pooling: str = "mean",
    ):
        import torch
        from torch import nn

        super_cls = nn.Module

        class _Net(super_cls):
            def __init__(self):
                super().__init__()
                self.pooling = pooling
                self.input_norm = nn.LayerNorm(feature_dim)
                self.lstm = nn.LSTM(
                    feature_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=True,
                )
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                self.dropout = nn.Dropout(dropout)
                self.heads = nn.ModuleDict({
                    name: nn.Linear(hidden_dim * 2, num_levels)
                    for name in DAISEE_LABEL_COLUMNS
                })

            def forward(self, x):
                x = self.input_norm(x)
                output, _ = self.lstm(x)
                if self.pooling == "attention":
                    weights = self.attention(output).softmax(dim=1)
                    pooled = (output * weights).sum(dim=1)
                else:
                    pooled = output.mean(dim=1)
                pooled = self.dropout(pooled)
                return {name: head(pooled) for name, head in self.heads.items()}

        self.model = _Net()


class DAiSEELSTMClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        with Path(checkpoint_path).open("rb") as file:
            checkpoint = torch.load(file, map_location=self.device)
        self.feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        self.sequence_length = int(checkpoint.get("sequence_length", 16))
        self.label_mode = str(checkpoint.get("label_mode", "four"))
        self.num_levels = int(checkpoint.get("num_levels", 4))
        self.net = DAiSEEFeatureLSTM(
            feature_dim=self.feature_dim,
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
            num_layers=int(checkpoint.get("num_layers", 2)),
            num_levels=self.num_levels,
            dropout=float(checkpoint.get("dropout", 0.25)),
            pooling=str(checkpoint.get("pooling", "mean")),
        ).model
        self.net.load_state_dict(checkpoint["model_state"])
        self.net.to(self.device)
        self.net.eval()

    def predict_features(self, features: np.ndarray) -> DAiSEEPrediction:
        torch = self.torch
        if features.ndim == 2:
            features = features[None, ...]
        tensor = torch.from_numpy(features.astype(np.float32)).to(self.device)
        with torch.no_grad():
            outputs: dict[str, Any] = self.net(tensor)
            probabilities = {
                name: torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
                for name, logits in outputs.items()
            }
            levels = {
                name: self._expand_level(int(torch.argmax(logits, dim=1).item()))
                for name, logits in outputs.items()
            }
            confidence_hint = max(max(values) for values in probabilities.values())

        prediction = map_daisee_to_project_state(levels, confidence_hint)
        prediction.probabilities = probabilities
        return prediction

    def _expand_level(self, level: int) -> int:
        if self.label_mode == "binary":
            return 3 if level >= 1 else 0
        if self.label_mode == "ternary":
            return [0, 1, 3][max(0, min(level, 2))]
        return level
