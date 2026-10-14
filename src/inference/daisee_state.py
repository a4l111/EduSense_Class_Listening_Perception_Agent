"""DAiSEE learning-state model utilities.

DAiSEE provides clip-level labels for boredom, engagement, confusion, and
frustration.  This module trains/loads a lightweight video classifier and maps
those signals into the project states used by the AI virtual teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DAISEE_LABEL_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]
PROJECT_STATES = ["focused", "confused", "tired", "distracted", "absent"]


@dataclass
class DAiSEEPrediction:
    state: str
    confidence: float
    levels: dict[str, int]
    probabilities: dict[str, list[float]]


class DAiSEEClipNet:
    """Mean-pooled frame encoder with four ordinal classification heads."""

    def __init__(self, num_levels: int = 4, pretrained: bool = False):
        import torch
        from torch import nn
        from torchvision import models

        super_cls = nn.Module

        class _Net(super_cls):
            def __init__(self):
                super().__init__()
                weights = (
                    models.MobileNet_V3_Small_Weights.DEFAULT
                    if pretrained
                    else None
                )
                backbone = models.mobilenet_v3_small(weights=weights)
                in_features = backbone.classifier[0].in_features
                backbone.classifier = nn.Identity()
                self.backbone = backbone
                self.heads = nn.ModuleDict({
                    name: nn.Linear(in_features, num_levels)
                    for name in DAISEE_LABEL_COLUMNS
                })

            def forward(self, x):
                batch, time, channels, height, width = x.shape
                x = x.reshape(batch * time, channels, height, width)
                features = self.backbone(x)
                features = features.reshape(batch, time, -1).mean(dim=1)
                return {name: head(features) for name, head in self.heads.items()}

        self.model = _Net()


def map_daisee_to_project_state(levels: dict[str, int], confidence_hint: float = 0.6) -> DAiSEEPrediction:
    boredom = int(levels.get("Boredom", 0))
    engagement = int(levels.get("Engagement", 0))
    confusion = int(levels.get("Confusion", 0))
    frustration = int(levels.get("Frustration", 0))

    if boredom >= 2:
        state = "tired"
        confidence = max(confidence_hint, 0.64 + 0.08 * (boredom - 2))
    elif engagement == 0 or frustration >= 2:
        state = "distracted"
        confidence = max(confidence_hint, 0.64 + 0.07 * max(1 - engagement, frustration - 1))
    elif confusion >= 2:
        state = "confused"
        confidence = max(confidence_hint, 0.68 + 0.08 * (confusion - 2))
    elif boredom >= 1 and engagement <= 1:
        state = "tired"
        confidence = max(confidence_hint, 0.60 + 0.05 * boredom)
    elif engagement <= 1:
        state = "distracted"
        confidence = max(confidence_hint, 0.60 + 0.05 * (1 - engagement))
    elif engagement >= 2 and boredom <= 1 and confusion <= 1:
        state = "focused"
        confidence = max(confidence_hint, 0.70 + 0.06 * (engagement - 2))
    else:
        state = "focused"
        confidence = max(confidence_hint, 0.58)

    return DAiSEEPrediction(
        state=state,
        confidence=float(min(confidence, 0.95)),
        levels=levels,
        probabilities={},
    )


class DAiSEEStateClassifier:
    """Runtime wrapper for a trained DAiSEE clip classifier."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        frames_per_clip: int = 8,
        image_size: int = 224,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        with Path(checkpoint_path).open("rb") as file:
            checkpoint = torch.load(file, map_location=self.device)

        self.net = DAiSEEClipNet(pretrained=False).model
        self.net.load_state_dict(checkpoint["model_state"])
        self.net.to(self.device)
        self.net.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _preprocess(self, frame: np.ndarray):
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb).to(self.device).float().permute(2, 0, 1) / 255.0
        return (tensor - self.mean) / self.std

    def predict_frames(self, frames: list[np.ndarray]) -> DAiSEEPrediction:
        torch = self.torch
        if not frames:
            return map_daisee_to_project_state({})

        sampled = frames[-self.frames_per_clip:]
        if len(sampled) < self.frames_per_clip:
            sampled = [sampled[0]] * (self.frames_per_clip - len(sampled)) + sampled

        clip = torch.stack([self._preprocess(frame) for frame in sampled], dim=0).unsqueeze(0)
        with torch.no_grad():
            outputs: dict[str, Any] = self.net(clip)
            probabilities = {
                name: torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
                for name, logits in outputs.items()
            }
            levels = {
                name: int(torch.argmax(logits, dim=1).item())
                for name, logits in outputs.items()
            }
            confidence_hint = max(max(values) for values in probabilities.values())

        prediction = map_daisee_to_project_state(levels, confidence_hint)
        prediction.probabilities = probabilities
        return prediction

    def predict_frame(self, frame: np.ndarray) -> DAiSEEPrediction:
        return self.predict_frames([frame])
