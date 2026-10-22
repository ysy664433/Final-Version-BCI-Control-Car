import torch
from torch import nn
from torch.nn import functional

from src.plan_a.model import EEGNetLight as Backbone
from src.plan_a.model import ShallowConvNetTeacher


class EEGNetLightPlanB(Backbone):
    def __init__(self, n_channels, n_times=1000, n_classes=2, csp_dim=6,
                 dropout=0.3, causal=False, contrastive=True, projection_dim=128):
        super().__init__(n_channels, n_times, n_classes, csp_dim, dropout,
                         enhance=False, dann=False, causal=causal)
        self.projector = nn.Sequential(
            nn.Linear(192, 192), nn.GELU(), nn.Linear(192, projection_dim),
        ) if contrastive else None

    def forward(self, eeg, csp_features, return_features=False):
        if not return_features:
            return super().forward(eeg, csp_features)
        logits, _, features = super().forward(eeg, csp_features, grl_coefficient=0.0)
        projected = self.projector(features) if self.projector is not None else features
        return logits, functional.normalize(projected.float(), dim=1)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    if features.ndim != 3 or features.shape[1] < 2:
        raise ValueError("Expected contrastive features [batch, views >= 2, dimensions].")
    if labels.ndim != 1 or len(labels) != features.shape[0] or temperature <= 0:
        raise ValueError("Invalid labels or contrastive temperature.")
    features = functional.normalize(features.float(), dim=-1)
    flattened = features.flatten(0, 1)
    labels = labels.repeat_interleave(features.shape[1])
    diagonal = torch.eye(len(flattened), dtype=torch.bool, device=features.device)
    same_class = labels[:, None] == labels[None, :]
    positives = same_class & ~diagonal
    valid = positives.any(1) & (~same_class).any(1)
    if not valid.any():
        return flattened.sum() * 0.0
    with torch.autocast(device_type=features.device.type, enabled=False):
        similarities = flattened @ flattened.T / temperature
    similarities = similarities.masked_fill(diagonal, -torch.inf)
    log_probabilities = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_log_probabilities = log_probabilities.masked_fill(~positives, 0.0)
    per_anchor = -positive_log_probabilities.sum(1) / positives.sum(1).clamp_min(1)
    return per_anchor[valid].mean()
