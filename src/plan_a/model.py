import math

import torch
from torch import nn


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = coefficient
        return features.clone()

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class WeakFeatureEnhance(nn.Module):
    def __init__(self, channels, causal=False):
        super().__init__()
        self.causal = causal
        self.attention = nn.Conv1d(channels, channels, 5, groups=channels)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, eeg):
        padding = (4, 0) if self.causal else (2, 2)
        weights = self.attention(nn.functional.pad(eeg, padding))
        return eeg * (2.0 * torch.sigmoid(weights))


class EEGNetLight(nn.Module):
    def __init__(self, n_channels, n_times=1000, n_classes=2, csp_dim=6,
                 dropout=0.3, enhance=True, dann=True, causal=False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.enhance = WeakFeatureEnhance(n_channels, causal) if enhance else nn.Identity()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.proj = nn.Linear(64, 128)
        positions = torch.arange(n_times).unsqueeze(1).float()
        frequencies = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000) / 128))
        encoding = torch.zeros(1, n_times, 128)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", encoding)
        layer = nn.TransformerEncoderLayer(
            128, 4, 256, dropout, activation="gelu", batch_first=True,
        )
        self.trm = nn.TransformerEncoder(layer, 2)
        self.csp_fc = nn.Sequential(nn.Linear(csp_dim, 64), nn.GELU(), nn.Dropout(0.4))
        self.head = nn.Sequential(
            nn.LayerNorm(192), nn.Linear(192, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.5), nn.Linear(128, n_classes),
        )
        self.domain = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2),
        ) if dann else None

    def forward(self, eeg, csp_features, grl_coefficient=None):
        temporal = self.temporal(self.enhance(eeg)).transpose(1, 2)
        temporal = self.proj(temporal) + self.position_encoding[:, :temporal.shape[1]]
        temporal = self.trm(temporal).mean(dim=1)
        features = torch.cat((temporal, self.csp_fc(csp_features)), dim=1)
        logits = self.head(features)
        if grl_coefficient is None:
            return logits
        domain_logits = None
        if self.domain is not None:
            domain_logits = self.domain(GradientReverseLayer.apply(features, grl_coefficient))
        return logits, domain_logits, features


class ShallowConvNetTeacher(nn.Module):
    def __init__(self, n_channels, n_times=1000, n_classes=2):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 40, (1, 25))
        self.conv_spat = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), (1, 15))
        self.dropout = nn.Dropout(0.5)
        self.clf = nn.Linear(40 * ((n_times - 25 + 1 - 75) // 15 + 1), n_classes)

    def forward(self, eeg):
        features = self.bn(self.conv_spat(self.conv_time(eeg.unsqueeze(1))))
        features = torch.log(self.pool(features.square()).clamp_min(1e-6))
        return self.clf(self.dropout(features).flatten(1))
