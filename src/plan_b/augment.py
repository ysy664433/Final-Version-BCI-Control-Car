import math

import numpy as np
import torch


class EEGAugment:
    def __init__(self, sample_rate=250, band_strength=0.2, channel_mask=0.2, noise_std=0.05):
        parameters = (sample_rate, band_strength, channel_mask, noise_std)
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("Augmentation parameters must be finite.")
        if sample_rate <= 0 or not 0 <= band_strength <= 0.5 or not 0 <= channel_mask <= 0.5 or not 0 <= noise_std <= 0.5:
            raise ValueError("Use sample_rate > 0 and augmentation strengths in [0, 0.5].")
        self.sample_rate = sample_rate
        self.band_strength = band_strength
        self.channel_mask = channel_mask
        self.noise_std = noise_std

    def __call__(self, eeg):
        if eeg.ndim != 3 or eeg.shape[1] < 2 or eeg.shape[2] < 2:
            raise ValueError("Expected EEG [batch, channels >= 2, time >= 2].")
        augmented = eeg.float().clone()
        if self.band_strength:
            spectrum = torch.fft.rfft(augmented, dim=-1)
            frequencies = torch.fft.rfftfreq(eeg.shape[-1], 1 / self.sample_rate, device=eeg.device)
            gains = torch.ones((len(eeg), 1, len(frequencies)), device=eeg.device)
            for low, high in ((4, 8), (8, 13), (13, 30), (30, 45)):
                selected = (frequencies >= low) & (frequencies < high)
                gain = 1 + self.band_strength * (2 * torch.rand((len(eeg), 1, 1), device=eeg.device) - 1)
                gains[:, :, selected] = gain
            augmented = torch.fft.irfft(spectrum * gains, n=eeg.shape[-1], dim=-1)
        if self.noise_std:
            amplitude = augmented.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            augmented = augmented + torch.randn_like(augmented) * amplitude * self.noise_std
        if self.channel_mask:
            retained = torch.rand((*eeg.shape[:2], 1), device=eeg.device) >= self.channel_mask
            empty = ~retained.any(dim=1).flatten()
            retained[empty, 0, 0] = True
            augmented = augmented * retained
        return augmented


def prepare_view(normalized_eeg, preprocessing, augmenter=None):
    mean = torch.as_tensor(preprocessing["mean"], dtype=torch.float32)
    scale = torch.as_tensor(preprocessing["scale"], dtype=torch.float32)
    raw = normalized_eeg.float() * scale + mean
    if augmenter is not None:
        raw = augmenter(raw)
    features = preprocessing["csp"].transform(raw.numpy().astype(np.float64))
    features = preprocessing["csp_scaler"].transform(features)
    return (raw - mean) / scale, torch.as_tensor(features, dtype=torch.float32)
