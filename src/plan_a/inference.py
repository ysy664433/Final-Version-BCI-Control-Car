import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .data import CausalPreprocessor
from .model import EEGNetLight


class Predictor:
    def __init__(self, run_directory, device="cpu"):
        self.directory = Path(run_directory)
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        self.preprocessing = joblib.load(self.directory / "preprocessing.joblib")
        self.device = torch.device(device)
        self.model = EEGNetLight(**self.config["model"]).to(self.device)
        checkpoint = torch.load(self.directory / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.no_grad()
    def predict_trials(self, eeg, batch_size=8):
        eeg = np.asarray(eeg, dtype=np.float32)
        expected = (len(self.config["channels"]), self.config["window_samples"])
        if eeg.ndim != 3 or eeg.shape[1:] != expected or len(eeg) == 0:
            raise ValueError(f"Expected nonempty trials [batch, {expected[0]}, {expected[1]}].")
        if batch_size < 1 or not np.isfinite(eeg).all():
            raise ValueError("Positive batch size and finite EEG values required.")
        probabilities = []
        for start in range(0, len(eeg), batch_size):
            batch = eeg[start:start + batch_size]
            features = self.preprocessing["csp"].transform(batch.astype(np.float64))
            features = self.preprocessing["csp_scaler"].transform(features)
            normalized = (batch - self.preprocessing["mean"]) / self.preprocessing["scale"]
            logits = self.model(
                torch.as_tensor(normalized, dtype=torch.float32, device=self.device),
                torch.as_tensor(features, dtype=torch.float32, device=self.device),
            )
            probabilities.append(logits.softmax(1).cpu().numpy())
        return np.concatenate(probabilities)


class OpenBMIStreamProcessor:
    def __init__(self, run_directory, input_channels, sample_rate=1000, device="cpu"):
        self.predictor = Predictor(run_directory, device)
        if self.predictor.config["mode"] != "online":
            raise ValueError("Streaming requires a model trained with --modes online.")
        if len(set(input_channels)) != len(input_channels):
            raise ValueError("Input channel names must be unique.")
        self.indices = [list(input_channels).index(name) for name in self.predictor.config["channels"]]
        self.input_channel_count = len(input_channels)
        self.filter = CausalPreprocessor(len(self.indices), sample_rate)
        self.window = self.predictor.config["window_samples"]
        self.buffer = np.empty((len(self.indices), 0), dtype=np.float64)
        self.processed_samples = 0
        self.pending = []

    def schedule_trial(self, onset_sample):
        if int(onset_sample) != onset_sample or onset_sample < self.filter.samples_seen:
            raise ValueError("Schedule an integer, zero-based cue sample before its chunk arrives.")
        start = (int(onset_sample) + self.filter.step - 1) // self.filter.step
        self.pending.append((start, start + self.window))
        self.pending.sort()

    def process_chunk(self, raw_chunk):
        raw_chunk = np.asarray(raw_chunk)
        if raw_chunk.ndim != 2 or raw_chunk.shape[0] != self.input_channel_count:
            raise ValueError("Raw chunk must match input_channels and have shape [channels, samples].")
        if not np.isfinite(raw_chunk).all():
            raise ValueError("EEG chunk contains nonfinite values.")
        filtered = self.filter.process(raw_chunk[self.indices])
        combined = np.concatenate((self.buffer, filtered), axis=1)
        buffer_start = self.processed_samples - self.buffer.shape[1]
        self.processed_samples += filtered.shape[1]
        results = []
        remaining = []
        for start, end in self.pending:
            if end > self.processed_samples:
                remaining.append((start, end))
                continue
            if start < buffer_start:
                raise RuntimeError("Trial data no longer available in buffer.")
            trial = combined[:, start - buffer_start:end - buffer_start]
            probabilities = self.predictor.predict_trials(trial[np.newaxis])[0]
            predicted = int(probabilities.argmax())
            results.append({
                "onset_output_sample": start, "label": self.predictor.config["labels"][predicted],
                "confidence": float(probabilities[predicted]), "probabilities": probabilities.tolist(),
            })
        self.pending = remaining
        self.buffer = combined[:, -self.window:]
        return results
