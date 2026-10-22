import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.plan_a.data import CausalPreprocessor
from src.plan_a.inference import OpenBMIStreamProcessor as BaseStreamProcessor
from src.plan_a.inference import Predictor as BasePredictor
from .model import EEGNetLightPlanB


class Predictor(BasePredictor):
    def __init__(self, run_directory, device="cpu"):
        self.directory = Path(run_directory)
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        if self.config.get("framework") != "plan_b":
            raise ValueError("Plan B Predictor requires a Plan B checkpoint.")
        self.preprocessing = joblib.load(self.directory / "preprocessing.joblib")
        self.device = torch.device(device)
        self.model = EEGNetLightPlanB(**self.config["model"]).to(self.device)
        checkpoint = torch.load(self.directory / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()


class OpenBMIStreamProcessor(BaseStreamProcessor):
    def __init__(self, run_directory, input_channels, sample_rate=1000, device="cpu"):
        self.predictor = Predictor(run_directory, device)
        if self.predictor.config["mode"] != "online":
            raise ValueError("Streaming requires an online Plan B model.")
        if len(set(input_channels)) != len(input_channels):
            raise ValueError("Input channel names must be unique.")
        self.indices = [list(input_channels).index(name) for name in self.predictor.config["channels"]]
        self.input_channel_count = len(input_channels)
        self.filter = CausalPreprocessor(len(self.indices), sample_rate)
        self.window = self.predictor.config["window_samples"]
        self.buffer = np.empty((len(self.indices), 0), dtype=np.float64)
        self.processed_samples = 0
        self.pending = []
