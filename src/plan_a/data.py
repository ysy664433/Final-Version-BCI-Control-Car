from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos


MOTOR_CHANNELS = (
    "FC1", "FC2", "FC5", "FC6", "C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6",
)


def filter_coefficients(sample_rate):
    if sample_rate <= 200:
        raise ValueError("Input sampling rate must exceed 200 Hz.")
    bandpass = butter(4, [0.5, 100], fs=sample_rate, btype="bandpass", output="sos")
    notch = tf2sos(*iirnotch(50, 30, fs=sample_rate))
    return np.vstack((bandpass, notch))


class CausalPreprocessor:
    def __init__(self, channels, sample_rate=1000, output_rate=250):
        self.step = int(sample_rate // output_rate)
        if self.step < 1 or sample_rate != self.step * output_rate:
            raise ValueError("Sampling rate must be an integer multiple of output rate.")
        self.sos = filter_coefficients(sample_rate)
        self.state = np.zeros((len(self.sos), channels, 2))
        self.samples_seen = 0

    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != self.state.shape[1]:
            raise ValueError("Expected chunk [channels, samples].")
        if chunk.shape[1] == 0:
            return chunk.copy()
        filtered, self.state = sosfilt(self.sos, chunk, axis=-1, zi=self.state)
        first = (-self.samples_seen) % self.step
        self.samples_seen += chunk.shape[1]
        return filtered[:, first::self.step]


def load_trials(data_root, subject, session, block_name, mode, channels=MOTOR_CHANNELS):
    path = Path(data_root) / f"session{session}" / f"s{subject}" / (
        f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )
    variable = f"EEG_MI_{block_name}"
    block = loadmat(path, simplify_cells=True, variable_names=[variable])[variable]
    channel_names = [str(name).strip() for name in block["chan"]]
    indices = [channel_names.index(name) for name in channels]
    sample_rate = int(block["fs"])
    step = sample_rate // 250
    if step < 1 or sample_rate != step * 250:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")
    signal = np.asarray(block["x"][:, indices], dtype=np.float64).T
    if mode == "online":
        processed = CausalPreprocessor(len(channels), sample_rate).process(signal)
    elif mode == "offline":
        processed = sosfiltfilt(filter_coefficients(sample_rate), signal, axis=-1)[:, ::step]
    else:
        raise ValueError("Mode must be offline or online.")
    onsets = np.asarray(block["t"], dtype=np.int64).ravel()
    labels = np.asarray(block["y_dec"], dtype=np.int64).ravel()
    class_map = {int(row[0]): str(row[1]).strip().lower() for row in block["class"]}
    if set(class_map.values()) != {"left", "right"}:
        raise ValueError(f"Unexpected OpenBMI class map: {class_map}")
    if len(onsets) != len(labels):
        raise ValueError("Trial onset and label counts differ.")
    trials = []
    for onset in onsets:
        if onset < 0 or onset + 4 * sample_rate > signal.shape[1]:
            raise ValueError("Trial extends beyond recording.")
        start = (int(onset) + step - 1) // step
        trial = processed[:, start:start + 1000]
        if trial.shape[1] != 1000:
            raise ValueError("Incomplete 4-second trial.")
        trials.append(trial)
    targets = np.array([0 if class_map[int(label)] == "left" else 1 for label in labels])
    return np.stack(trials).astype(np.float32), targets


def cached_trials(data_root, cache_root, subject, session, block_name, mode):
    import hashlib
    import json

    source = Path(data_root).resolve() / f"session{session}" / f"s{subject}" / (
        f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )
    metadata = {
        "version": 2, "source": str(source), "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns, "mode": mode,
        "block": block_name, "channels": MOTOR_CHANNELS,
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    cache_path = Path(cache_root) / f"{key}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["eeg"], cached["labels"]
    eeg, labels = load_trials(data_root, subject, session, block_name, mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, eeg=eeg, labels=labels)
    temporary.replace(cache_path)
    return eeg, labels
