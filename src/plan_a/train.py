import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import MOTOR_CHANNELS, cached_trials
from .model import EEGNetLight, ShallowConvNetTeacher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "baseline": (False, False), "enhance": (True, False),
    "dann": (False, True), "plan_a": (True, True),
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def supplied_groups(path, subjects):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = {int(row["subject_id"]): row for row in csv.DictReader(handle)}
    missing = set(subjects) - rows.keys()
    if missing:
        raise ValueError(f"Missing subjects in supplied group CSV: {sorted(missing)}")
    accuracies = {subject: float(rows[subject]["mean_accuracy"]) for subject in subjects}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in accuracies.values()):
        raise ValueError("Group CSV accuracy values must be finite and in [0, 1].")
    return {subject: int(value < 0.70) for subject, value in accuracies.items()}


def training_group(eeg, labels, seed, threshold):
    from mne.decoding import CSP
    from scipy.signal import butter, sosfiltfilt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    band = butter(4, [8, 30], fs=250, btype="bandpass", output="sos")
    filtered = sosfiltfilt(band, eeg.astype(np.float64), axis=-1)
    pipeline = make_pipeline(
        CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    accuracy = float(cross_val_score(pipeline, filtered, labels, cv=folds).mean())
    return int(accuracy < threshold), accuracy


def prepare(args, mode):
    from mne.decoding import CSP
    from sklearn.preprocessing import StandardScaler

    supplied = supplied_groups(args.groups_csv, args.subjects)
    train_eeg, train_labels, train_subjects = [], [], []
    val_eeg, val_labels, val_subjects = [], [], []
    groups = {}
    for subject in args.subjects:
        print(f"[{mode}] Loading subject {subject}", flush=True)
        eeg, labels = cached_trials(args.data_root, args.cache_root, subject, 1, "train", mode)
        training_domain, accuracy = training_group(eeg, labels, args.group_seed, args.threshold)
        groups[subject] = {
            "supplied": supplied[subject], "training": training_domain,
            "training_csp_lda_accuracy": accuracy,
        }
        train_eeg.append(eeg)
        train_labels.append(labels)
        train_subjects.extend([subject] * len(labels))
        eeg, labels = cached_trials(args.data_root, args.cache_root, subject, 1, "test", mode)
        val_eeg.append(eeg)
        val_labels.append(labels)
        val_subjects.extend([subject] * len(labels))
    train_eeg, val_eeg = np.concatenate(train_eeg), np.concatenate(val_eeg)
    train_labels, val_labels = np.concatenate(train_labels), np.concatenate(val_labels)
    train_subjects, val_subjects = np.asarray(train_subjects), np.asarray(val_subjects)
    train_domains = np.array([groups[int(subject)][args.domain_source] for subject in train_subjects])
    val_domains = np.array([groups[int(subject)][args.domain_source] for subject in val_subjects])
    if set(train_domains) != {0, 1}:
        raise ValueError("DANN needs both normal and low-performing subjects. Select both groups.")
    if set(zip(train_domains.tolist(), train_labels.tolist())) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Each domain must contain both left and right labels.")
    csp = CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)
    csp_train = csp.fit_transform(train_eeg.astype(np.float64), train_labels)
    csp_val = csp.transform(val_eeg.astype(np.float64))
    csp_scaler = StandardScaler().fit(csp_train)
    mean = train_eeg.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    scale = train_eeg.std(axis=(0, 2), keepdims=True, dtype=np.float64).clip(1e-8)
    train_eeg = ((train_eeg - mean) / scale).astype(np.float32)
    val_eeg = ((val_eeg - mean) / scale).astype(np.float32)
    preprocessing = {
        "csp": csp, "csp_scaler": csp_scaler, "mean": mean, "scale": scale,
        "channels": MOTOR_CHANNELS, "groups": groups,
    }
    source_metadata = []
    for subject in args.subjects:
        source = args.data_root / "session1" / f"s{subject}" / f"sess01_subj{subject:02d}_EEG_MI.mat"
        source_metadata.append((subject, source.stat().st_size, source.stat().st_mtime_ns))
    preprocessing["data_fingerprint"] = hashlib.sha256(
        json.dumps({"sources": source_metadata, "groups": groups, "preprocessing_version": 2},
                   sort_keys=True).encode()
    ).hexdigest()
    def dataset(eeg, labels, csp_features, domains, subjects):
        return TensorDataset(
            torch.from_numpy(eeg), torch.as_tensor(labels, dtype=torch.long),
            torch.as_tensor(csp_scaler.transform(csp_features), dtype=torch.float32),
            torch.as_tensor(domains, dtype=torch.long), torch.as_tensor(subjects, dtype=torch.long),
        )
    return (
        dataset(train_eeg, train_labels, csp_train, train_domains, train_subjects),
        dataset(val_eeg, val_labels, csp_val, val_domains, val_subjects), preprocessing,
    )


def subject_metrics(labels, predictions, subjects, domains):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

    rows = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        rows.append({
            "subject": int(subject), "domain": int(domains[selected][0]),
            "accuracy": float(accuracy_score(labels[selected], predictions[selected])),
            "balanced_accuracy": float(balanced_accuracy_score(labels[selected], predictions[selected])),
            "kappa": float(cohen_kappa_score(labels[selected], predictions[selected])),
            "trials": int(selected.sum()),
        })
    return rows


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    labels, predictions, subjects, domains = [], [], [], []
    for eeg, targets, csp_features, domain, subject in loader:
        logits = model(eeg.to(device), csp_features.to(device))
        labels.extend(targets.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        domains.extend(domain.tolist())
        subjects.extend(subject.tolist())
    rows = subject_metrics(np.array(labels), np.array(predictions), np.array(subjects), np.array(domains))
    target_rows = [row for row in rows if row["domain"] == 1]
    score = float(np.mean([row["balanced_accuracy"] for row in target_rows]))
    return score, rows


def checkpoint_save(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_variant(args, mode, variant, seed, train_data, val_data, preprocessing):
    output = args.output_root / mode / variant / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    if (output / "config.json").exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {output}. Use --resume or another --output-root.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check server PyTorch installation.")
    set_seed(seed)
    enhance, dann = VARIANTS[variant]
    model_config = {
        "n_channels": len(MOTOR_CHANNELS), "n_times": 1000, "n_classes": 2,
        "csp_dim": 6, "dropout": 0.3 if mode == "offline" else 0.1,
        "enhance": enhance, "dann": dann, "causal": mode == "online",
    }
    model = EEGNetLight(**model_config).to(device)
    set_seed(seed + 1)
    teacher = ShallowConvNetTeacher(len(MOTOR_CHANNELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    config = {
        "format_version": 1, "mode": mode, "variant": variant, "seed": seed,
        "model": model_config, "channels": list(MOTOR_CHANNELS), "labels": ["left", "right"],
        "sample_rate": 250, "window_samples": 1000, "input_unit": "OpenBMI native MAT units",
        "subjects": args.subjects, "domain_source": args.domain_source,
        "groups_csv_sha256": hashlib.sha256(args.groups_csv.read_bytes()).hexdigest(),
        "data_fingerprint": preprocessing["data_fingerprint"],
        "code_fingerprint": hashlib.sha256(b"".join(
            (Path(__file__).parent / name).read_bytes() for name in ("model.py", "data.py", "train.py")
        )).hexdigest(),
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                        "numpy": np.__version__, "cuda": torch.version.cuda},
        "protocol": {"train": "session1/EEG_MI_train", "validation": "session1/EEG_MI_test",
                     "test": "session2/EEG_MI_test", "unused": "session2/EEG_MI_train"},
        "hyperparameters": {key: value for key, value in vars(args).items()
                            if key not in {"resume", "output_root", "data_root", "cache_root", "groups_csv"}},
    }
    if args.resume and (output / "config.json").exists():
        old_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if old_config != config:
            raise ValueError("Resume configuration differs. Use original options or a new output directory.")
        if not last_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {last_path}")
    joblib.dump(preprocessing, output / "preprocessing.joblib")
    write_json(output / "config.json", config)
    start_epoch, best_score, patience, history = 0, -1.0, 0, []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        teacher_optimizer.load_state_dict(checkpoint["teacher_optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score, patience, history = checkpoint["best_score"], checkpoint["patience"], checkpoint["history"]
    joint = train_data.tensors[3] * 2 + train_data.tensors[1]
    counts = torch.bincount(joint, minlength=4)
    sample_weights = 1.0 / counts[joint].double()
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    for epoch in range(start_epoch, args.epochs):
        if patience >= args.patience:
            break
        set_seed(seed + epoch * 1009)
        generator = torch.Generator().manual_seed(seed + epoch)
        sampler = WeightedRandomSampler(sample_weights, len(train_data), replacement=True, generator=generator)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
            pin_memory=device.type == "cuda", generator=generator,
        )
        model.train()
        teacher.train()
        totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "domain": 0.0}
        examples = 0
        skipped_domain_batches = 0
        for batch_index, (eeg, targets, csp_features, domains, _) in enumerate(train_loader):
            eeg, targets = eeg.to(device), targets.to(device)
            csp_features, domains = csp_features.to(device), domains.to(device)
            teacher_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                teacher_logits = teacher(eeg)
                teacher_loss = functional.cross_entropy(teacher_logits, targets)
            scaler.scale(teacher_loss).backward()
            scaler.unscale_(teacher_optimizer)
            nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            scaler.step(teacher_optimizer)
            progress = (epoch + batch_index / len(train_loader)) / max(1, args.epochs - 1)
            coefficient = args.grl_max * (2 / (1 + math.exp(-10 * progress)) - 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, domain_logits, _ = model(eeg, csp_features, coefficient)
                ce = functional.cross_entropy(logits, targets)
                variance = logits.float().softmax(1).var(dim=1, unbiased=False).mean()
                temperature = args.temperature
                kd = functional.kl_div(
                    functional.log_softmax(logits.float() / temperature, dim=1),
                    functional.softmax(teacher_logits.detach().float() / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
                domain_loss = logits.new_zeros(())
                if domain_logits is not None:
                    if domains.unique().numel() == 2:
                        domain_loss = 0.5 * sum(
                            functional.cross_entropy(domain_logits[domains == domain], domains[domains == domain])
                            for domain in (0, 1)
                        )
                    else:
                        skipped_domain_batches += 1
                loss = ((1 - args.distill_alpha) * (ce + args.variance_weight * variance)
                        + args.distill_alpha * kd + args.domain_weight * domain_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = targets.shape[0]
            examples += batch_size
            for name, value in (("loss", loss), ("ce", ce), ("kd", kd), ("domain", domain_loss)):
                totals[name] += float(value.detach()) * batch_size
        score, rows = validate(model, val_loader, device)
        scheduler.step(score)
        improved = score > best_score
        patience = 0 if improved else patience + 1
        best_score = max(best_score, score)
        history.append({"epoch": epoch + 1, **{key: value / examples for key, value in totals.items()},
                        "val_target_balanced_accuracy": score, "grl": coefficient,
                        "skipped_domain_batches": skipped_domain_batches})
        checkpoint = {
            "model": model.state_dict(), "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(), "teacher_optimizer": teacher_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_score": best_score, "patience": patience, "history": history,
        }
        checkpoint_save(last_path, checkpoint)
        if improved:
            checkpoint_save(output / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                                "validation_score": score})
            write_json(output / "validation_subjects.json", rows)
        write_json(output / "history.json", history)
        print(f"[{mode}/{variant}/{seed}] epoch={epoch + 1} loss={totals['loss'] / examples:.4f} "
              f"val_target_bacc={score:.4f} best={best_score:.4f}", flush=True)


def parser():
    argument_parser = argparse.ArgumentParser(description="OpenBMI supervised Plan A + knowledge distillation")
    argument_parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "mne_data")
    argument_parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs/plan_a_cache")
    argument_parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "models/openbmi_plan_a")
    argument_parser.add_argument("--groups-csv", type=Path, default=PROJECT_ROOT / "openbmi_bci_illiteracy_result.csv")
    argument_parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    argument_parser.add_argument("--modes", choices=["offline", "online"], nargs="+", default=["offline", "online"])
    argument_parser.add_argument("--variants", choices=list(VARIANTS), nargs="+", default=["baseline", "plan_a"])
    argument_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    argument_parser.add_argument("--domain-source", choices=["supplied", "training"], default="supplied")
    argument_parser.add_argument("--group-seed", type=int, default=42)
    argument_parser.add_argument("--threshold", type=float, default=0.70)
    argument_parser.add_argument("--epochs", type=int, default=120)
    argument_parser.add_argument("--patience", type=int, default=30)
    argument_parser.add_argument("--batch-size", type=int, default=8)
    argument_parser.add_argument("--workers", type=int, default=0)
    argument_parser.add_argument("--lr", type=float, default=5e-4)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-2)
    argument_parser.add_argument("--distill-alpha", type=float, default=0.5)
    argument_parser.add_argument("--temperature", type=float, default=3.0)
    argument_parser.add_argument("--variance-weight", type=float, default=1e-5)
    argument_parser.add_argument("--domain-weight", type=float, default=0.3)
    argument_parser.add_argument("--grl-max", type=float, default=1.0)
    argument_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    argument_parser.add_argument("--amp", action="store_true")
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument("--prepare-only", action="store_true")
    return argument_parser


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or args.workers < 0:
        raise ValueError("Positive epochs/patience, batch size >= 2 and workers >= 0 required.")
    if not 0 <= args.distill_alpha <= 1 or args.temperature <= 0:
        raise ValueError("Invalid distillation parameters.")
    if not 0 < args.threshold < 1 or min(args.domain_weight, args.grl_max, args.variance_weight) < 0:
        raise ValueError("Invalid threshold or loss weight.")
    if not args.subjects or len(set(args.subjects)) != len(args.subjects) or not set(args.subjects) <= set(range(1, 55)):
        raise ValueError("Provide unique OpenBMI subjects in 1..54.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants) or len(set(args.modes)) != len(args.modes):
        raise ValueError("Seeds, variants and modes must not contain duplicates.")
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    import mne
    mne.set_log_level("ERROR")
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.version.split()[0]}, torch: {torch.__version__}, device: {args.device}", flush=True)
    if args.domain_source == "supplied":
        print("Supplied groups were estimated using both sessions; results are exploratory. "
              "Use --domain-source training for a test-independent grouping.", flush=True)
    for mode in args.modes:
        train_data, val_data, preprocessing = prepare(args, mode)
        directory = args.output_root / mode
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / f"groups_{args.domain_source}.json", preprocessing["groups"])
        if args.prepare_only:
            continue
        for seed in args.seeds:
            for variant in args.variants:
                train_variant(args, mode, variant, seed, train_data, val_data, preprocessing)


if __name__ == "__main__":
    main()
