import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.plan_a.data import MOTOR_CHANNELS
from src.plan_a.train import PROJECT_ROOT, checkpoint_save, prepare, set_seed, validate, write_json
from .augment import EEGAugment, prepare_view
from .model import EEGNetLightPlanB, ShallowConvNetTeacher, supervised_contrastive_loss


VARIANTS = {
    "baseline": (False, False), "augment": (True, False),
    "contrast": (False, True), "plan_b": (True, True),
}


def train_variant(args, mode, variant, seed, train_data, val_data, preprocessing):
    output = args.output_root / mode / variant / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and not args.resume:
        raise FileExistsError(f"Run exists: {output}. Use --resume or another output root.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    augment, contrastive = VARIANTS[variant]
    set_seed(seed)
    model_config = {
        "n_channels": len(MOTOR_CHANNELS), "n_times": 1000, "n_classes": 2, "csp_dim": 6,
        "dropout": 0.3 if mode == "offline" else 0.1, "causal": mode == "online",
        "contrastive": contrastive, "projection_dim": args.projection_dim,
    }
    model = EEGNetLightPlanB(**model_config).to(device)
    set_seed(seed + 1)
    teacher = ShallowConvNetTeacher(len(MOTOR_CHANNELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    files = [Path(__file__).parent / name for name in ("model.py", "augment.py", "train.py")]
    files += [Path(__file__).parents[1] / "plan_a" / name for name in ("model.py", "data.py", "train.py")]
    config = {
        "format_version": 1, "framework": "plan_b", "mode": mode, "variant": variant, "seed": seed,
        "model": model_config, "channels": list(MOTOR_CHANNELS), "labels": ["left", "right"],
        "sample_rate": 250, "window_samples": 1000, "input_unit": "OpenBMI native MAT units",
        "subjects": args.subjects, "domain_source": args.domain_source,
        "groups_csv_sha256": hashlib.sha256(args.groups_csv.read_bytes()).hexdigest(),
        "data_fingerprint": preprocessing["data_fingerprint"],
        "code_fingerprint": hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest(),
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                        "numpy": np.__version__, "cuda": torch.version.cuda},
        "protocol": {"train": "session1/EEG_MI_train", "validation": "session1/EEG_MI_test",
                     "test": "session2/EEG_MI_test", "unused": "session2/EEG_MI_train"},
        "hyperparameters": {key: value for key, value in vars(args).items()
                            if key not in {"resume", "output_root", "data_root", "cache_root", "groups_csv"}},
    }
    last_path = output / "last.pt"
    if args.resume and (output / "config.json").exists():
        if json.loads((output / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration, code or data differs from the saved run.")
        if not last_path.exists():
            raise FileNotFoundError(last_path)
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
    weights = 1.0 / counts[joint].double()
    augmenter = EEGAugment(band_strength=args.band_strength, channel_mask=args.channel_mask,
                           noise_std=args.noise_std) if augment else None
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    for epoch in range(start_epoch, args.epochs):
        if patience >= args.patience:
            break
        set_seed(seed + epoch * 1009)
        generator = torch.Generator().manual_seed(seed + epoch)
        sampler = WeightedRandomSampler(weights, len(train_data), replacement=True, generator=generator)
        loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=device.type == "cuda", generator=generator)
        model.train()
        teacher.train()
        totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "contrastive": 0.0}
        examples, skipped_contrastive_batches = 0, 0
        for eeg, targets, cached_csp, _, _ in loader:
            views = [prepare_view(eeg, preprocessing, augmenter) for _ in range(2)] if augment else [(eeg, cached_csp)]
            if contrastive and not augment:
                views = [(eeg, cached_csp), (eeg, cached_csp)]
            batch_size, view_count = len(targets), len(views)
            targets = targets.to(device)
            teacher_optimizer.zero_grad(set_to_none=True)
            teacher_outputs, teacher_losses = [], []
            with torch.autocast(device_type=device.type, enabled=amp):
                for view_eeg, _ in views:
                    teacher_logits = teacher(view_eeg.to(device))
                    teacher_outputs.append(teacher_logits.detach())
                    teacher_losses.append(functional.cross_entropy(teacher_logits, targets))
                teacher_loss = torch.stack(teacher_losses).mean()
            scaler.scale(teacher_loss).backward()
            scaler.unscale_(teacher_optimizer)
            nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            scaler.step(teacher_optimizer)
            optimizer.zero_grad(set_to_none=True)
            ce_losses, kd_losses, variances, projections = [], [], [], []
            with torch.autocast(device_type=device.type, enabled=amp):
                for (view_eeg, view_csp), teacher_logits in zip(views, teacher_outputs):
                    if contrastive:
                        logits, projected = model(view_eeg.to(device), view_csp.to(device), return_features=True)
                        projections.append(projected)
                    else:
                        logits = model(view_eeg.to(device), view_csp.to(device))
                    ce_losses.append(functional.cross_entropy(logits, targets))
                    variances.append(logits.float().softmax(1).var(dim=1, unbiased=False).mean())
                    kd_losses.append(functional.kl_div(
                        functional.log_softmax(logits.float() / args.temperature, dim=1),
                        functional.softmax(teacher_logits.float() / args.temperature, dim=1),
                        reduction="batchmean",
                    ) * args.temperature ** 2)
                ce, kd, variance = (torch.stack(values).mean() for values in (ce_losses, kd_losses, variances))
                contrastive_loss = logits.new_zeros(())
                if contrastive:
                    contrastive_loss = supervised_contrastive_loss(
                        torch.stack(projections, dim=1), targets, args.contrastive_temperature,
                    )
                    if targets.unique().numel() < 2:
                        skipped_contrastive_batches += 1
                loss = ((1 - args.distill_alpha) * (ce + args.variance_weight * variance)
                        + args.distill_alpha * kd + args.contrastive_weight * contrastive_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            examples += batch_size
            for name, value in (("loss", loss), ("ce", ce), ("kd", kd), ("contrastive", contrastive_loss)):
                totals[name] += float(value.detach()) * batch_size
        score, rows = validate(model, val_loader, device)
        scheduler.step(score)
        improved = score > best_score
        patience = 0 if improved else patience + 1
        best_score = max(best_score, score)
        history.append({"epoch": epoch + 1, **{key: value / examples for key, value in totals.items()},
                        "val_target_balanced_accuracy": score, "views_per_trial": view_count,
                        "skipped_contrastive_batches": skipped_contrastive_batches})
        checkpoint_save(last_path, {
            "model": model.state_dict(), "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(), "teacher_optimizer": teacher_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
            "best_score": best_score, "patience": patience, "history": history,
        })
        if improved:
            checkpoint_save(output / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                                "validation_score": score})
            write_json(output / "validation_subjects.json", rows)
        write_json(output / "history.json", history)
        print(f"[{mode}/{variant}/{seed}] epoch={epoch + 1} loss={totals['loss'] / examples:.4f} "
              f"contrastive={totals['contrastive'] / examples:.4f} val_target_bacc={score:.4f}", flush=True)


def parser():
    result = argparse.ArgumentParser(description="Independent Plan B: distillation + EEG augmentation + SupCon")
    result.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "mne_data")
    result.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs/openbmi_cache")
    result.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "models/openbmi_plan_b")
    result.add_argument("--groups-csv", type=Path, default=PROJECT_ROOT / "openbmi_bci_illiteracy_result.csv")
    result.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    result.add_argument("--modes", choices=["offline", "online"], nargs="+", default=["offline", "online"])
    result.add_argument("--variants", choices=list(VARIANTS), nargs="+", default=["baseline", "plan_b"])
    result.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    result.add_argument("--domain-source", choices=["supplied", "training"], default="supplied")
    result.add_argument("--group-seed", type=int, default=42)
    result.add_argument("--threshold", type=float, default=0.70)
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--patience", type=int, default=30)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--lr", type=float, default=5e-4)
    result.add_argument("--weight-decay", type=float, default=1e-2)
    result.add_argument("--distill-alpha", type=float, default=0.5)
    result.add_argument("--temperature", type=float, default=3.0)
    result.add_argument("--variance-weight", type=float, default=1e-5)
    result.add_argument("--contrastive-weight", type=float, default=0.1)
    result.add_argument("--contrastive-temperature", type=float, default=0.1)
    result.add_argument("--projection-dim", type=int, default=128)
    result.add_argument("--band-strength", type=float, default=0.2)
    result.add_argument("--channel-mask", type=float, default=0.2)
    result.add_argument("--noise-std", type=float, default=0.05)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--amp", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--prepare-only", action="store_true")
    return result


def main():
    args = parser().parse_args()
    numeric = (args.lr, args.weight_decay, args.distill_alpha, args.temperature, args.variance_weight,
               args.contrastive_weight, args.contrastive_temperature, args.threshold)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Training parameters must be finite.")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 2 or args.workers < 0 or args.projection_dim < 1:
        raise ValueError("Invalid epochs, patience, batch size, workers or projection dimension.")
    if not 0 <= args.distill_alpha <= 1 or min(args.temperature, args.contrastive_temperature, args.lr) <= 0:
        raise ValueError("Invalid distillation/contrastive parameters.")
    if min(args.contrastive_weight, args.variance_weight, args.weight_decay) < 0 or not 0 < args.threshold < 1:
        raise ValueError("Invalid loss weight or grouping threshold.")
    for values in (args.subjects, args.modes, args.variants, args.seeds):
        if len(set(values)) != len(values):
            raise ValueError("Subjects, modes, variants and seeds must not contain duplicates.")
    if not set(args.subjects) <= set(range(1, 55)):
        raise ValueError("OpenBMI subject IDs must be in 1..54.")
    EEGAugment(band_strength=args.band_strength, channel_mask=args.channel_mask, noise_std=args.noise_std)
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    import mne
    mne.set_log_level("ERROR")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.domain_source == "supplied":
        print("Supplied groups contain test-session information; this is exploratory. "
              "Use --domain-source training for independent grouping.", flush=True)
    for mode in args.modes:
        train_data, val_data, preprocessing = prepare(args, mode)
        directory = args.output_root / mode
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / f"groups_{args.domain_source}.json", preprocessing["groups"])
        if not args.prepare_only:
            for seed in args.seeds:
                for variant in args.variants:
                    train_variant(args, mode, variant, seed, train_data, val_data, preprocessing)


if __name__ == "__main__":
    main()
