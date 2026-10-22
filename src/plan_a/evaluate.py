import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .data import cached_trials
from .inference import Predictor
from .train import PROJECT_ROOT, subject_metrics, write_json


def evaluate_run(directory, args):
    predictor = Predictor(directory, args.device)
    config = predictor.config
    rows = []
    prediction_directory = args.output_root / config["mode"] / config["variant"] / f"seed_{config['seed']}"
    prediction_directory.mkdir(parents=True, exist_ok=True)
    for subject in config["subjects"]:
        eeg, labels = cached_trials(args.data_root, args.cache_root, subject, 2, "test", config["mode"])
        probabilities = predictor.predict_trials(eeg, args.batch_size)
        predictions = probabilities.argmax(1)
        group = predictor.preprocessing["groups"][subject]
        domains = np.full(len(labels), group[config["domain_source"]])
        row = subject_metrics(labels, predictions, np.full(len(labels), subject), domains)[0]
        row.update({
            "mode": config["mode"], "variant": config["variant"], "seed": config["seed"],
            "supplied_low": group["supplied"], "training_low": group["training"],
            "domain_source": config["domain_source"],
        })
        rows.append(row)
        np.savez(prediction_directory / f"subject_{subject}.npz", labels=labels, probabilities=probabilities)
        print(f"[{config['mode']}/{config['variant']}/{config['seed']}] "
              f"subject={subject} accuracy={row['accuracy']:.4f}", flush=True)
    return rows, config


def paired_statistics(baseline, candidate, bootstrap_samples=10000):
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    if len(differences) < 2:
        return {"subjects": len(differences), "mean_gain": float(differences.mean()),
                "ci95": None, "p_two_sided": None, "note": "At least two subjects required."}
    generator = np.random.default_rng(2026)
    bootstrapped = generator.choice(differences, size=(bootstrap_samples, len(differences)), replace=True).mean(1)
    probability = 1.0 if np.allclose(differences, 0) else float(wilcoxon(differences, alternative="two-sided").pvalue)
    return {
        "subjects": len(differences), "baseline_accuracy": float(np.mean(baseline)),
        "candidate_accuracy": float(np.mean(candidate)), "mean_gain": float(differences.mean()),
        "ci95": np.quantile(bootstrapped, [0.025, 0.975]).tolist(), "p_two_sided": probability,
        "improved_subjects": int((differences > 0).sum()),
        "worsened_subjects": int((differences < 0).sum()),
        "unchanged_subjects": int((differences == 0).sum()),
        "baseline_below_70": int((np.asarray(baseline) < 0.70).sum()),
        "candidate_below_70": int((np.asarray(candidate) < 0.70).sum()),
    }


def compare(rows):
    reports = []
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        domain_sources = {row["domain_source"] for row in mode_rows}
        if len(domain_sources) != 1:
            raise ValueError("Cannot compare runs with different training domain sources.")
        uses_test_information = domain_sources == {"supplied"}
        variants = sorted({row["variant"] for row in mode_rows} - {"baseline"})
        for variant in variants:
            baseline_seeds = {row["seed"] for row in mode_rows if row["variant"] == "baseline"}
            candidate_seeds = {row["seed"] for row in mode_rows if row["variant"] == variant}
            if not baseline_seeds or baseline_seeds != candidate_seeds:
                raise ValueError(f"{mode}/{variant}: baseline and candidate must have the same seeds.")
            for group in ("all", "supplied_low", "training_low"):
                per_subject = {}
                for row in mode_rows:
                    if row["variant"] not in {"baseline", variant} or (group != "all" and not row[group]):
                        continue
                    per_subject.setdefault(row["subject"], {}).setdefault(row["variant"], []).append(row["accuracy"])
                if not per_subject:
                    continue
                subjects = sorted(per_subject)
                for subject in subjects:
                    values = per_subject[subject]
                    if any(len(values.get(name, [])) != len(baseline_seeds) for name in ("baseline", variant)):
                        raise ValueError("Missing paired subject/seed results.")
                baseline = [np.mean(per_subject[subject]["baseline"]) for subject in subjects]
                candidate = [np.mean(per_subject[subject][variant]) for subject in subjects]
                reports.append({"mode": mode, "variant": variant, "group": group,
                                "subject_ids": subjects, "seeds": sorted(baseline_seeds),
                                "training_groups_use_test_information": uses_test_information,
                                "inference_scope": "exploratory" if uses_test_information or group == "supplied_low"
                                                   else "held_out_session",
                                **paired_statistics(baseline, candidate)})
    valid = sorted([report for report in reports if report["p_two_sided"] is not None],
                   key=lambda report: report["p_two_sided"])
    previous = 0.0
    for index, report in enumerate(valid):
        previous = max(previous, min(1.0, report["p_two_sided"] * (len(valid) - index)))
        report["p_holm"] = previous
        report["positive_gain_significant"] = bool(previous < 0.05 and report["mean_gain"] > 0)
    return reports


def main():
    parser = argparse.ArgumentParser(description="Held-out OpenBMI session 2 paired evaluation")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models/openbmi_plan_a")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "mne_data")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs/plan_a_cache")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/openbmi_plan_a")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    directories = sorted(path.parent for path in args.model_root.glob("*/*/seed_*/best.pt"))
    if not directories:
        raise FileNotFoundError(f"No trained runs under {args.model_root}")
    rows, configs = [], []
    for directory in directories:
        run_rows, config = evaluate_run(directory, args)
        rows.extend(run_rows)
        configs.append(config)
    for mode in {config["mode"] for config in configs}:
        selected = [config for config in configs if config["mode"] == mode]
        reference = selected[0]
        for config in selected[1:]:
            for key in ("subjects", "domain_source", "groups_csv_sha256", "data_fingerprint",
                        "code_fingerprint", "protocol", "channels"):
                if config[key] != reference[key]:
                    raise ValueError(f"Cannot pair runs with different {key}.")
            original = {key: value for key, value in reference["hyperparameters"].items()
                        if key not in {"variants", "seeds", "device", "workers", "amp", "prepare_only"}}
            candidate = {key: value for key, value in config["hyperparameters"].items()
                         if key not in {"variants", "seeds", "device", "workers", "amp", "prepare_only"}}
            if candidate != original:
                raise ValueError("Paired runs must use matching training hyperparameters.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "subject_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": configs[0]["protocol"],
        "interpretation": "Seeds are averaged within subject before paired inference. "
                          "95% intervals bootstrap subjects; p values use two-sided Wilcoxon with Holm correction. "
                          "Supplied groups used both sessions and are exploratory. Training-only groups with "
                          "domain_source=training avoid that test-derived grouping. The 70% cutoff is an "
                          "operational performance threshold, not a clinical diagnosis.",
        "runs": [{"mode": config["mode"], "variant": config["variant"], "seed": config["seed"],
                  "domain_source": config["domain_source"]} for config in configs],
        "comparisons": compare(rows),
    }
    write_json(args.output_root / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
