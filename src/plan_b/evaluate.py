import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.plan_a.data import cached_trials
from src.plan_a.evaluate import compare
from src.plan_a.train import PROJECT_ROOT, subject_metrics, write_json
from .inference import Predictor


def evaluate_run(directory, args):
    predictor = Predictor(directory, args.device)
    config = predictor.config
    rows = []
    output = args.output_root / config["mode"] / config["variant"] / f"seed_{config['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    for subject in config["subjects"]:
        eeg, labels = cached_trials(args.data_root, args.cache_root, subject, 2, "test", config["mode"])
        probabilities = predictor.predict_trials(eeg, args.batch_size)
        groups = predictor.preprocessing["groups"][subject]
        domains = np.full(len(labels), groups[config["domain_source"]])
        row = subject_metrics(labels, probabilities.argmax(1), np.full(len(labels), subject), domains)[0]
        row.update({
            "mode": config["mode"], "variant": config["variant"], "seed": config["seed"],
            "supplied_low": groups["supplied"], "training_low": groups["training"],
            "domain_source": config["domain_source"],
        })
        rows.append(row)
        np.savez(output / f"subject_{subject}.npz", labels=labels, probabilities=probabilities)
        print(f"[B/{config['mode']}/{config['variant']}/{config['seed']}] "
              f"subject={subject} accuracy={row['accuracy']:.4f}", flush=True)
    return rows, config


def check_configurations(configs):
    for mode in {config["mode"] for config in configs}:
        selected = [config for config in configs if config["mode"] == mode]
        reference = selected[0]
        for config in selected:
            if config.get("framework") != "plan_b":
                raise ValueError("Plan B evaluator cannot load Plan A runs.")
            for key in ("subjects", "domain_source", "groups_csv_sha256", "data_fingerprint",
                        "code_fingerprint", "protocol", "channels", "model"):
                if key == "model":
                    original = {name: value for name, value in reference[key].items() if name != "contrastive"}
                    candidate = {name: value for name, value in config[key].items() if name != "contrastive"}
                else:
                    original, candidate = reference[key], config[key]
                if original != candidate:
                    raise ValueError(f"Cannot pair Plan B runs with different {key}.")
            excluded = {"variants", "seeds", "device", "workers", "prepare_only"}
            original = {key: value for key, value in reference["hyperparameters"].items() if key not in excluded}
            candidate = {key: value for key, value in config["hyperparameters"].items() if key not in excluded}
            if original != candidate:
                raise ValueError("Plan B runs must use the same experimental hyperparameters.")


def compare_variants(rows):
    reports = compare(rows)
    for report in reports:
        report.update({"reference": "baseline", "candidate": report["variant"]})
    available = {row["variant"] for row in rows}
    for reference in ("augment", "contrast"):
        if {reference, "plan_b"} <= available:
            paired = []
            for row in rows:
                if row["variant"] in {reference, "plan_b"}:
                    selected = dict(row)
                    selected["variant"] = "baseline" if row["variant"] == reference else "plan_b"
                    paired.append(selected)
            for report in compare(paired):
                report.update({"reference": reference, "candidate": "plan_b"})
                reports.append(report)
    valid = sorted((report for report in reports if report["p_two_sided"] is not None),
                   key=lambda report: report["p_two_sided"])
    previous = 0.0
    for index, report in enumerate(valid):
        previous = max(previous, min(1.0, report["p_two_sided"] * (len(valid) - index)))
        report["p_holm"] = previous
        report["positive_gain_significant"] = bool(previous < 0.05 and report["mean_gain"] > 0)
    return reports


def main():
    parser = argparse.ArgumentParser(description="Independent held-out Plan B evaluation")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models/openbmi_plan_b")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "mne_data")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs/openbmi_cache")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/openbmi_plan_b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    directories = sorted(path.parent for path in args.model_root.glob("*/*/seed_*/best.pt"))
    if not directories:
        raise FileNotFoundError(f"No Plan B runs under {args.model_root}")
    configs = [json.loads((directory / "config.json").read_text(encoding="utf-8")) for directory in directories]
    check_configurations(configs)
    rows = []
    for directory in directories:
        run_rows, _ = evaluate_run(directory, args)
        rows.extend(run_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "subject_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "framework": "plan_b", "protocol": configs[0]["protocol"],
        "interpretation": "Seeds are averaged within subject; subject bootstrap CI and paired Wilcoxon "
                          "with Holm correction. Supplied test-derived groups are exploratory. "
                          "Evaluation uses unaugmented held-out session 2 trials.",
        "comparisons": compare_variants(rows),
    }
    write_json(args.output_root / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
