import argparse
import copy
import csv
import json
from pathlib import Path

from src.plan_a.evaluate import compare, evaluate_run as evaluate_a
from src.plan_a.train import PROJECT_ROOT, write_json
from src.plan_b.evaluate import evaluate_run as evaluate_b


def discover(root, framework):
    allowed = {"baseline", "plan_a"} if framework == "A" else {"baseline", "plan_b"}
    runs = []
    for path in sorted(root.glob("*/*/seed_*/best.pt")):
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        if config["variant"] not in allowed:
            continue
        actual = config.get("framework", "plan_a")
        if actual != ("plan_a" if framework == "A" else "plan_b"):
            raise ValueError(f"Wrong framework under {root}: {path}")
        runs.append((framework, path.parent, config))
    if not runs:
        raise FileNotFoundError(f"No {framework} runs under {root}")
    return runs


def check_pairing(runs):
    shared = ("subjects", "domain_source", "groups_csv_sha256", "data_fingerprint",
              "protocol", "channels", "labels", "sample_rate", "window_samples", "input_unit")
    hyperparameters = ("epochs", "patience", "batch_size", "lr", "weight_decay", "distill_alpha",
                       "temperature", "variance_weight", "group_seed", "threshold", "amp")
    backbone_keys = ("n_channels", "n_times", "n_classes", "csp_dim", "dropout", "causal")
    for mode in {config["mode"] for _, _, config in runs}:
        selected = [(framework, config) for framework, _, config in runs if config["mode"] == mode]
        reference = selected[0][1]
        expected = {"A/baseline", "A/plan_a", "B/baseline", "B/plan_b"}
        seeds_by_method = {}
        for framework, config in selected:
            method = f"{framework}/{config['variant']}"
            seeds = seeds_by_method.setdefault(method, set())
            if config["seed"] in seeds:
                raise ValueError(f"Duplicate run {mode}/{method}/{config['seed']}")
            seeds.add(config["seed"])
            for key in shared:
                if config[key] != reference[key]:
                    raise ValueError(f"A/B comparison requires matching {key} in {mode}.")
            for key in hyperparameters:
                if config["hyperparameters"][key] != reference["hyperparameters"][key]:
                    raise ValueError(f"A/B comparison requires matching {key} in {mode}.")
            for key in backbone_keys:
                if config["model"][key] != reference["model"][key]:
                    raise ValueError(f"A/B comparison requires matching backbone {key}.")
        if seeds_by_method.keys() != expected:
            raise ValueError(f"Each mode needs all four methods: {sorted(expected)}")
        if any(seeds != seeds_by_method["A/baseline"] for seeds in seeds_by_method.values()):
            raise ValueError("All four methods must have the same seeds.")


def compare_methods(rows):
    reports = []
    pairs = (("A/baseline", "A/plan_a"), ("B/baseline", "B/plan_b"),
             ("A/plan_a", "B/plan_b"), ("A/baseline", "B/baseline"))
    for reference, candidate in pairs:
        paired = []
        for row in rows:
            if row["method"] in {reference, candidate}:
                selected = copy.copy(row)
                selected["variant"] = "baseline" if row["method"] == reference else candidate
                paired.append(selected)
        for report in compare(paired):
            report.update({"reference": reference, "candidate": candidate})
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
    parser = argparse.ArgumentParser(description="Matched OpenBMI baseline / Plan A / Plan B comparison")
    parser.add_argument("--plan-a-root", type=Path, default=PROJECT_ROOT / "models/openbmi_plan_a")
    parser.add_argument("--plan-b-root", type=Path, default=PROJECT_ROOT / "models/openbmi_plan_b")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "mne_data")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs/openbmi_cache")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/openbmi_plan_ab")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    runs = discover(args.plan_a_root, "A") + discover(args.plan_b_root, "B")
    check_pairing(runs)
    rows = []
    output_root = args.output_root
    for framework, directory, _ in runs:
        args.output_root = output_root / framework
        run_rows, _ = (evaluate_a if framework == "A" else evaluate_b)(directory, args)
        for row in run_rows:
            row["framework"] = framework
            row["method"] = f"{framework}/{row['variant']}"
        rows.extend(run_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "subject_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "interpretation": "Gain is candidate minus reference. Seeds are averaged within subject. "
                          "Holm correction covers all comparisons in this report. Differences include "
                          "the full methods; B uses two training views, so compute budgets differ. "
                          "Supplied groups derived from both sessions are exploratory.",
        "comparisons": compare_methods(rows),
    }
    write_json(output_root / "comparison.json", report)
    write_json(output_root / "run_configurations.json", [config for _, _, config in runs])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
