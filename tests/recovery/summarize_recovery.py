"""Summarize saved fits, retaining diagnostic failures and listing missing fits."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

MODELS = ("1PL", "2PL", "3PL", "4PL", "GRM", "PCM", "GPCM")


def collect(run_dir, reps=100):
    output = run_dir
    files = sorted(path for model in MODELS for path in output.glob(f"{model}_[0-9]*.json"))
    metadata = [json.loads(path.read_text()) for path in files]
    if reps < 1:
        raise ValueError("--reps must be positive")
    settings = {json.dumps(m["settings"], sort_keys=True) for m in metadata}
    if len(settings) > 1:
        raise ValueError("Mixed configurations; collect separate runs separately")
    grouped = defaultdict(list)
    inventory = []
    for model in MODELS:
        for replication in range(reps):
            stem = output / f"{model}_{replication:03d}"
            if not stem.with_suffix(".json").exists() or not stem.with_suffix(".csv").exists():
                inventory.append({"model": model, "replication": replication, "status": "missing"})
                continue
            meta = json.loads(stem.with_suffix(".json").read_text())
            if (meta["model"], meta["replication"]) != (model, replication):
                raise ValueError(f"Metadata identity mismatch: {stem}")
            checks = [meta["raw_diagnostics"], meta["item_diagnostics"]]
            problem = meta["divergences"] > 0 or any(
                not d["diagnostics_finite"]
                or (meta["settings"]["chains"] > 1 and
                    (d["max_rhat"] is None or d["max_rhat"] >= 1.01))
                or d["min_bulk_ess"] is None or d["min_bulk_ess"] < 400
                or d["min_tail_ess"] is None or d["min_tail_ess"] < 400
                for d in checks)
            status = "diagnostic_problem" if problem else "ok"
            if meta["settings"]["chains"] == 1 and not problem:
                status = "single_chain_unchecked"
            inventory.append({"model": model, "replication": replication, "status": status,
                              "divergences": meta["divergences"]})
            with stem.with_suffix(".csv").open() as handle:
                for row in csv.DictReader(handle):
                    error = float(row["posterior_mean"]) - float(row["truth"])
                    covered = float(row["lower_95"]) <= float(row["truth"]) <= float(row["upper_95"])
                    grouped[model, row["parameter"]].append((replication, error, covered))
    with (run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "parameter", "completed_reps", "bias", "rmse", "coverage_95"])
        for (model, parameter), rows in sorted(grouped.items()):
            writer.writerow([model, parameter, len({r[0] for r in rows}),
                             sum(r[1] for r in rows) / len(rows),
                             math.sqrt(sum(r[1] ** 2 for r in rows) / len(rows)),
                             sum(r[2] for r in rows) / len(rows)])
    report = {"expected_fits": 7 * reps, "fits": inventory}
    (run_dir / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    counts = {s: sum(r["status"] == s for r in inventory)
              for s in ("ok", "diagnostic_problem", "single_chain_unchecked", "missing")}
    print(json.dumps(counts))
    print(f"Wrote {run_dir / 'summary.csv'} and {run_dir / 'inventory.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reps", type=int, default=100)
    args = parser.parse_args()
    collect(args.run_dir, args.reps)
