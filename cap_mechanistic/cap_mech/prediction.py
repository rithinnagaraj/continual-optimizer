from pathlib import Path
import pandas as pd
from .checkpointing import require_gate, prepare_run, file_hash, atomic_json, digest
from .constants import MECHANISMS, SEEDS
from .runner import manifest, NAMES, provenance
from .records import read_table, write_csv
from .statistics import nested_loso

def run(args):
    require_gate(args.output_dir, args.allow_unverified_reference)
    source = Path(args.output_dir)/"reproduction"
    frame = read_table(source, "task_features")
    if frame.empty:
        raise RuntimeError("No prospective features in the reference runs")
    if not args.smoke:
        expected = {(s, t) for s in SEEDS for t in range(2, 201)}
        if set(zip(frame.seed, frame.task)) != expected or len(frame) != len(expected):
            raise RuntimeError("M5 requires all 995 unmodified Stream-A transitions")
    config = manifest(args, "m5", {str(p.relative_to(source)): file_hash(p)
                                  for p in sorted(source.glob("raw/*/task_features.jsonl"))})
    root = Path(args.output_dir)/NAMES["m5"]
    prepare_run(root, config, args.resume)
    models = {"baseline": ["task", "previous_task_accuracy"],
              "cap_state": MECHANISMS["features"],
              "combined": ["task"]+MECHANISMS["features"]}
    predictions, metrics, coefficients = [], [], []
    for target in ("early_200", "early_auc", "steps_to_85_percent", "task_end_accuracy"):
        for model, features in models.items():
            pred, met, coef = nested_loso(frame, features, target, MECHANISMS["ridge_alphas"])
            for df in (pred, met, coef):
                df["model"], df["outcome"] = model, target
                for key, val in provenance(digest(config["checkpoint_hashes"]), config).items():
                    df[key] = val
            # The capped threshold outcome is a descriptive prediction, not survival inference.
            predictions.append(pred)
            metrics.append(met)
            coefficients.append(coef)
    pred = pd.concat(predictions, ignore_index=True)
    write_csv(root/"raw"/"m5_task_features.csv", frame)
    write_csv(root/"raw"/"m5_fold_predictions.csv", pred)
    write_csv(root/"summary"/"m5_model_metrics.csv", pd.concat(metrics, ignore_index=True))
    write_csv(root/"summary"/"m5_coefficients.csv", pd.concat(coefficients, ignore_index=True))
    from .plotting import prediction_plot
    prediction_plot(pred, root/"summary"/"m5_prediction_plot.png")
    atomic_json(root/"complete.json", {"config_hash": digest(config), "scientifically_verified": not args.allow_unverified_reference and not args.smoke})
