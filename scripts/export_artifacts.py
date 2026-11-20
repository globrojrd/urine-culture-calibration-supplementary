"""Export portable split identities, logits and histories from the original study.

This reads saved experiments; it does not retrain or alter any source file.
Use export_review_inputs.py separately for the seven frozen evaluation policies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from urineculture.data import load_manifest, ORIGINAL_SPLIT_SHA256
from urineculture.factorial import read_logits, LOGIT_COLUMNS


def export_artifacts(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output == project or project in output.parents:
        raise ValueError("Export outside the original experiment folder.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory.")
    source_manifest = project / "01_splits/urocultivo_extension_split_v1.csv"
    frame = load_manifest(source_manifest)
    frame = frame[["sample_id", "filename", "label", "split_extension"]].copy()
    frame["filepath"] = frame.label + "/" + frame.filename
    sources, records = [], []
    for arch in ("resnet18", "efficientnetv2_s", "swin_t"):
        for loss in ("cross_entropy", "label_smoothing", "focal_loss"):
            for seed in (42, 123, 2026):
                relative = Path(arch) / loss
                if loss == "focal_loss":
                    relative /= "gamma_2"
                relative /= f"seed_{seed}"
                record = {"architecture": arch, "loss": loss, "seed": seed}
                for split in ("calibration", "threshold_selection", "final_test"):
                    source = project / "03_logits" / relative / f"{split}_logits.csv"
                    destination = Path("logits") / relative / f"{split}.csv"
                    sources.append((source, destination, split))
                    record[split] = destination.as_posix()
                records.append(record)
    histories = [(project / f"08_logs/resnet18/focal_loss/gamma_2/seed_{s}/training_history.csv",
                  Path(f"histories/resnet18_focal_seed_{s}.csv")) for s in (42, 123, 2026)]
    missing = [str(s.relative_to(project)) for s, _, _ in sources if not s.is_file()]
    missing += [str(s.relative_to(project)) for s, _ in histories if not s.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved artifacts:\n" + "\n".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir()
    frame.to_csv(output / "splits/urocultivo_extension_split_v1.csv", index=False)
    audit = []
    label_map = {"Negative": 0, "Positive": 1, "Uncertain": 2}
    for source, destination, split in sources:
        logits = read_logits(source)
        expected = frame[frame.split_extension == split].set_index("sample_id")
        if set(logits.sample_id) != set(expected.index):
            raise ValueError(f"Logits do not match the original {split} manifest: {source.name}")
        if logits.true_label.tolist() != expected.loc[logits.sample_id, "label"].map(label_map).tolist():
            raise ValueError(f"Logit labels differ from manifest: {source.name}")
        path = output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        logits[["sample_id", "true_label", *LOGIT_COLUMNS]].to_csv(path, index=False)
        audit.append({"source": source.relative_to(project).as_posix(),
                      "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "export": destination.as_posix(),
                      "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    history_columns = ["epoch", "train_loss", "validation_loss", "train_macro_f1", "validation_macro_f1"]
    for source, destination in histories:
        history = pd.read_csv(source)
        if not set(history_columns).issubset(history):
            raise ValueError(f"History lacks measured fields: {source}")
        path = output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        history[history_columns].to_csv(path, index=False)
        audit.append({"source": source.relative_to(project).as_posix(),
                      "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "export": destination.as_posix(),
                      "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(records).to_csv(output / "experiments.csv", index=False)
    report = {"source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
              "notebook_manifest_sha256": ORIGINAL_SPLIT_SHA256,
              "portable_paths": True, "sample_ids_and_input_order_preserved": True,
              "images_and_checkpoints_included": False, "artifacts": audit}
    (output / "artifact_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Exported original saved artifacts to", export_artifacts(args.project, args.output))


if __name__ == "__main__":
    main()
