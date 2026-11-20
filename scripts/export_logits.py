"""Export aligned logits from the 27 trained checkpoints (notebook cell 78).

Only trusted local checkpoints may be loaded: historical Torch checkpoints
contain Python metadata and require weights_only=False.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from urineculture.data import load_manifest, create_dataloaders, LABEL_MAP
from urineculture.training import create_model, experiment_subpath
from urineculture.factorial import ARCHITECTURES, LOSSES, SEEDS, LOGIT_COLUMNS, sha256_file, save_json


def export(args):
    frame = load_manifest(args.split_manifest, args.image_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    experiments, audit = [], []
    splits = ("calibration", "threshold_selection", "final_test")
    for architecture in ARCHITECTURES:
        for loss in LOSSES:
            for seed in SEEDS:
                checkpoint = (Path(args.checkpoint_root) / experiment_subpath(architecture, loss, seed)
                              / "best_model_by_macro_f1.pth")
                loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if not isinstance(loaded, dict) or "model_state_dict" not in loaded:
                    raise ValueError(f"Expected historical checkpoint dictionary: {checkpoint}")
                if loaded.get("label_map", LABEL_MAP) != LABEL_MAP:
                    raise ValueError(f"Checkpoint class mapping differs: {checkpoint}")
                model = create_model(architecture, pretrained=False)
                state = {key.removeprefix("module."): value for key, value in loaded["model_state_dict"].items()}
                model.load_state_dict(state, strict=True)
                model.to(device).eval()
                loaders = create_dataloaders(frame, seed, batch_size=32 if architecture == "resnet18" else 16,
                                             num_workers=args.num_workers)
                directory = output / architecture / loss / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                experiment = {"architecture": architecture, "loss": loss, "seed": seed}
                for split in splits:
                    rows = []
                    with torch.inference_mode():
                        for batch in loaders[split]:
                            z = model(batch["image"].to(device, non_blocking=True)).cpu().numpy()
                            y = batch["label"].cpu().numpy()
                            if z.shape != (len(y), 3) or not np.isfinite(z).all():
                                raise ValueError("Model produced invalid logits.")
                            for index, sample_id in enumerate(batch["sample_id"]):
                                rows.append({"sample_id": str(sample_id), "true_label": int(y[index]),
                                             **dict(zip(LOGIT_COLUMNS, map(float, z[index])))})
                    exported = pd.DataFrame(rows)
                    expected = frame[frame.split_extension == split]
                    if exported.sample_id.tolist() != expected.sample_id.astype(str).tolist():
                        raise ValueError("Export changed the original manifest sample order.")
                    if exported.true_label.tolist() != expected.label.map(LABEL_MAP).tolist():
                        raise ValueError("Export labels disagree with the saved manifest.")
                    destination = directory / f"{split}.csv"
                    if destination.exists():
                        raise FileExistsError(f"Refusing to overwrite logit table: {destination}")
                    exported.to_csv(destination, index=False)
                    experiment[split] = destination.relative_to(output).as_posix()
                    audit.append({"architecture": architecture, "loss": loss, "seed": seed, "split": split,
                                  "checkpoint_sha256": sha256_file(checkpoint),
                                  "logits_sha256": sha256_file(destination), "number_of_samples": len(exported)})
                experiments.append(experiment)
                del model, loaded, loaders
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"Exported {architecture}, {loss}, seed {seed}.")
    pd.DataFrame(experiments).to_csv(output / "experiments.csv", index=False)
    save_json(output / "export_audit.json", {"split_manifest_sha256": sha256_file(args.split_manifest),
                                            "torch_version": torch.__version__, "device": str(device),
                                            "exports": audit})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--image-root", help="Directory containing Negative, Positive, and Uncertain images.")
    parser.add_argument("--checkpoint-root", required=True, help="The 02_checkpoints directory.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
