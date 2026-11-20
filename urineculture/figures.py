"""Reviewer-requested figures from measured histories and original images (cell 125).

No images, curves or clinical morphology are synthesized. Existing article assets
can be copied unchanged; regeneration requires the actual source files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 123, 2026)
HISTORY_COLUMNS = ("epoch", "train_loss", "validation_loss", "train_macro_f1", "validation_macro_f1")


def training_curves(history_paths, output):
    import matplotlib.pyplot as plt
    output = Path(output)
    frames = []
    for seed in SEEDS:
        path = Path(history_paths[seed])
        frame = pd.read_csv(path)
        missing = set(HISTORY_COLUMNS) - set(frame)
        if missing:
            raise ValueError(f"Missing history columns in {path}: {missing}")
        if sorted(frame["epoch"].tolist()) != list(range(1, 21)):
            raise ValueError(f"Expected 20 unique consecutive epochs in {path}")
        frame = frame[list(HISTORY_COLUMNS)].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Nonfinite history data in {path}")
        frame["seed"] = seed
        frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    summary = history.groupby("epoch").agg(
        train_loss=("train_loss", "mean"), train_loss_sd=("train_loss", "std"),
        validation_loss=("validation_loss", "mean"), validation_loss_sd=("validation_loss", "std"),
        train_f1=("train_macro_f1", "mean"), train_f1_sd=("train_macro_f1", "std"),
        validation_f1=("validation_macro_f1", "mean"), validation_f1_sd=("validation_macro_f1", "std"),
    ).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.75))
    for axes, specs, title, ylabel in [
        (ax[0], [("train_loss", "train_loss_sd", "Training"), ("validation_loss", "validation_loss_sd", "Validation")],
         "(a) Training and validation loss", "Focal loss"),
        (ax[1], [("train_f1", "train_f1_sd", "Training"), ("validation_f1", "validation_f1_sd", "Validation")],
         "(b) Training and validation Macro-F1", "Macro-F1"),
    ]:
        for metric, sd, label in specs:
            x, y, s = summary["epoch"].to_numpy(), summary[metric].to_numpy(), summary[sd].to_numpy()
            line, = axes.plot(x, y, linewidth=1.7, marker="o", markersize=2.3, label=label)
            axes.fill_between(x, y-s, y+s, color=line.get_color(), alpha=.16)
        axes.set(xlabel="Epoch", ylabel=ylabel, title=title, xlim=(1, 20), xticks=[1, 5, 10, 15, 20])
        axes.grid(alpha=.2)
        axes.legend(frameon=False)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"training_validation_curves.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    history.to_csv(output / "training_histories_three_seeds.csv", index=False)
    summary.to_csv(output / "training_summary_mean_sd.csv", index=False)
    return summary


def representative_plates(split_manifest, output, dataset_root=None):
    """Regenerate the exact three manuscript images; fail rather than substitute one.

    Original cell125 selected the middle filename in each sorted training class.
    File availability is not allowed to change selection. dataset_root, when used,
    points to a directory containing Negative/, Positive/ and Uncertain/.
    """
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    data = pd.read_csv(split_manifest)
    if not {"filepath", "label", "split_extension"}.issubset(data):
        raise ValueError("Split manifest needs filepath, label and split_extension")
    train = data[data["split_extension"].astype(str).str.strip().eq("train")].copy()
    expected = {"Negative": "N32.jpg", "Positive": "P306.jpg", "Uncertain": "U307.jpg"}
    chosen = []
    for label, expected_name in expected.items():
        candidates = train[train["label"].astype(str).str.strip().eq(label)].sort_values("filepath")
        if candidates.empty:
            raise ValueError(f"No training records for {label}")
        original = Path(str(candidates.iloc[len(candidates)//2]["filepath"]))
        if original.name != expected_name:
            raise ValueError(f"Manifest selects {original.name} for {label}, but the manuscript used {expected_name}; audit the split instead of substituting an image")
        path = Path(dataset_root) / label / original.name if dataset_root else original
        if not path.is_file():
            raise FileNotFoundError(f"Missing original image: {path}")
        chosen.append((label, path))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    for k, (ax, (label, filename)) in enumerate(zip(axes, chosen)):
        with Image.open(filename) as source:
            ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
        ax.set_title(f"({chr(ord('a')+k)}) {label}")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=.5)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"dataset_representative_plates.{suffix}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    # Relative names avoid publishing machine-specific paths.
    manifest = pd.DataFrame({"class": [x[0] for x in chosen],
                             "source_file": [f"{label}/{path.name}" for label, path in chosen],
                             "subset": ["train"]*3})
    manifest.to_csv(output / "representative_images_manifest.csv", index=False)
    return manifest
