"""Fresh-run training extracted from notebook cells 65-77 (zero-based).

This module preserves the observed implementation, including the fixed divisor
on the final incomplete gradient-accumulation group. Historical interrupted-run
recovery is deliberately not presented as exactly reproducible: the notebook did
not save DataLoader generator or persistent-worker RNG states.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torchvision import __version__ as torchvision_version
from torchvision import models

from .data import LABEL_MAP, create_dataloaders, set_global_seed, sha256_file

ARCHITECTURES = ("resnet18", "efficientnetv2_s", "swin_t")
LOSSES = ("cross_entropy", "label_smoothing", "focal_loss")


def create_model(architecture, pretrained=True):
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif architecture == "efficientnetv2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    elif architecture == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, 3)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


class MulticlassFocalLoss(nn.Module):
    """Original mean multiclass focal loss, no alpha weighting or clipping."""

    def __init__(self, gamma=2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative.")
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = torch.exp(log_probabilities)
        target_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-(1.0 - target_probabilities).pow(self.gamma) * target_log_probabilities).mean()


def create_loss(loss, gamma=2.0):
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if loss == "focal_loss":
        return MulticlassFocalLoss(gamma)
    raise ValueError(f"Unknown loss: {loss}")


def run_epoch(model, loader, criterion, device, optimizer=None, accumulation_steps=1):
    training = optimizer is not None
    model.train(training)
    running_loss, all_targets, all_predictions = 0.0, [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if training else torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            raw_loss = criterion(logits, targets)
            if not torch.isfinite(raw_loss):
                raise RuntimeError("Non-finite loss.")
            if training:
                # Preserve cells 72-77: divide by two even for the last partial
                # accumulation group. This is not an exact batch-32 gradient.
                (raw_loss / accumulation_steps).backward()
                if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            running_loss += raw_loss.item() * images.size(0)
            all_targets.extend(targets.detach().cpu().numpy())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
    metrics = {"loss": float(running_loss / len(loader.dataset)),
               "accuracy": float(accuracy_score(all_targets, all_predictions)),
               "macro_f1": float(f1_score(all_targets, all_predictions, average="macro", zero_division=0))}
    if not training:
        precision, recall, class_f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, labels=[0, 1, 2], zero_division=0
        )
        for index, label in enumerate(LABEL_MAP):
            for key, values in (("precision", precision), ("recall", recall), ("f1", class_f1), ("support", support)):
                metrics[f"{label.lower()}_{key}"] = int(values[index]) if key == "support" else float(values[index])
    return metrics


def experiment_subpath(architecture, loss, seed, gamma=2.0):
    path = Path(architecture) / loss
    if loss == "focal_loss":
        path /= f"gamma_{gamma:g}"
    return path / f"seed_{seed}"


def _save_checkpoint(path, model, optimizer, epoch, best_score, configuration):
    torch.save({"epoch": int(epoch), "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_macro_f1": float(best_score), "configuration": configuration,
                "label_map": LABEL_MAP}, path)


def _save_curves(history, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for metric, name in (("loss", "loss_curve.png"), ("macro_f1", "macro_f1_curve.png")):
        figure, axis = plt.subplots(figsize=(8, 5))
        for prefix, label in (("train", "Training"), ("validation", "Validation")):
            axis.plot(history["epoch"], history[f"{prefix}_{metric}"], label=label)
        axis.set(xlabel="Epoch", ylabel="Loss" if metric == "loss" else "Macro-F1")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / name, dpi=200)
        plt.close(figure)


def train_one_run(frame, output_dir, architecture, loss, seed, device,
                  num_workers=2, gamma=2.0, pilot=False, split_manifest=None):
    """Run all 20 epochs (10 for the focal pilot), selecting earliest max-F1 epoch.

    Existing output is never overwritten or implicitly resumed. If interrupted,
    preserve that directory and choose a fresh output directory for a new run.
    """
    if architecture not in ARCHITECTURES or loss not in LOSSES:
        raise ValueError("Unsupported architecture/loss.")
    if pilot and (architecture != "resnet18" or loss != "focal_loss" or seed != 42):
        raise ValueError("The original pilot uses ResNet-18/focal, seed 42 only.")
    epochs = 10 if pilot else 20
    batch_size, accumulation_steps = (32, 1) if architecture == "resnet18" else (16, 2)
    output_dir = Path(output_dir)
    subpath = experiment_subpath(architecture, loss, seed, gamma)
    checkpoint_dir = output_dir / "02_checkpoints" / subpath
    log_dir = output_dir / "08_logs" / subpath
    figure_dir = output_dir / "07_figures" / "training_curves" / subpath
    for directory in (checkpoint_dir, log_dir, figure_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Run output already exists: {directory}. Use a fresh output directory.")
    for directory in (checkpoint_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    # Loader construction precedes model construction, exactly as in the notebook.
    loaders = create_dataloaders(frame, seed, batch_size, num_workers)
    model = create_model(architecture).to(device)
    criterion = create_loss(loss, gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    configuration = {
        "architecture": architecture, "loss": loss, "seed": seed, "number_of_epochs": epochs,
        "learning_rate": 1e-4, "weight_decay": 1e-4, "optimizer": "AdamW", "betas": [0.9, 0.999],
        "label_smoothing": 0.1 if loss == "label_smoothing" else 0.0,
        "focal_gamma": gamma if loss == "focal_loss" else None,
        "physical_batch_size": batch_size, "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size_nominal": batch_size * accumulation_steps,
        "incomplete_accumulation_group": "fixed divisor preserved from original notebook",
        "num_workers": num_workers, "persistent_workers": num_workers > 0, "image_size": 224,
        "pretrained_weights": "torchvision DEFAULT", "checkpoint_selection_metric": "model_validation_macro_f1",
        "checkpoint_tie_break": "earliest epoch (strict improvement)", "pilot": pilot,
        "train_samples": len(loaders["train"].dataset), "model_validation_samples": len(loaders["model_validation"].dataset),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest else None,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision_version, "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__, "cuda": torch.version.cuda, "device": str(device)},
    }
    (log_dir / "run_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    history, best_score, best_epoch = [], -np.inf, None
    start = time.time()
    best_path = checkpoint_dir / "best_model_by_macro_f1.pth"
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, accumulation_steps)
        validation_metrics = run_epoch(model, loaders["model_validation"], criterion, device)
        improved = validation_metrics["macro_f1"] > best_score
        if improved:
            best_score, best_epoch = validation_metrics["macro_f1"], epoch
            _save_checkpoint(best_path, model, optimizer, epoch, best_score, configuration)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"validation_{k}": v for k, v in validation_metrics.items()},
               "epoch_duration_seconds": time.time() - epoch_start, "is_best_checkpoint": improved}
        history.append(row)
        pd.DataFrame(history).to_csv(log_dir / "training_history.csv", index=False)
        _save_checkpoint(checkpoint_dir / "last_epoch_model.pth", model, optimizer, epoch, best_score, configuration)
        print(f"{architecture}/{loss}/seed_{seed} epoch {epoch:02d}/{epochs}: "
              f"train loss {train_metrics['loss']:.4f}, validation F1 {validation_metrics['macro_f1']:.4f}", flush=True)
    elapsed = time.time() - start
    history_frame = pd.DataFrame(history)
    best_metrics = history[best_epoch - 1]
    metadata = {"configuration": configuration, "best_epoch": best_epoch,
                "best_validation_macro_f1": best_score, "best_epoch_metrics": best_metrics,
                "total_training_seconds": elapsed, "status": "completed_fresh_run"}
    (log_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _save_curves(history_frame, figure_dir)
    return {"architecture": architecture, "loss": loss, "gamma": gamma if loss == "focal_loss" else None,
            "seed": seed, "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
            "best_validation_accuracy": best_metrics["validation_accuracy"],
            "best_validation_loss": best_metrics["validation_loss"], "training_seconds": elapsed,
            "checkpoint_path": str(best_path), "history_path": str(log_dir / "training_history.csv")}


def run_focal_pilot(frame, output_dir, device, num_workers=2, split_manifest=None):
    """Cell 70: compare gamma 1/2/3 with seed42; never use other data splits."""
    root = Path(output_dir) / "focal_gamma_pilot"
    rows = [train_one_run(frame, root, "resnet18", "focal_loss", 42, device,
                          num_workers=num_workers, gamma=gamma, pilot=True, split_manifest=split_manifest)
            for gamma in (1.0, 2.0, 3.0)]
    summary = pd.DataFrame(rows).sort_values(
        ["best_validation_macro_f1", "best_validation_loss", "gamma"], ascending=[False, True, True]
    ).reset_index(drop=True)
    summary.to_csv(root / "gamma_selection_summary.csv", index=False)
    selected = {"selected_gamma": float(summary.iloc[0]["gamma"]), "seed": 42, "epochs": 10,
                "selection": ["maximum validation Macro-F1", "minimum validation loss", "smaller gamma"]}
    (root / "selected_gamma.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected
