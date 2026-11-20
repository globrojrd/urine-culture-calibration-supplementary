"""Dataset audit, fixed image-level split, and loaders from notebook cells 61-63.

Cell numbers are zero-based. Patient/specimen grouping is unavailable. An image
split and a byte-hash audit do not establish patient-level independence.
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABEL_MAP = {"Negative": 0, "Positive": 1, "Uncertain": 2}
SPLITS = ("train", "model_validation", "calibration", "threshold_selection", "final_test")
SEEDS = (42, 123, 2026)
SPLIT_SEED = 2607
ORIGINAL_SPLIT_SHA256 = "3e8e49d6f7ec2bbf30d36d0795bee99c5261bae5d9318276182d56d75ce5055c"
EXPECTED_COUNTS = {
    "train": (300, 299, 301),
    "model_validation": (50, 50, 50),
    "calibration": (50, 50, 50),
    "threshold_selection": (50, 49, 51),
    "final_test": (50, 50, 50),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(frame, require_study_counts=True):
    required = {"filepath", "filename", "label", "sample_id", "split_extension"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing values.")
    if frame["sample_id"].duplicated().any() or frame["filepath"].duplicated().any():
        raise ValueError("Repeated sample IDs or image paths in manifest.")
    if frame.duplicated(["label", "filename"]).any():
        raise ValueError("Repeated class/filename pairs cannot be safely relocated.")
    if set(frame["label"]) != set(LABEL_MAP) or set(frame["split_extension"]) != set(SPLITS):
        raise ValueError("Manifest must contain the three study classes and five study splits.")
    if require_study_counts:
        table = pd.crosstab(frame["split_extension"], frame["label"])
        for split, expected in EXPECTED_COUNTS.items():
            actual = tuple(int(table.loc[split, label]) for label in LABEL_MAP)
            if actual != expected:
                raise ValueError(f"Unexpected {split} class counts: {actual}; expected {expected}.")
    return frame


def create_fixed_split(original_csv):
    """Reproduce cell 62; input CSV row order and original path strings matter.

    Prefer the already saved study manifest. Do not rebuild an original CSV by
    directory traversal: even the same images in a new row order change the split.
    """
    from sklearn.model_selection import train_test_split

    original = pd.read_csv(original_csv)
    columns = ["filepath", "filename", "label"]
    if not set(columns).issubset(original.columns):
        raise ValueError(f"Original CSV requires {columns}.")
    frame = original[columns].copy()
    if len(frame) != 1500 or frame.isna().any().any():
        raise ValueError("Expected 1500 complete rows from the original CSV.")
    if set(frame["label"]) != set(LABEL_MAP):
        raise ValueError("Unexpected class names in original CSV.")
    frame["sample_id"] = frame["filepath"].map(
        lambda path: hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    )
    train, remainder = train_test_split(
        frame, test_size=0.4, random_state=SPLIT_SEED, shuffle=True, stratify=frame["label"]
    )
    development, evaluation = train_test_split(
        remainder, test_size=0.5, random_state=SPLIT_SEED + 1,
        shuffle=True, stratify=remainder["label"]
    )
    validation, calibration = train_test_split(
        development, test_size=0.5, random_state=SPLIT_SEED + 2,
        shuffle=True, stratify=development["label"]
    )
    threshold, test = train_test_split(
        evaluation, test_size=0.5, random_state=SPLIT_SEED + 3,
        shuffle=True, stratify=evaluation["label"]
    )
    parts = []
    for name, part in zip(SPLITS, (train, validation, calibration, threshold, test)):
        part = part.copy()
        part["split_extension"] = name
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["split_extension"] = pd.Categorical(result["split_extension"], categories=SPLITS, ordered=True)
    result = result.sort_values(["split_extension", "label", "filename"], kind="stable").reset_index(drop=True)
    result["split_extension"] = result["split_extension"].astype(str)
    return validate_manifest(result)


def resolve_image_paths(frame, image_root=None):
    """Relocate files without changing sample IDs or row order.

    With image_root, use image_root/<label>/<filename>. This allows publishing
    relative paths without disclosing the original machine's directory layout.
    """
    result = frame.copy()
    if image_root is not None:
        root = Path(image_root).expanduser().resolve()
        for row in result.itertuples(index=False):
            if Path(str(row.filename)).name != str(row.filename):
                raise ValueError("filename must be a basename, not a directory path.")
        result["filepath"] = [str(root / row.label / row.filename) for row in result.itertuples(index=False)]
    return result


def load_manifest(path, image_root=None, require_study_counts=True):
    frame = pd.read_csv(path, dtype={"sample_id": str})
    validate_manifest(frame, require_study_counts)
    return resolve_image_paths(frame, image_root)


def audit_images(frame):
    """Read every image and hash original file bytes; fail on missing/corrupt/duplicate files."""
    records = []
    for row in frame.itertuples(index=False):
        path = Path(row.filepath)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
        records.append({"sample_id": row.sample_id, "filename": row.filename,
                        "label": row.label, "split_extension": row.split_extension,
                        "sha256": sha256_file(path), "width": width, "height": height, "mode": mode})
    audit = pd.DataFrame(records)
    if audit["sha256"].duplicated().any():
        raise ValueError("Byte-identical image files found; resolve duplicates before training.")
    return audit


def set_global_seed(seed):
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_transforms(training=False):
    from torchvision import transforms as T

    items = [T.Resize((224, 224), antialias=True)]
    if training:
        items += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=15)]
    items += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(items)


class UrineCultureDataset:
    """Map-style PyTorch dataset; no image is changed or synthesized."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        import torch

        row = self.dataframe.iloc[index]
        with Image.open(row["filepath"]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return {"image": image, "label": torch.tensor(LABEL_MAP[row["label"]], dtype=torch.long),
                "sample_id": str(row["sample_id"]), "filepath": str(row["filepath"])}


def create_dataloaders(frame, seed, batch_size=32, num_workers=2):
    """Keep the original shared generator, split order, and persistent workers."""
    import torch
    from torch.utils.data import DataLoader

    set_global_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in SPLITS:
        dataset = UrineCultureDataset(frame[frame["split_extension"] == split], make_transforms(split == "train"))
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False, worker_init_fn=seed_worker,
            generator=generator, persistent_workers=(num_workers > 0)
        )
    return loaders
