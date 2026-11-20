"""Audit images and prepare a portable copy of the original fixed study manifest."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urineculture.data import (ORIGINAL_SPLIT_SHA256, SPLIT_SEED, audit_images, create_fixed_split,
                               load_manifest, resolve_image_paths, sha256_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--existing-splits", type=Path, help="Preferred: the original saved extension split CSV.")
    source.add_argument("--original-csv", type=Path, help="Original 1500-row CSV, with original row order and filepath strings.")
    parser.add_argument("--image-root", type=Path, help="Directory containing Negative, Positive, and Uncertain folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/01_splits"))
    parser.add_argument("--skip-image-audit", action="store_true", help="Manifest preparation only; does not verify images.")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to preserve existing manifests.")
    if args.existing_splits:
        frame = load_manifest(args.existing_splits, args.image_root)
    else:
        frame = resolve_image_paths(create_fixed_split(args.original_csv), args.image_root)
    audit = None if args.skip_image_audit else audit_images(frame)
    portable = frame.copy()
    portable["filepath"] = [f"{row.label}/{row.filename}" for row in portable.itertuples(index=False)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "urocultivo_extension_split_v1.csv"
    portable.to_csv(destination, index=False)
    if audit is not None:
        audit.to_csv(args.output_dir / "image_audit.csv", index=False)
    table = portable.groupby(["split_extension", "label"]).size().rename("count")
    table.to_csv(args.output_dir / "split_class_counts.csv")
    source_path = args.existing_splits or args.original_csv
    metadata = {"source_csv_sha256": sha256_file(source_path), "output_csv_sha256": sha256_file(destination),
                "notebook_original_split_sha256": ORIGINAL_SPLIT_SHA256, "split_seed": SPLIT_SEED,
                "source_kind": "existing_study_manifest" if args.existing_splits else "recreated_from_original_csv",
                "sample_ids_preserved": True, "image_audit_performed": audit is not None,
                "note": "Paths relocated; output bytes/hash differ from original. No patient identifiers available."}
    (args.output_dir / "split_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {destination}. Image audit performed: {audit is not None}.")


if __name__ == "__main__":
    main()
