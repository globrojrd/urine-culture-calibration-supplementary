"""Train the 27-model factorial design or the original focal-loss pilot."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Root containing the three class directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--architectures", nargs="+", choices=["resnet18", "efficientnetv2_s", "swin_t"],
                        default=["resnet18", "efficientnetv2_s", "swin_t"])
    parser.add_argument("--losses", nargs="+", choices=["cross_entropy", "label_smoothing", "focal_loss"],
                        default=["cross_entropy", "label_smoothing", "focal_loss"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42, 123, 2026], default=[42, 123, 2026])
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0.")
    parser.add_argument("--num-workers", type=int, default=2, help="Original protocol: 2. Changing this affects reproducibility.")
    parser.add_argument("--pilot", action="store_true", help="10-epoch ResNet-18 gamma pilot; main training uses fixed gamma=2.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned grid without importing PyTorch or training.")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative.")
    grid = [{"architecture": architecture, "loss": loss, "seed": seed}
            for architecture in args.architectures for loss in args.losses for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"pilot": args.pilot, "runs": "ResNet-18/seed42/gamma1,2,3" if args.pilot else grid,
                          "epochs": 10 if args.pilot else 20, "image_data_verified": False}, indent=2))
        return
    import pandas as pd
    import torch
    from urineculture.data import audit_images, load_manifest
    from urineculture.training import run_focal_pilot, train_one_run

    frame = load_manifest(args.split_csv, args.image_root)
    audit = audit_images(frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "training_input_image_audit.csv"
    if not audit_path.exists():
        audit.to_csv(audit_path, index=False)
    if args.pilot:
        result = run_focal_pilot(frame, args.output_dir, device, args.num_workers, args.split_csv)
        print(json.dumps(result, indent=2))
    else:
        rows = []
        for run in grid:
            rows.append(train_one_run(frame, args.output_dir, **run, device=device,
                                      num_workers=args.num_workers, split_manifest=args.split_csv))
        result_dir = args.output_dir / "06_results"
        result_dir.mkdir(parents=True, exist_ok=True)
        summary_path = result_dir / "training_runs_summary.csv"
        if summary_path.exists():
            previous = pd.read_csv(summary_path)
            rows = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
