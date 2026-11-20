"""Recreate the two reviewer figures from the original measured inputs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urineculture.figures import training_curves, representative_plates, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--project", type=Path, help="Original Drive experiment root")
    sources.add_argument("--inputs", type=Path, help="Exported portable data directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        help="Directory containing Negative/, Positive/, Uncertain/; omit to use manifest paths")
    parser.add_argument("--only", choices=("both", "curves", "plates"), default="both")
    args = parser.parse_args()
    source = args.project or args.inputs
    if args.output.resolve() == source.resolve() or source.resolve() in args.output.resolve().parents:
        raise ValueError("Use an output directory outside the original experiment folder")
    if args.only in ("both", "curves"):
        histories = ({s: args.project / f"08_logs/resnet18/focal_loss/gamma_2/seed_{s}/training_history.csv" for s in SEEDS}
                     if args.project else {s: args.inputs / f"histories/resnet18_focal_seed_{s}.csv" for s in SEEDS})
        training_curves(histories, args.output)
    if args.only in ("both", "plates"):
        if args.inputs and args.dataset_root is None:
            raise ValueError("Portable image manifests require --dataset-root with the original images")
        split = (args.project / "01_splits/urocultivo_extension_split_v1.csv" if args.project
                 else args.inputs / "splits/urocultivo_extension_split_v1.csv")
        representative_plates(split, args.output, args.dataset_root)
    print(f"Generated measured figures in {args.output}; inspect the image panels before submission")


if __name__ == "__main__":
    main()
