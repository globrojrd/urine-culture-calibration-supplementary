"""Fit development policies or evaluate frozen policies from exported logits."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urineculture.factorial import develop, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "evaluate"))
    parser.add_argument("--manifest", required=True, help="Experiment CSV with relative logit-table paths.")
    parser.add_argument("--output", required=True, help="Directory holding calibrators, thresholds, and results.")
    parser.add_argument("--allow-partial-factorial", action="store_true",
                        help="Allow fewer than 27 models; each included architecture/loss still needs all three seeds.")
    args = parser.parse_args()
    operation = develop if args.phase == "develop" else evaluate
    result = operation(args.manifest, args.output, not args.allow_partial_factorial)
    if args.phase == "develop":
        print("Development ranking frozen:", result["architecture"], result["loss"], result["calibration_method"])
    else:
        print(f"Evaluated {len(result)} frozen seed-level policies.")


if __name__ == "__main__":
    main()
