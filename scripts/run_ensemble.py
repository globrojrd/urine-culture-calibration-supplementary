"""Fit on development, or evaluate a saved ensemble policy."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("fit", "evaluate"):
        p = sub.add_parser(action)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        if action == "evaluate":
            p.add_argument("--frozen-policy", required=True)
    args = parser.parse_args()
    from urineculture.ensemble import fit_ensemble, evaluate_ensemble
    if args.action == "fit":
        state = fit_ensemble(args.manifest, args.output)
        print("Frozen development threshold:", state["policy"]["selected_threshold"])
    else:
        result = evaluate_ensemble(args.manifest, args.frozen_policy, args.output)
        print("Released:", result["number_of_releases"], "of", result["number_of_samples"])
        print("Unsafe releases:", result["unsafe_releases"])


if __name__ == "__main__":
    main()
