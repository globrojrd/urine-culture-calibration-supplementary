"""Recompute 100 development-only repartitions with frozen checkpoints."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urineculture.sensitivity import run_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="27-row CSV with architecture,loss,seed,calibration,threshold_selection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run_sensitivity(args.manifest, args.output))


if __name__ == "__main__":
    main()
