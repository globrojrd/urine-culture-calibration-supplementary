"""Stage fresh frozen-pipeline outputs for the six paired comparisons."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urineculture.statistics import read_predictions, align_comparator, validate_study_cohort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factorial", type=Path, required=True)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError("Choose an empty staging directory")
    base = read_predictions(a.ensemble)
    validate_study_cohort(base)
    frames = {"ensemble": base}
    for seed in (42, 123, 2026):
        for name, relative in [
            (f"ResNet_individual_seed_{seed}", f"resnet18/focal_loss/seed_{seed}/vector_scaling/final_test_predictions.csv"),
            (f"Swin_uncalibrated_seed_{seed}", f"swin_t/cross_entropy/seed_{seed}/uncalibrated/final_test_predictions.csv")]:
            frame = read_predictions(a.factorial / relative)
            validate_study_cohort(frame)
            frames[name] = align_comparator(base, frame)
    a.output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(a.output / f"{name}.csv", index=False)
    print("Staged seven aligned policies in", a.output)


if __name__ == "__main__":
    main()
