"""Architecture timing from cell101; random weights and synthetic input, not accuracy.

Requires CUDA. Fresh hardware timings will differ from recorded NVIDIA L4 values.
Measures forward passes only; excludes image loading. Does not reproduce the
recorded training times, which require the original 27 training logs.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from urineculture.training import create_model
    if not torch.cuda.is_available():
        raise RuntimeError("This historical benchmark requires a CUDA GPU")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty benchmark-output directory")
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.models = torch.nn.ModuleList([create_model("resnet18", pretrained=False) for _ in range(3)])
            self.register_buffer("scales", torch.tensor([1.58202608, .73603594, .89579428]))
            self.register_buffer("biases", torch.tensor([.22879514, -.33283879, .10404365]))
        def forward(self, x):
            z = torch.stack([m(x) for m in self.models]).mean(0)
            return torch.softmax(z*self.scales + self.biases, dim=1)
    rows = []
    for name in ("resnet18", "efficientnetv2_s", "swin_t", "ensemble"):
        model = (Ensemble() if name == "ensemble" else create_model(name, pretrained=False)).cuda().eval()
        x = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        with torch.inference_mode():
            for _ in range(50):
                model(x)
            torch.cuda.synchronize()
            for _ in range(300):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
        batch = 32 if name == "resnet18" else 16
        accumulation = 1 if name == "resnet18" else 2
        rows.append({"system": name, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "optimizer_steps_per_run": None if name == "ensemble" else math.ceil(math.ceil(900/batch)/accumulation)*20,
                     "mean_latency_ms": float(np.mean(timings)), "median_latency_ms": float(np.median(timings)),
                     "sd_latency_ms": float(np.std(timings, ddof=1)), "p95_latency_ms": float(np.percentile(timings, 95)),
                     "peak_allocated_gpu_memory_mb": torch.cuda.max_memory_allocated()/1024**2})
        del model, x
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "inference_benchmark.csv", index=False)
    (args.output / "environment.json").write_text(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "warmup": 50, "measured": 300,
        "weights": "random, architecture benchmark as in source cell101", "input": "synthetic batch1 RGB224x224"}, indent=2))
    print("Fresh architecture benchmark saved in", args.output)


if __name__ == "__main__":
    main()
