"""Measure per-window inference latency of both streams (peer review item W6).

A deployed system receives one new window at a time, so everything here
is timed at batch size 1, end to end per window, on real harmonized
KFall windows:

  Prediction stream (ConvLSTM, TinyTransformer), 1.0 s windows, new window
  every 0.1 s  ->  real-time budget = 100 ms per window:
      aux-channel computation (accel magnitude, jerk, tilt)  +  model forward
  Detection stream (XGBoost), 2.0 s windows, new window every 1.0 s
  ->  budget = 1000 ms per window:
      54-feature extraction  +  XGBoost predict_proba

Each is run on: CPU with 1 thread (closest proxy for a phone / embedded
core available here), CPU with all threads, and (deep models) the GPU.
Reports median / p95 / p99 / mean, plus model size, parameter count and
FLOPs per window.

What this does NOT measure: the harmonization band-pass filter (the
offline pipeline uses zero-phase filtfilt over a whole trial, which a
live device cannot run; a causal filter costs microseconds per sample
and is not the bottleneck), sensor I/O, or BLE transfer. All numbers are
from the development laptop, not a wearable -- the report records the
exact hardware so they are not mistaken for on-device figures.

Usage:
    python scripts/benchmark_inference_latency.py
    python scripts/benchmark_inference_latency.py --n-iters 2000 --n-windows 300
Writes results/latency/latency_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.flop_counter import FlopCounterMode

from detection.features import FEATURE_NAMES, compute_window_features
from detection.model import load_model as load_detection_model
from prediction.dataset import load_window
from prediction.features import augment_window
from prediction.predict import load_prediction_model
from shared.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
PREDICTION_BUDGET_MS = 100.0   # prediction stride 0.1 s
DETECTION_BUDGET_MS = 1000.0   # detection stride 1.0 s


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _sample_raw_windows(manifest_path: Path, n: int, length: int, seed: int = 0) -> list[np.ndarray]:
    """`n` real (length, 6) raw harmonized windows from random KFall trials."""
    trials = load_manifest(manifest_path)
    trials = trials[(trials["dataset"] == "kfall") & (trials["duration_s"] * trials["sample_rate_hz"] > length)]
    rng = np.random.default_rng(seed)
    rows = trials.sample(n=n, replace=True, random_state=seed)
    cache: dict = {}
    windows = []
    for _, r in rows.iterrows():
        n_samples = round(r["duration_s"] * r["sample_rate_hz"])
        start = int(rng.integers(0, n_samples - length))
        row = pd.Series({"harmonized_path": r["harmonized_path"], "start_frame": start, "end_frame": start + length})
        windows.append(load_window(row, length, signal_cache=cache).astype(np.float32))
    return windows


def _time_ms(fn, inputs: list, n_iters: int, n_warmup: int, sync=None) -> dict:
    for i in range(n_warmup):
        fn(inputs[i % len(inputs)])
    if sync:
        sync()
    times = np.empty(n_iters)
    for i in range(n_iters):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter_ns()
        fn(x)
        if sync:
            sync()
        times[i] = (time.perf_counter_ns() - t0) / 1e6
    return {
        "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)), "mean_ms": float(times.mean()),
        "n_iters": n_iters,
    }


def _flops(model: torch.nn.Module, x: torch.Tensor) -> int:
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            model(x)
    return int(counter.get_total_flops())


def _lstm_flops(model: torch.nn.Module, seq_len_after_conv: int) -> int:
    """FlopCounterMode does not count the fused LSTM kernel; add it
    analytically: per step, 4 gates x (input + hidden) matmuls = 2*4*h*(i+h)
    FLOPs, plus small elementwise terms (ignored)."""
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            h, i = m.hidden_size, m.input_size
            total += 2 * 4 * h * (i + h) * seq_len_after_conv * m.num_layers
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "harmonized" / "manifest.parquet"))
    parser.add_argument("--prediction-report", default=str(REPO_ROOT / "results" / "prediction_model" / "evaluation_report.json"))
    parser.add_argument("--detection-model", default=str(REPO_ROOT / "results" / "detection_model" / "xgboost_model.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "latency" / "latency_report.json"))
    parser.add_argument("--n-windows", type=int, default=200)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--n-warmup", type=int, default=100)
    parser.add_argument("--pin-cpu", type=int, default=None,
                        help="Pin the process to this logical CPU (e.g. a performance core) so single-thread "
                             "timings are not affected by the OS migrating between P- and E-cores. "
                             "'all threads' runs are then limited to that core too, so use without pinning for those.")
    args = parser.parse_args()

    if args.pin_cpu is not None:
        import psutil
        psutil.Process().cpu_affinity([args.pin_cpu])
    n_threads_all = 1 if args.pin_cpu is not None else os.cpu_count()
    report = {
        "hardware": {
            "cpu": _cpu_name(), "logical_cpus": n_threads_all, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "xgboost": xgb.__version__,
            "note": "Development laptop (plugged in). Not on-device wearable/mobile measurements.",
        },
        "protocol": {
            "batch_size": 1, "n_iters": args.n_iters, "n_warmup": args.n_warmup,
            "inputs": f"{args.n_windows} real harmonized KFall windows, cycled",
            "pinned_cpu": args.pin_cpu,
            "prediction_budget_ms": PREDICTION_BUDGET_MS, "detection_budget_ms": DETECTION_BUDGET_MS,
        },
        "prediction": {}, "detection": {},
    }

    print(f"CPU: {report['hardware']['cpu']} ({n_threads_all} logical)  GPU: {report['hardware']['gpu']}")
    print("Sampling real windows...")
    pred_raw = _sample_raw_windows(Path(args.manifest), args.n_windows, 100)
    det_raw = _sample_raw_windows(Path(args.manifest), args.n_windows, 200, seed=1)

    # ---------------- prediction stream
    pred_report = json.loads(Path(args.prediction_report).read_text())
    for entry in pred_report["architectures"]:
        arch, path = entry["architecture"], Path(entry["model_path"])
        model = load_prediction_model(path).eval()
        x0 = torch.from_numpy(augment_window(pred_raw[0]).T[np.newaxis].astype(np.float32))
        flops = _flops(model, x0)
        if any(isinstance(m, torch.nn.LSTM) for m in model.modules()):
            with torch.no_grad():
                seq = model.conv_stack(x0).shape[-1]   # LSTM runs on the pooled conv output (12 steps), not 100 samples
            flops += _lstm_flops(model, seq)
        r = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_kb": round(path.stat().st_size / 1024, 1),
            "flops_per_window": flops,
        }
        print(f"\n[{arch}] params={r['parameters']:,}  checkpoint={r['checkpoint_kb']} KB  FLOPs/window={flops/1e6:.2f} M")

        def aux(raw):
            return augment_window(raw)
        r["aux_channels_cpu"] = _time_ms(aux, pred_raw, args.n_iters, args.n_warmup)

        aug = [torch.from_numpy(augment_window(w).T[np.newaxis].copy().astype(np.float32)) for w in pred_raw]
        for label, threads in (("cpu_1_thread", 1), ("cpu_all_threads", n_threads_all)):
            torch.set_num_threads(threads)
            m = model.to("cpu")
            with torch.no_grad():
                r[f"model_{label}"] = _time_ms(lambda x: m(x), aug, args.n_iters, args.n_warmup)
        if torch.cuda.is_available():
            torch.set_num_threads(n_threads_all)
            mg = model.to("cuda")
            aug_gpu = [a.to("cuda") for a in aug]
            with torch.no_grad():
                r["model_gpu"] = _time_ms(lambda x: mg(x), aug_gpu, args.n_iters, args.n_warmup, sync=torch.cuda.synchronize)
            model.to("cpu")

        # End-to-end on 1 CPU thread: aux + forward, the realistic single-core path.
        torch.set_num_threads(1)

        def e2e(raw):
            x = torch.from_numpy(augment_window(raw).T[np.newaxis].copy().astype(np.float32))
            with torch.no_grad():
                return model(x)
        r["end_to_end_cpu_1_thread"] = _time_ms(e2e, pred_raw, args.n_iters, args.n_warmup)
        r["budget_utilisation_cpu_1_thread"] = r["end_to_end_cpu_1_thread"]["p99_ms"] / PREDICTION_BUDGET_MS
        torch.set_num_threads(n_threads_all)
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k:28s} median={v['median_ms']:.3f} ms  p95={v['p95_ms']:.3f}  p99={v['p99_ms']:.3f}")
        report["prediction"][arch] = r

    # ---------------- detection stream
    det_model = load_detection_model(Path(args.detection_model))
    booster = det_model.get_booster()
    r = {
        "n_trees": len(booster.get_dump()),
        "model_file_kb": round(Path(args.detection_model).stat().st_size / 1024, 1),
        "n_features": len(FEATURE_NAMES),
    }
    print(f"\n[xgboost detection] trees={r['n_trees']}  model={r['model_file_kb']} KB  features={r['n_features']}")
    r["features_cpu"] = _time_ms(lambda w: compute_window_features(w, sample_rate_hz=100.0), det_raw, args.n_iters, args.n_warmup)
    feats = [np.array([[f[n] for n in FEATURE_NAMES]], dtype=np.float32)
             for f in (compute_window_features(w, sample_rate_hz=100.0) for w in det_raw)]
    for label, threads in (("cpu_1_thread", 1), ("cpu_all_threads", n_threads_all)):
        det_model.set_params(n_jobs=threads)
        r[f"model_{label}"] = _time_ms(lambda X: det_model.predict_proba(X), feats, args.n_iters, args.n_warmup)
    det_model.set_params(n_jobs=1)

    def det_e2e(raw):
        f = compute_window_features(raw, sample_rate_hz=100.0)
        return det_model.predict_proba(np.array([[f[n] for n in FEATURE_NAMES]], dtype=np.float32))
    r["end_to_end_cpu_1_thread"] = _time_ms(det_e2e, det_raw, args.n_iters, args.n_warmup)
    r["budget_utilisation_cpu_1_thread"] = r["end_to_end_cpu_1_thread"]["p99_ms"] / DETECTION_BUDGET_MS
    for k, v in r.items():
        if isinstance(v, dict):
            print(f"  {k:28s} median={v['median_ms']:.3f} ms  p95={v['p95_ms']:.3f}  p99={v['p99_ms']:.3f}")
    report["detection"]["xgboost"] = r

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
