"""
Profile cuDNN optimizations for LINet3: channels_last + grouped conv.

Benchmarks three configurations to isolate the effect of each optimization:
1. BASELINE  — NCHW format, sequential per-stream conv (original)
2. CL_ONLY   — channels_last format, sequential per-stream conv
3. ALL_ON    — channels_last format + grouped cuDNN conv

Generates Chrome trace files and wall-clock timing for each.
Open traces in chrome://tracing or https://ui.perfetto.dev/ to compare.

Usage:
    python3 tests/benchmarks/profile_cudnn_optimizations.py [--mode all|baseline|cl_only|all_on] [--warmup 5] [--iters 3]

    # Quick comparison (wall-clock only, no trace files):
    python3 tests/benchmarks/profile_cudnn_optimizations.py --mode all --iters 0 --timing-iters 30

    # Only channels_last vs baseline:
    python3 tests/benchmarks/profile_cudnn_optimizations.py --mode baseline --iters 0 --timing-iters 30
    python3 tests/benchmarks/profile_cudnn_optimizations.py --mode cl_only --iters 0 --timing-iters 30
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda"
BATCH_SIZE = 16
INPUT_H, INPUT_W = 416, 544
NUM_CLASSES = 15
STREAM_INPUT_CHANNELS = [3, 1, 1]  # RGB, Depth, Orth


# ============================================================================
# Optimization toggle
# ============================================================================

def set_optimizations(grouped_conv: bool = True, channels_last: bool = True):
    """Toggle individual optimizations for A/B profiling.

    Args:
        grouped_conv: If True, use grouped cuDNN conv for uniform-shape layers.
                      If False, use sequential per-stream convolution.
        channels_last: If True, convert inputs to NHWC memory format.
                       If False, keep NCHW (original baseline).
    """
    import src.models.linear_integration.li_net3.conv as mod_conv
    mod_conv.USE_OPTIMIZED_OPS = grouped_conv
    mod_conv.USE_CHANNELS_LAST = channels_last


# ============================================================================
# Model + data setup
# ============================================================================

def create_model(channels_last: bool = True):
    """Create a fresh LINet3 ResNet-18 model in training mode."""
    from src.models.linear_integration.li_net3.li_net import li_resnet18

    model = li_resnet18(
        num_classes=NUM_CLASSES,
        stream_input_channels=STREAM_INPUT_CHANNELS,
        dropout_p=0.5,
        device=DEVICE,
        use_amp=True,
    )
    model.train()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def create_inputs(channels_last: bool = True):
    """Create synthetic input batch."""
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    rgb = torch.randn(BATCH_SIZE, 3, INPUT_H, INPUT_W, device=DEVICE).to(memory_format=mem_fmt)
    depth = torch.randn(BATCH_SIZE, 1, INPUT_H, INPUT_W, device=DEVICE).to(memory_format=mem_fmt)
    orth = torch.randn(BATCH_SIZE, 1, INPUT_H, INPUT_W, device=DEVICE).to(memory_format=mem_fmt)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
    return [rgb, depth, orth], targets


# ============================================================================
# Profiling
# ============================================================================

def run_profile(model, stream_batches, targets, trace_path: str, warmup: int, profile_iters: int):
    """Run torch.profiler and export Chrome trace."""
    criterion = nn.CrossEntropyLoss()

    # Warmup (lets cuDNN benchmark select algorithms)
    for _ in range(warmup):
        model.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(stream_batches)
            loss = criterion(outputs, targets)
        loss.backward()
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(profile_iters):
            model.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(stream_batches)
                loss = criterion(outputs, targets)
            loss.backward()
    torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"  Trace saved: {trace_path}")

    # Print summary table
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    return prof


def run_wall_clock(model, stream_batches, targets, warmup: int, iters: int):
    """Wall-clock timing with CUDA events."""
    criterion = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(stream_batches)
            loss = criterion(outputs, targets)
        loss.backward()
    torch.cuda.synchronize()

    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        model.zero_grad()
        torch.cuda.synchronize()
        start.record()
        with autocast(device_type="cuda"):
            outputs = model(stream_batches)
            loss = criterion(outputs, targets)
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    return mean_ms, min_ms, max_ms


# ============================================================================
# Main
# ============================================================================

def _clear_grouped_conv_buffers(model):
    """Clear any cached grouped conv buffers."""
    for module in model.modules():
        for attr in ("_grouped_weight_buffer", "_grouped_input_buffer"):
            if hasattr(module, attr):
                setattr(module, attr, None)


def _run_config(name, label, model, state_dict, targets, output_dir, drive_output_dir, args,
                grouped_conv, channels_last):
    """Run profiling + wall-clock for a single configuration."""
    print(f"\n{'='*70}")
    print(label)
    print(f"{'='*70}")

    set_optimizations(grouped_conv=grouped_conv, channels_last=channels_last)

    # Recreate model with correct memory format
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    else:
        model = model.to(memory_format=torch.contiguous_format)
    model.load_state_dict(state_dict)
    _clear_grouped_conv_buffers(model)

    # Create inputs matching the memory format
    stream_batches, _ = create_inputs(channels_last=channels_last)

    if args.iters > 0:
        trace_file = f"trace_{name}.json"
        trace_path = str(output_dir / trace_file)
        run_profile(model, stream_batches, targets, trace_path, args.warmup, args.iters)
        if drive_output_dir:
            shutil.copy2(trace_path, drive_output_dir / trace_file)
            print(f"  Copied to Drive: {drive_output_dir / trace_file}")

    mean_ms, min_ms, max_ms = run_wall_clock(
        model, stream_batches, targets, args.warmup, args.timing_iters
    )
    print(f"\n  Wall-clock (fwd+bwd): mean={mean_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")
    return {"mean_ms": mean_ms, "min_ms": min_ms, "max_ms": max_ms}


# Named configurations: (grouped_conv, channels_last, label)
CONFIGS = {
    "baseline": (False, False, "BASELINE (NCHW, sequential per-stream conv — original)"),
    "cl_only":  (False, True,  "CHANNELS_LAST ONLY (NHWC, sequential per-stream conv)"),
    "all_on":   (True,  True,  "ALL ON (NHWC + grouped cuDNN conv + BN-ReLU fusion)"),
}


def main():
    parser = argparse.ArgumentParser(description="Profile cuDNN optimizations for LINet3")
    parser.add_argument("--mode", choices=["all", "baseline", "cl_only", "all_on"], default="all",
                        help="Which config(s) to profile (default: all three)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations before profiling")
    parser.add_argument("--iters", type=int, default=3,
                        help="Iterations to profile (for trace files, 0=skip traces)")
    parser.add_argument("--timing-iters", type=int, default=20,
                        help="Iterations for wall-clock timing")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "tests" / "benchmarks"),
                        help="Directory for trace output files")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU machine.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Batch size: {BATCH_SIZE}, Input: {INPUT_H}x{INPUT_W}")
    print(f"Warmup: {args.warmup}, Profile iters: {args.iters}, Timing iters: {args.timing_iters}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # On Colab, also copy traces to Drive so they survive runtime disconnects
    drive_output_dir = None
    drive_mount = Path("/content/drive/MyDrive")
    if drive_mount.exists():
        drive_output_dir = drive_mount / "profiling_traces"
        drive_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Colab detected — traces will also be copied to: {drive_output_dir}")

    # Create model and save initial state for fair comparison across configs.
    print("\nCreating shared model...")
    set_optimizations(grouped_conv=True, channels_last=True)
    model = create_model(channels_last=True)
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    _, targets = create_inputs(channels_last=True)

    # Select configs to run
    if args.mode == "all":
        configs_to_run = ["baseline", "cl_only", "all_on"]
    else:
        configs_to_run = [args.mode]

    results = {}
    for config_name in configs_to_run:
        grouped_conv, channels_last, label = CONFIGS[config_name]
        results[config_name] = _run_config(
            config_name, label, model, state_dict, targets,
            output_dir, drive_output_dir, args,
            grouped_conv=grouped_conv, channels_last=channels_last,
        )

    del model
    torch.cuda.empty_cache()

    # Summary comparison
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")

        for name, r in results.items():
            _, _, label = CONFIGS[name]
            print(f"  {name:12s}: {r['mean_ms']:7.1f} ms  — {label}")

        if "baseline" in results:
            base = results["baseline"]["mean_ms"]
            for name, r in results.items():
                if name == "baseline":
                    continue
                delta = base - r["mean_ms"]
                speedup = base / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
                faster = "faster" if delta > 0 else "SLOWER"
                print(f"\n  {name} vs baseline: {delta:+.1f} ms ({faster}), {speedup:.3f}x")


if __name__ == "__main__":
    main()
