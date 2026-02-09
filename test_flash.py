import torch
import math
import sys
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_dot_product_attention
import torch.cuda.profiler as profiler

# --- Setup ---
# Note: Keep the sources as they are in your directory
flash_attn_lib = load(
    name="flash_attn_lib",
    sources=["flash_attn_kernel.cu", "wrapper.cpp"],
    extra_cuda_cflags=["-O3"]
)


def validate(custom_out, ref_out, atol=1e-3, rtol=1e-3):
    if custom_out.shape != ref_out.shape:
        return False, "Shape mismatch"
    if torch.isnan(custom_out).any():
        return False, "Custom output contains NaNs"

    is_close = torch.allclose(custom_out, ref_out, atol=atol, rtol=rtol)
    diff = (custom_out - ref_out).abs().max().item()
    return is_close, diff


def run_profile_mode():
    """
    Minimal execution path designed specifically for NCU/NSYS profiling.
    """
    B, nh, N, d = 4, 8, 1024, 64
    device = torch.device("cuda")
    dtype = torch.float16

    q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
    k = torch.randn(B, nh, N, d, device=device, dtype=dtype)
    v = torch.randn(B, nh, N, d, device=device, dtype=dtype)

    print(f"--- Profiling Mode (N={N}, d={d}) ---")

    # 1. Warmup: Ensure kernels are loaded and PTX is JITed
    for _ in range(5):
        _ = flash_attn_lib.forward(q, k, v)

    torch.cuda.synchronize()

    # 2. Profiler Start: NCU will trigger on this API call
    profiler.start()

    # We only run it once to prevent NCU from hanging on multi-pass collection
    out = flash_attn_lib.forward(q, k, v)

    torch.cuda.synchronize()
    profiler.stop()
    print("Profiling trigger complete.")


def run_benchmark_mode():
    """
    Standard sweep and validation mode.
    """
    B, nh = 4, 8
    device = torch.device("cuda")
    dtype = torch.float16
    seq_lengths = [1024, 2048]
    head_dims = [64, 128]

    print(f"{'SeqLen':>7} | {'d':>3} | {'Status':>7} | {'Custom(ms)':>10} | {'PyT(ms)':>10} | {'Speedup':>7}")
    print("-" * 70)

    for d in head_dims:
        for N in seq_lengths:
            q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            k = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            v = torch.randn(B, nh, N, d, device=device, dtype=dtype)

            with torch.no_grad():
                ref_out = scaled_dot_product_attention(q, k, v)
                custom_out = flash_attn_lib.forward(q, k, v)

            is_correct, max_err = validate(custom_out, ref_out)
            status = "PASS" if is_correct else "FAIL"

            # Quick Timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Custom
            start.record()
            for _ in range(20):
                flash_attn_lib.forward(q, k, v)
            end.record()
            torch.cuda.synchronize()
            t_custom = start.elapsed_time(end) / 20

            # PyTorch
            start.record()
            for _ in range(20):
                scaled_dot_product_attention(q, k, v)
            end.record()
            torch.cuda.synchronize()
            t_torch = start.elapsed_time(end) / 20

            print(
                f"{N:7d} | {d:3d} | {status:7s} | {t_custom:10.3f} | {t_torch:10.3f} | {t_torch/t_custom:7.2f}x")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--profile":
        run_profile_mode()
        sys.exit(0)
    else:
        run_benchmark_mode()
