import torch
import math
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_dot_product_attention

# --- Setup ---
flash_attn_lib = load(
    name="flash_attn_lib",
    sources=["flash_attn_kernel.cu", "wrapper.cpp"],
    extra_cuda_cflags=["-O3"]
)


def benchmark_stats(fn):
    # Warmup
    torch.cuda.synchronize()
    try:
        _ = fn()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Kernel execution failed: {e}")
        return 0.0, 0.0

    # Time Measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(50):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    # Memory Measurement
    torch.cuda.reset_peak_memory_stats()
    fn()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / 50
    return avg_ms, peak_mem


def validate(custom_out, ref_out, atol=1e-5, rtol=1e-4):
    if custom_out.shape != ref_out.shape:
        return False, "Shape mismatch"

    # Check for NaNs first - common in Flash Attention if softmax scaling is wrong
    if torch.isnan(custom_out).any():
        return False, "Custom output contains NaNs"

    diff = (custom_out - ref_out).abs()
    is_close = torch.allclose(custom_out, ref_out, atol=atol, rtol=rtol)
    max_diff = diff.max().item()
    return is_close, max_diff


def run_sweep():
    B, nh = 4, 8
    device = torch.device("cuda")

    dtype = torch.float16

    seq_lengths = [1024, 2048, 4096]
    head_dims = [64, 128]

    print(f"{'SeqLen':>7} | {'d':>3} | {'Custom(ms)':>10} | {'PyT(ms)':>10} | {'Speedup':>7} | {'PyT Mem(MB)':>12}")
    print("-" * 75)

    for d in head_dims:
        for N in seq_lengths:
            q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            k = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            v = torch.randn(B, nh, N, d, device=device, dtype=dtype)

            with torch.no_grad():
                ref_out = scaled_dot_product_attention(q, k, v)
                custom_out = flash_attn_lib.forward(q, k, v)

            is_correct, max_err = validate(
                custom_out, ref_out, atol=1e-3, rtol=1e-3)
            status = "PASS" if is_correct else f"FAIL (Max Diff: {max_err:.6f})"
            print(status)

            try:
                # Custom Kernel
                t_custom, _ = benchmark_stats(
                    lambda: flash_attn_lib.forward(q, k, v))

                # PyTorch SDPA
                t_torch, mem_torch = benchmark_stats(
                    lambda: scaled_dot_product_attention(q, k, v))

                speedup = t_torch / t_custom
                print(
                    f"{N:7d} | {d:3d} | {t_custom:10.3f} | {t_torch:10.3f} | {speedup:7.2f}x | {mem_torch:12.2f}")

            except Exception as e:
                print(f"{N:7d} | {d:3d} | Error: {str(e)[:25]}")


if __name__ == "__main__":
    run_sweep()
