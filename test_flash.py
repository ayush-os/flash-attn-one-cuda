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
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    # Time Measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(50):
        fn()
    end_event.record()

    # Memory Measurement
    torch.cuda.reset_peak_memory_stats()
    fn()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / 50
    return avg_ms, peak_mem


def run_sweep():
    B, nh = 4, 8
    device = torch.device("cuda")

    dtype = torch.float32

    seq_lengths = [1024, 2048, 4096]
    head_dims = [64, 128]

    print(f"{'SeqLen':>7} | {'d':>3} | {'Custom(ms)':>10} | {'PyT(ms)':>10} | {'Speedup':>7} | {'PyT Mem(MB)':>12}")
    print("-" * 75)

    for d in head_dims:
        for N in seq_lengths:
            q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            k = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            v = torch.randn(B, nh, N, d, device=device, dtype=dtype)

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
