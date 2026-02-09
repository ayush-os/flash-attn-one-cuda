import torch
import math
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_dot_product_attention

# Load your custom kernel
flash_attn_lib = load(
    name="flash_attn_lib",
    sources=["flash_attn_kernel.cu", "wrapper.cpp"],
    extra_cuda_cflags=["-O3"]
)


def benchmark_fn(fn, warmup=20, iterations=100):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        fn()
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iterations


def run_sweep():
    # Fixed parameters
    B, nh = 4, 8
    device = torch.device("cuda")
    dtype = torch.float16  # FlashAttn is usually optimized for FP16

    # Sweep parameters
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    head_dims = [64, 128]

    print(f"{'SeqLen':>8} | {'HeadDim':>8} | {'Custom (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>8}")
    print("-" * 65)

    for d in head_dims:
        for N in seq_lengths:
            # Initialize tensors
            q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            k = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            v = torch.randn(B, nh, N, d, device=device, dtype=dtype)

            # Define functions to benchmark
            def fn_custom(): return flash_attn_lib.forward(q, k, v)
            def fn_pytorch(): return scaled_dot_product_attention(q, k, v)

            # Measure
            try:
                t_custom = benchmark_fn(fn_custom)
                t_torch = benchmark_fn(fn_pytorch)
                speedup = t_torch / t_custom

                print(
                    f"{N:8d} | {d:8d} | {t_custom:12.4f} | {t_torch:12.4f} | {speedup:7.2f}x")
            except Exception as e:
                print(f"{N:8d} | {d:8d} | Error: {str(e)[:20]}...")


if __name__ == "__main__":
    run_sweep()
