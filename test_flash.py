import torch
import math
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_dot_product_attention

# --- Setup ---
# Compiling with nvcc flags for architecture (adjust sm_80 for A100, sm_75 for T4/RTX20xx, etc.)
flash_attn_lib = load(
    name="flash_attn_lib",
    sources=["flash_attn_kernel.cu", "wrapper.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)


def benchmark_stats(fn):
    torch.cuda.synchronize()
    try:
        fn()  # Warmup
        torch.cuda.synchronize()
    except Exception as e:
        print(f"FAILED: {e}")
        return 0.0, 0.0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    start_event.record()
    for _ in range(50):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    avg_ms = start_event.elapsed_time(end_event) / 50
    return avg_ms, peak_mem


def validate(custom_out, ref_out):
    # FP16 requires relaxed tolerances
    atol = 1e-3
    rtol = 1e-3

    if custom_out.shape != ref_out.shape:
        return False, "Shape mismatch"
    if torch.isnan(custom_out).any():
        return False, "NaNs in output"

    diff = (custom_out - ref_out).abs()
    max_diff = diff.max().item()
    is_close = torch.allclose(custom_out, ref_out, atol=atol, rtol=rtol)
    return is_close, max_diff


def run_sweep():
    B, nh = 4, 8
    device = torch.device("cuda")

    # UPDATED: Use Half precision
    dtype = torch.float16

    seq_lengths = [1024, 2048, 4096]
    head_dims = [64, 128]

    print(f"{'SeqLen':>7} | {'d':>3} | {'Custom(ms)':>10} | {'PyT(ms)':>10} | {'Speedup':>7} | {'PyT Mem(MB)':>12}")
    print("-" * 75)

    for d in head_dims:
        for N in seq_lengths:
            # Inputs must be FP16
            q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            k = torch.randn(B, nh, N, d, device=device, dtype=dtype)
            v = torch.randn(B, nh, N, d, device=device, dtype=dtype)

            with torch.no_grad():
                # PyTorch SDPA supports FP16 automatically
                ref_out = scaled_dot_product_attention(q, k, v)
                try:
                    custom_out = flash_attn_lib.forward(q, k, v)
                except Exception as e:
                    print(f"{N:7d} | {d:3d} | Execution Error: {e}")
                    continue

            is_correct, max_err = validate(custom_out, ref_out)
            status = "PASS" if is_correct else f"FAIL (Max Diff: {max_err:.6f})"
            if not is_correct:
                print(f"Validation: {status}")

            try:
                t_custom, _ = benchmark_stats(
                    lambda: flash_attn_lib.forward(q, k, v))
                t_torch, mem_torch = benchmark_stats(
                    lambda: scaled_dot_product_attention(q, k, v))

                speedup = t_torch / (t_custom + 1e-8)
                print(
                    f"{N:7d} | {d:3d} | {t_custom:10.3f} | {t_torch:10.3f} | {speedup:7.2f}x | {mem_torch:12.2f}")

            except Exception as e:
                print(f"{N:7d} | {d:3d} | Bench Error: {str(e)[:25]}")


if __name__ == "__main__":
    run_sweep()
