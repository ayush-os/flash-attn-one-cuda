import torch
import json
import itertools
from torch.utils.cpp_extension import load


def autotune_flash_attn():
    device = torch.device("cuda")
    # Search Space
    configs = [
        {"warps": 1, "block_size": 32},
        {"warps": 2, "block_size": 64},
        {"warps": 4, "block_size": 128},
        {"warps": 8, "block_size": 256},
    ]

    # Problem size to tune for
    B, nh, N, d = 4, 8, 2048, 64
    q = torch.randn(B, nh, N, d, device=device, dtype=torch.float16)
    k = torch.randn(B, nh, N, d, device=device, dtype=torch.float16)
    v = torch.randn(B, nh, N, d, device=device, dtype=torch.float16)

    best_time = float('inf')
    best_config = None

    print(f"Starting Autotune for N={N}, d={d}...")

    for config in configs:
        try:
            # Recompile with new flags if necessary, or pass via template
            # For simplicity, we assume the kernel uses these as template args
            ext = load(
                name=f"flash_attn_tuned_{config['warps']}",
                sources=["flash_attn_kernel.cu", "wrapper.cpp"],
                extra_cuda_cflags=[
                    f"-DWARPS_PER_BLOCK={config['warps']}", "-O3"]
            )

            # Warmup
            for _ in range(10):
                ext.forward(q, k, v)

            # Timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(100):
                ext.forward(q, k, v)
            end.record()
            torch.cuda.synchronize()

            avg_ms = start.elapsed_time(end) / 100
            print(f"Config {config}: {avg_ms:.4f} ms")

            if avg_ms < best_time:
                best_time = avg_ms
                best_config = config

        except Exception as e:
            print(f"Config {config} failed: {e}")

    print(f"\n🚀 Best Config: {best_config} at {best_time:.4f} ms")
    return best_config


if __name__ == "__main__":
    best = autotune_flash_attn()
    # Save to a config file for the production build
    with open("best_config.json", "w") as f:
        json.dump(best, f)
