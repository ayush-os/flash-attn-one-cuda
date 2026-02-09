import torch
from torch.utils.cpp_extension import load
import math

flash_attn_lib = load(
    name="flash_attn_lib",
    sources=["flash_attn_kernel.cu", "wrapper.cpp"],
    extra_cuda_cflags=["-O3"]
)

def test_accuracy_and_perf():
    # Setup dimensions
    B, nh, N, d = 4, 8, 1024, 64
    device = torch.device("cuda")
    
    q = torch.randn(B, nh, N, d, device=device, dtype=torch.float32)
    k = torch.randn(B, nh, N, d, device=device, dtype=torch.float32)
    v = torch.randn(B, nh, N, d, device=device, dtype=torch.float32)

    # --- 1. Ground Truth (PyTorch) ---
    def manual_attn(q, k, v):
        scale = 1.0 / math.sqrt(d)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        return torch.matmul(attn, v)

    expected = manual_attn(q, k, v)

    # --- 2. My Kernel ---
    actual = flash_attn_lib.forward(q, k, v)

    # --- Accuracy Check ---
    diff = (expected - actual).abs().max()
    print(f"Max absolute difference: {diff.item():.6f}")
    if diff < 1e-3:
        print("✅ Accuracy Check Passed!")
    else:
        print("❌ Accuracy Check Failed!")

    # --- 3. Benchmarking ---
    # Warmup
    for _ in range(10):
        flash_attn_lib.forward(q, k, v)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        flash_attn_lib.forward(q, k, v)
    end_event.record()

    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / 100
    print(f"Average execution time: {avg_ms:.3f} ms")

if __name__ == "__main__":
    test_accuracy_and_perf()