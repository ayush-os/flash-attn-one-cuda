import torch
import flash_attn

def verify():
    B, nh, N, d = 2, 8, 128, 32
    
    q = torch.randn(B, nh, N, d).cuda()
    k = torch.randn(B, nh, N, d).cuda()
    v = torch.randn(B, nh, N, d).cuda()

    # 1. Reference Implementation (Manual Math)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # 2. My Implementation
    actual = flash_attn.forward(q, k, v)

    # 3. Check Correctness
    diff = (expected - actual).abs().max().item()
    print(f"Max absolute difference: {diff:.6f}")
    
    if torch.allclose(expected, actual, atol=1e-3):
        print("✅ SUCCESS: Outputs match!")
    else:
        print("❌ FAILURE: Significant numerical drift detected.")

if __name__ == "__main__":
    verify()