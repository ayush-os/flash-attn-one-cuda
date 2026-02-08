#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// TODO: Define your __global__ kernel here.
// Remember to use __shared__ memory for Q, K, V tiles.
// __global__ void flash_attn_kernel(...) { ... }

torch::Tensor flash_attn_cuda_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    const int B = q.size(0);
    const int nh = q.size(1);
    const int N = q.size(2);
    const int d = q.size(3);

    auto dev = q.device();
    auto options = torch::TensorOptions().dtype(q.dtype()).device(dev);
    
    // Allocate Output O and stats l, m
    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, options);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, options);

    // Calculate block sizes (Step 1)
    // Assuming 48KB SRAM for simplicity, adjust for your GPU
    const int SRAM_SIZE = 48000; 
    int Bc = std::max(1, SRAM_SIZE / (4 * d * (int)sizeof(float)));
    int Br = std::min(Bc, d);

    // TODO: Define your grid and block dimensions
    // Suggestion: One block per (Batch * Head)
    dim3 grid(B, nh);
    dim3 block(Br); 

    // TODO: Launch your kernel
    // flash_attn_kernel<<<grid, block, shared_mem_size>>>(
    //     q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
    //     out.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(),
    //     N, d, Bc, Br
    // );

    return out;
}