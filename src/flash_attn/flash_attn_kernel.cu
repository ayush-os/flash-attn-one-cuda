#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attn_kernel(const float *__restrict__ q,
                                  const float *__restrict__ k,
                                  const float *__restrict__ v,
                                  float *out,
                                  float *l,
                                  float *m,
                                  const int B,
                                  const int nh,
                                  const in N,
                                  const int d,
                                  const int Bc,
                                  const int Br,
                                  const float softmax_scale)
{
}

torch::Tensor flash_attn_cuda_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    const int B = q.size(0);
    const int nh = q.size(1);
    const int N = q.size(2);
    const int d = q.size(3);

    float softmax_scale = 1.0 / sqrt(d);

    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());

    // Step 1: Set block sizes
    const int SRAM_SIZE = 48000;
    int Bc = std::max(1, SRAM_SIZE / (4 * d * (int)sizeof(float)));
    int Br = 32;

    // Step 2: Init O, l, m
    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, options);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, options);

    dim3 grid(B, nh);
    dim3 block(Br);

    // Size in bytes for dynamic shared memory
    // Q_tile (Br * d) + K_tile (Bc * d) + V_tile (Bc * d)
    // O, l, m will be stored in regs
    size_t shared_mem_bytes = (Br * d + 2 * Bc * d) * sizeof(float);

    flash_attn_kernel<<<grid, block, shared_mem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        B,
        nh,
        N,
        d,
        Bc,
        Br,
        softmax_scale);

    return out;
}