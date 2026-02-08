#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int head_dim>
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
                                  const int Tc,
                                  const int Tr,
                                  const float softmax_scale)
{
    extern __shared__ float s[];
    float *Qi = s;
    float *Kj = &Qi[Br * d];
    char *Vj = &Kj[Bc * d];

    float Oi[head_dim] = {0.0f};
    float li = 0.0f;
    float mi = __int_as_float(0xff800000);

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int lm_offset = (blockIdx.x * nh * N) + (blockIdx.y * N);

    // inverting loops from paper's algorithm for less HBM writes
    for (int i = 0; i < N; i += Br) // step 7
    {
        // step 8: load Qi, Oi, li, mi into sram
        int row_idx = i + threadIdx.x;

        const float *q_row_ptr = q + qkv_offset + (row_idx * d);
        float *o_row_ptr = out + qkv_offset + (row_idx * d);
        for (int k = 0; k < d; k++)
        {
            Qi[threadIdx.x * d + k] = q_row_ptr[k];
            Oi[k] = out_row_ptr[k]
        }
        li = l[lm_offset + row_idx];
        mi = m[lm_offset + row_idx];
        // end step 8

        for (int j = 0; j < N; j += Bc) // step 5
        {
            // step 6: load Kj, Vj into sram
            int offset = j * d;
            const float *k_ptr = k + qkv_offset + offset;
            const float *v_ptr = v + qkv_offset + offset;
            for (int k = threadIdx.x; k < (Bc * d); k += blockDim.x)
            {
                Kj[k] = k_ptr[k];
                Vj[k] = v_ptr[k];
            }

            __syncthreads();
        }
    }
}

void launch_flash_attn_kernel(const float *q,
                              const float *k,
                              const float *v,
                              float *out,
                              float *l,
                              float *m,
                              int B,
                              int nh,
                              int N,
                              int d,
                              int Bc,
                              int Br,
                              int Tc,
                              int Tr,
                              float scale,
                              dim3 grid,
                              dim3 block,
                              size_t smem)
{
    if (d <= 32)
    {
        flash_attn_kernel<32><<<grid, block, smem>>>(q, k, v, out, l, m, B, nh, N, d, Bc, Br, Tc, Tr, scale);
    }
    else if (d <= 64)
    {
        flash_attn_kernel<64><<<grid, block, smem>>>(q, k, v, out, l, m, B, nh, N, d, Bc, Br, Tc, Tr, scale);
    }
    else if (d <= 128)
    {
        flash_attn_kernel<128><<<grid, block, smem>>>(q, k, v, out, l, m, B, nh, N, d, Bc, Br, Tc, Tr, scale);
    }
    else
    {
        throw std::runtime_error(
            "Unsupported head dimension: " + std::to_string(d) +
            ". Supported sizes are <= 128.");
    }
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

    // Steps 3 and 4: calculate Tr and Tc
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    dim3 grid(B, nh);
    dim3 block(Br);

    // Size in bytes for dynamic shared memory
    // Q_tile (Br * d) + K_tile (Bc * d) + V_tile (Bc * d)
    // O, l, m will be stored in regs
    size_t smem_bytes = (Br * d + 2 * Bc * d) * sizeof(float);

    launch_flash_attn_kernel(
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
        Tc,
        Tr,
        softmax_scale,
        grid,
        block,
        smem_bytes);

    return out;
}