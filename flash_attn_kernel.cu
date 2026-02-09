#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int head_dim>
__global__ void flash_attn_kernel(const float *__restrict__ q_ptr,
                                  const float *__restrict__ k_ptr,
                                  const float *__restrict__ v_ptr,
                                  float *out,
                                  float *l,
                                  float *m,
                                  const int B,
                                  const int nh,
                                  const int N,
                                  const int d,
                                  const int Bc,
                                  const int Br,
                                  const float softmax_scale)
{
    const int d_padded = d + 1;
    extern __shared__ float s[];
    float *Qi = s;
    float *Kj = &Qi[Br * d_padded];
    float *Vj = &Kj[Bc * d_padded];

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int lm_offset = (blockIdx.x * nh * N) + (blockIdx.y * N);

    int i = blockIdx.z * Br;

    float Oi[head_dim];
    float li = 0.0f;
    float mi = -INFINITY;

    for (int idx = 0; idx < d; idx++)
        Oi[idx] = 0.0f;

    int row_idx = i + threadIdx.x;

    for (int idx = threadIdx.x; idx < (Br * d); idx += blockDim.x)
    {
        int row = idx / d;
        int col = idx % d;
        int q_row_idx = i + row;
        if (q_row_idx < N)
        {
            Qi[row * d_padded + col] = q_ptr[qkv_offset + (q_row_idx * d) + col];
        }
        else
        {
            Qi[row * d_padded + col] = 0.0f;
        }
    }

    for (int j = 0; j < N; j += Bc)
    {
        for (int idx = threadIdx.x; idx < (Bc * d); idx += blockDim.x)
        {
            int row = idx / d;
            int col = idx % d;
            int col_row_idx = j + row;
            if (col_row_idx < N)
            {
                Kj[row * d_padded + col] = k_ptr[qkv_offset + (j * d) + idx];
                Vj[row * d_padded + col] = v_ptr[qkv_offset + (j * d) + idx];
            }
            else
            {
                Kj[row * d_padded + col] = 0.0f;
                Vj[row * d_padded + col] = 0.0f;
            }
        }
        __syncthreads();

        if (row_idx < N)
        {
            for (int ii = 0; ii < Bc; ii++)
            {
                if ((j + ii) >= N)
                {
                    break;
                }
                float Sij = 0.f;
                for (int jj = 0; jj < d; jj++)
                {
                    Sij += Qi[(threadIdx.x * d_padded) + jj] * Kj[(ii * d_padded) + jj];
                }
                Sij *= softmax_scale;

                float old_mi = mi;
                mi = max(old_mi, Sij);

                float alpha = expf(old_mi - mi);
                float beta = expf(Sij - mi);

                li = li * alpha + beta;

                for (int k = 0; k < d; k++)
                {
                    Oi[k] = Oi[k] * alpha + beta * Vj[ii * d_padded + k];
                }
            }
        }
        __syncthreads();
    }

    if (row_idx < N)
    {
        float *out_row_ptr = out + qkv_offset + (row_idx * d);
        for (int k = 0; k < d; k++)
        {
            out_row_ptr[k] = Oi[k] / li;
        }
        l[lm_offset + row_idx] = li;
        m[lm_offset + row_idx] = mi;
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
                              float scale,
                              dim3 grid,
                              dim3 block,
                              size_t smem)
{
    if (d <= 32)
    {
        flash_attn_kernel<32><<<grid, block, smem>>>(q, k, v, out, l, m, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 64)
    {
        flash_attn_kernel<64><<<grid, block, smem>>>(q, k, v, out, l, m, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 128)
    {
        flash_attn_kernel<128><<<grid, block, smem>>>(q, k, v, out, l, m, B, nh, N, d, Bc, Br, scale);
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
    int Bc = std::max(1, SRAM_SIZE / (4 * (d + 1) * (int)sizeof(float)));
    int Br = 32;

    // Step 2: Init O, l, m
    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, options);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, options);

    dim3 grid(B, nh, (N + Br - 1) / Br);
    dim3 block(Br);

    size_t smem_bytes = (Br * (d + 1) + 2 * Bc * (d + 1)) * sizeof(float);

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
        softmax_scale,
        grid,
        block,
        smem_bytes);

    return out;
}