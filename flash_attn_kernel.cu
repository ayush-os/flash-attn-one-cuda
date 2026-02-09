#include <torch/extension.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int head_dim>
__global__ void flash_attn_kernel(const half *__restrict__ q_ptr,
                                  const half *__restrict__ k_ptr,
                                  const half *__restrict__ v_ptr,
                                  half *out,
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
    extern __shared__ half s[];
    half *Qi = s;
    half *Kj = &Qi[Br * d];
    half *Vj = &Kj[Bc * d];

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int lm_offset = (blockIdx.x * nh * N) + (blockIdx.y * N);

    int i = blockIdx.z * Br;

    float Oi[head_dim];
    float li = 0.0f;
    float mi = -INFINITY;

    for (int idx = 0; idx < d; idx++)
        Oi[idx] = 0.0f;

    int row_idx = i + threadIdx.x;

    if (row_idx < N)
    {
        for (int j = 0; j < d; j++)
        {
            Qi[threadIdx.x * d + j] = q_ptr[qkv_offset + (row_idx * d) + j];
        }
    }
    else
    {
        for (int j = 0; j < d; j++)
        {
            Qi[threadIdx.x * d + j] = __float2half(0.0f);
        }
    }

    for (int j = 0; j < N; j += Bc)
    {
        for (int idx = threadIdx.x; idx < (Bc * d); idx += blockDim.x)
        {
            int col_row_idx = j + (idx / d);
            if (col_row_idx < N)
            {
                Kj[idx] = k_ptr[qkv_offset + (j * d) + idx];
                Vj[idx] = v_ptr[qkv_offset + (j * d) + idx];
            }
            else
            {
                Kj[idx] = __float2half(-65504.0f);
                ;
                Vj[idx] = __float2half(0.0f);
            }
        }
        __syncthreads();

        if (row_idx < N)
        {
            for (int ii = 0; ii < Bc; ii++)
            {
                float Sij = 0.f;
                for (int jj = 0; jj < d; jj++)
                {
                    Sij += __half2float(Qi[(threadIdx.x * d) + jj]) * __half2float(Kj[(ii * d) + jj]);
                }
                Sij *= softmax_scale;

                float old_mi = mi;
                mi = max(old_mi, Sij);

                float alpha = expf(old_mi - mi);
                float beta = expf(Sij - mi);

                li = li * alpha + beta;

                for (int k = 0; k < d; k++)
                {
                    Oi[k] = Oi[k] * alpha + beta * __half2float(Vj[ii * d + k]);
                }
            }
        }
        __syncthreads();
    }

    if (row_idx < N)
    {
        half *out_row_ptr = out + qkv_offset + (row_idx * d);
        for (int k = 0; k < d; k++)
        {
            out_row_ptr[k] = __float2half(Oi[k] / li);
        }
        l[lm_offset + row_idx] = li;
        m[lm_offset + row_idx] = mi;
    }
}

void launch_flash_attn_kernel(const half *q,
                              const half *k,
                              const half *v,
                              half *out,
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

    auto logsum_options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    // Step 2: Init O, l, m
    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, logsum_options);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, logsum_options);

    dim3 grid(B, nh, (N + Br - 1) / Br);
    dim3 block(Br);

    // Size in bytes for dynamic shared memory
    // Q_tile (Br * d) + K_tile (Bc * d) + V_tile (Bc * d)
    // O, l, m will be stored in regs
    size_t smem_bytes = (Br * d + 2 * Bc * d) * sizeof(half);

    launch_flash_attn_kernel(
        reinterpret_cast<const half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(v.data_ptr<at::Half>()),
        reinterpret_cast<half *>(out.data_ptr<at::Half>()),
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