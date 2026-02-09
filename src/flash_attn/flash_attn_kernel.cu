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
                                  const int N,
                                  const int d,
                                  const int Bc,
                                  const int Br,
                                  const float softmax_scale)
{
    extern __shared__ float s[];
    float *Qi = s;
    float *Kj = &Qi[Br * d];
    float *Vj = &Kj[Bc * d];

    float Oi[head_dim];
    float li;
    float mi;

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int lm_offset = (blockIdx.x * nh * N) + (blockIdx.y * N);

    // inverting loops from paper's algorithm for less HBM writes
    for (int i = 0; i < N; i += Br) // step 7
    {
        // step 8: load Qi, Oi, li, mi into sram
        int row_idx = i + threadIdx.x;
        float *out_row_ptr;

        if (row_idx < N)
        {
            const float *q_row_ptr = q + qkv_offset + (row_idx * d);
            out_row_ptr = out + qkv_offset + (row_idx * d);
            for (int k = 0; k < d; k++)
            {
                Qi[threadIdx.x * d + k] = q_row_ptr[k];
                Oi[k] = out_row_ptr[k];
            }
            li = l[lm_offset + row_idx];
            mi = m[lm_offset + row_idx];
        }
        else
        {
            for (int k = 0; k < d; k++)
            {
                Qi[threadIdx.x * d + k] = 0.0f;
                Oi[k] = 0.0f;
            }
            li = 0.0f;
            mi = -INFINITY;
        }
        __syncthreads(); // technically removable, but better for clarity - at this point, Qi is fully loaded

        for (int j = 0; j < N; j += Bc) // step 5
        {
            // step 6: load Kj, Vj into sram
            for (int k = threadIdx.x; k < (Bc * d); k += blockDim.x)
            {
                int col_row_idx = j + (k / d);
                if (col_row_idx < N)
                {
                    Kj[k] = k[qkv_offset + (j * d) + k];
                    Vj[k] = v[qkv_offset + (j * d) + k];
                }
                else
                {
                    Kj[k] = 0.0f;
                    Vj[k] = 0.0f;
                }
            }
            __syncthreads(); // at this point Kj and Vj are fully loaded

            float mij = -INFINITY;
            float lij = 0.0f;
            // step 9: compute Sij = QiKj^T
            for (int ii = 0; ii < Bc; ii++) {
                float sum = 0.f;
                for (int jj = 0; jj < d; jj++) {
                    sum += Qi[(threadIdx.x * d) + jj] * Kj[(ii * d) + jj];
                }
                float Sij = sum * softmax_scale;

                // step 10: compute mij, lij, Pij
                float old_mij = mij;                
                mij = max(mij, Sij);

                float factor = expf(old_mij - mij);
                float Pij = expf(Sij - mij)
                lij = (lij * factor) + Pij;

                // step 12: Pij to Oi
                for (int k = 0; k < d; k++) {
                    Oi[k] = (Oi[k] * factor) + (Pij * Vj[ii * d + k]);
                }
            }
            __syncthreads(); // make sure thread does not start overwriting Kj and Vj currently in use
        }

        if (row_idx < N) {
            // step 11
            float mi_new = max(mi, mij);
            float factor_mi = expf(mi - mi_new);
            float factor_mij = expf(mij - mi_new);
            float li_new = factor_mi * li + factor_mij * lij;

            l[lm_offset + row_idx] = li_new;
            m[lm_offset + row_idx] = mi_new;

            for (int k = 0; k < d; k++) {
                out_row_ptr[k] = Oi[k] / li_new;
            }
        }

        __syncthreads();
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
        softmax_scale,
        grid,
        block,
        smem_bytes);

    return out;
}