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
    extern __shared__ float s[];
    float *Qi = s;
    float *Kj = &Qi[Br * d];
    float *Vj = &Kj[Bc * d];

    // Parallelize across Batch, Head, and Sequence Chunk
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z * Br; // Current row offset
    int tid = threadIdx.x;
    int row_idx = i + tid;

    int qkv_offset = (b * nh * N * d) + (h * N * d);
    int lm_offset = (b * nh * N) + (h * N);

    // Local registers for this thread's row
    float Oi[head_dim] = {0.0f};
    float li = 0.0f;
    float mi = -INFINITY;

    // Load Q row into shared memory once
    if (row_idx < N) {
        for (int j = 0; j < d; j++) {
            Qi[tid * d + j] = q_ptr[qkv_offset + (row_idx * d) + j];
        }
    }

    for (int j = 0; j < N; j += Bc) {
        // Load K, V tiles into shared memory (collaborative load)
        for (int idx = tid; idx < (Bc * d); idx += blockDim.x) {
            int col_row_idx = j + (idx / d);
            if (col_row_idx < N) {
                Kj[idx] = k_ptr[qkv_offset + (j * d) + idx];
                Vj[idx] = v_ptr[qkv_offset + (j * d) + idx];
            } else {
                Kj[idx] = 0.0f;
                Vj[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention for this tile
        if (row_idx < N) {
            for (int col = 0; col < Bc && (j + col) < N; col++) {
                float Sij = 0.0f;
                for (int jj = 0; jj < d; jj++) {
                    Sij += Qi[tid * d + jj] * Kj[col * d + jj];
                }
                Sij *= softmax_scale;

                float mi_old = mi;
                mi = max(mi_old, Sij);
                float alpha = expf(mi_old - mi);
                float beta = expf(Sij - mi);

                li = li * alpha + beta;
                for (int k = 0; k < d; k++) {
                    Oi[k] = Oi[k] * alpha + beta * Vj[col * d + k];
                }
            }
        }
        __syncthreads();
    }

    // Final write-back
    if (row_idx < N) {
        for (int k = 0; k < d; k++) {
            out[qkv_offset + (row_idx * d) + k] = Oi[k] / li;
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
    int Bc = std::max(1, SRAM_SIZE / (4 * d * (int)sizeof(float)));
    int Br = 32;

    // Step 2: Init O, l, m
    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, options);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, options);

    dim3 grid(B, nh, (N + Br - 1) / Br);
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