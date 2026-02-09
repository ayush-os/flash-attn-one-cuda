#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper to easily access float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])

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
    // PADDING: Changed to +4 to ensure alignment for float4 while reducing bank conflicts
    const int d_padded = d + 4;

    extern __shared__ float s[];
    float *Qi = s;
    float *Kj = &Qi[Br * d_padded];
    float *Vj = &Kj[Bc * d_padded];

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int lm_offset = (blockIdx.x * nh * N) + (blockIdx.y * N);

    int i = blockIdx.z * Br;

    float Oi[head_dim]; // Registers for output
    float li = 0.0f;
    float mi = -INFINITY;

    // Initialize Output Registers
    for (int idx = 0; idx < d; idx++)
        Oi[idx] = 0.0f;

    int row_idx = i + threadIdx.x;

    // --- 1. Load Q (Vectorized) ---
    // We treat the Q block as a collection of float4s.
    // Each thread copies "some" float4s.
    // Note: We must handle the padding manually (Global is compact, Shared is strided).
    const int num_float4s = (Br * d) / 4;
    const float4 *q_ptr_4 = reinterpret_cast<const float4 *>(q_ptr + qkv_offset);

    for (int idx = threadIdx.x; idx < num_float4s; idx += blockDim.x)
    {
        int row = idx / (d / 4);
        int col_vec = idx % (d / 4);
        int global_row = i + row;

        if (global_row < N)
        {
            float4 val = q_ptr_4[global_row * (d / 4) + col_vec];
            // Write to shared memory with padding
            reinterpret_cast<float4 *>(&Qi[row * d_padded + col_vec * 4])[0] = val;
        }
        else
        {
            reinterpret_cast<float4 *>(&Qi[row * d_padded + col_vec * 4])[0] = {0.f, 0.f, 0.f, 0.f};
        }
    }

    // --- 2. Main Loop ---
    for (int j = 0; j < N; j += Bc)
    {
        // Load K and V (Vectorized)
        // Similar logic to Q loading
        const int num_float4s_kv = (Bc * d) / 4;
        const float4 *k_ptr_4 = reinterpret_cast<const float4 *>(k_ptr + qkv_offset);
        const float4 *v_ptr_4 = reinterpret_cast<const float4 *>(v_ptr + qkv_offset);

        for (int idx = threadIdx.x; idx < num_float4s_kv; idx += blockDim.x)
        {
            int row = idx / (d / 4); // This is actually 'col' in K^T context, but row in K matrix
            int col_vec = idx % (d / 4);
            int global_row = j + row; // The row index in K/V matrices

            if (global_row < N)
            {
                float4 k_val = k_ptr_4[global_row * (d / 4) + col_vec];
                float4 v_val = v_ptr_4[global_row * (d / 4) + col_vec];
                reinterpret_cast<float4 *>(&Kj[row * d_padded + col_vec * 4])[0] = k_val;
                reinterpret_cast<float4 *>(&Vj[row * d_padded + col_vec * 4])[0] = v_val;
            }
            else
            {
                reinterpret_cast<float4 *>(&Kj[row * d_padded + col_vec * 4])[0] = {0.f, 0.f, 0.f, 0.f};
                reinterpret_cast<float4 *>(&Vj[row * d_padded + col_vec * 4])[0] = {0.f, 0.f, 0.f, 0.f};
            }
        }
        __syncthreads();

        // Compute Attention
        if (row_idx < N)
        {
            for (int ii = 0; ii < Bc; ii++)
            {
                if ((j + ii) >= N)
                    break;

                float Sij = 0.f;
                // Vectorized Dot Product (Q * K^T)
                for (int jj = 0; jj < d; jj += 4)
                {
                    float4 q_vec = FETCH_FLOAT4(Qi[threadIdx.x * d_padded + jj]);
                    float4 k_vec = FETCH_FLOAT4(Kj[ii * d_padded + jj]);

                    Sij += q_vec.x * k_vec.x;
                    Sij += q_vec.y * k_vec.y;
                    Sij += q_vec.z * k_vec.z;
                    Sij += q_vec.w * k_vec.w;
                }
                Sij *= softmax_scale;

                // Online Softmax
                float old_mi = mi;
                mi = max(old_mi, Sij);
                float alpha = expf(old_mi - mi);
                float beta = expf(Sij - mi);
                li = li * alpha + beta;

                // Vectorized Value Accumulation (O * V)
                for (int k = 0; k < d; k += 4)
                {
                    float4 v_vec = FETCH_FLOAT4(Vj[ii * d_padded + k]);
                    // Oi = Oi * alpha + beta * V
                    // We can't cast Oi array to float4 easily if it's in registers,
                    // but we can unroll manually.
                    Oi[k] = Oi[k] * alpha + beta * v_vec.x;
                    Oi[k + 1] = Oi[k + 1] * alpha + beta * v_vec.y;
                    Oi[k + 2] = Oi[k + 2] * alpha + beta * v_vec.z;
                    Oi[k + 3] = Oi[k + 3] * alpha + beta * v_vec.w;
                }
            }
        }
        __syncthreads();
    }

    // --- 3. Write Output (Vectorized + Coalesced) ---

    // Write l and m
    if (row_idx < N)
    {
        l[lm_offset + row_idx] = li;
        m[lm_offset + row_idx] = mi;
    }

    // Stage Output to Shared Memory (to coalesce global write)
    if (row_idx < N)
    {
        for (int k = 0; k < d; k += 4)
        {
            float4 val;
            val.x = Oi[k] / li;
            val.y = Oi[k + 1] / li;
            val.z = Oi[k + 2] / li;
            val.w = Oi[k + 3] / li;
            reinterpret_cast<float4 *>(&Qi[threadIdx.x * d_padded + k])[0] = val;
        }
    }
    __syncthreads();

    // Coalesced Write to Global Memory using float4
    float4 *out_ptr_4 = reinterpret_cast<float4 *>(out + qkv_offset);
    for (int idx = threadIdx.x; idx < num_float4s; idx += blockDim.x)
    {
        int row = idx / (d / 4);
        int col_vec = idx % (d / 4);
        int global_row = i + row;

        if (global_row < N)
        {
            float4 val = reinterpret_cast<float4 *>(&Qi[row * d_padded + col_vec * 4])[0];
            out_ptr_4[global_row * (d / 4) + col_vec] = val;
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

    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    size_t max_sram = props.sharedMemPerBlock;

    int Br = 64;
    int d_padded = d + 4;

    int remaining_sram = max_sram - (Br * d_padded * sizeof(float));
    int Bc = remaining_sram / (2 * d_padded * sizeof(float));

    Bc = std::min(Bc, N);

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, options);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, options);

    dim3 grid(B, nh, (N + Br - 1) / Br);
    dim3 block(Br);

    size_t smem_bytes = (Br * d_padded + 2 * Bc * d_padded) * sizeof(float);

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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    return out;
}