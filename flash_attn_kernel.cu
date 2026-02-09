#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Helper to interpret 4 halves as a single 64-bit load (int2)
#define FETCH_HALF4(pointer) (reinterpret_cast<const int2 *>(&(pointer))[0])

template <int head_dim>
__global__ void flash_attn_kernel(const __half *__restrict__ q_ptr,
                                  const __half *__restrict__ k_ptr,
                                  const __half *__restrict__ v_ptr,
                                  __half *out,
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
    // d_padded usually ensures alignment, typically d is multiple of 8 for Tensor Cores, 
    // but here we just need alignment for int2 (4 halves = 8 bytes).
    const int d_padded = d + 8; 

    // Shared memory is now __half
    extern __shared__ __half s[];
    __half *Qi = s;
    __half *Kj = &Qi[Br * d_padded];
    __half *Vj = &Kj[Bc * d_padded];

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int lm_offset = (blockIdx.x * nh * N) + (blockIdx.y * N);

    int i = blockIdx.z * Br;

    // Accumulators in Float for precision
    float Oi[head_dim];
    float li = 0.0f;
    float mi = -INFINITY;

    for (int idx = 0; idx < d; idx++)
        Oi[idx] = 0.0f;

    int row_idx = i + threadIdx.x;

    // Using int2 (8 bytes) to load 4 halves at once
    const int num_vecs = (Br * d) / 4;
    const int2 *q_ptr_vec = reinterpret_cast<const int2 *>(q_ptr + qkv_offset);

    // --- Load Q into SRAM ---
    for (int idx = threadIdx.x; idx < num_vecs; idx += blockDim.x)
    {
        int row = idx / (d / 4);
        int col_vec = idx % (d / 4);
        int global_row = i + row;
        
        if (global_row < N)
        {
            int2 val = q_ptr_vec[global_row * (d / 4) + col_vec];
            reinterpret_cast<int2 *>(&Qi[row * d_padded + col_vec * 4])[0] = val;
        }
        else
        {
            // Zero padding
            int2 zero_val;
            // 0 in int representation of half is 0x0000
            zero_val.x = 0; 
            zero_val.y = 0;
            reinterpret_cast<int2 *>(&Qi[row * d_padded + col_vec * 4])[0] = zero_val;
        }
    }

    // --- Loop over KV blocks ---
    for (int j = 0; j < N; j += Bc)
    {
        const int num_vecs_kv = (Bc * d) / 4;
        const int2 *k_ptr_vec = reinterpret_cast<const int2 *>(k_ptr + qkv_offset);
        const int2 *v_ptr_vec = reinterpret_cast<const int2 *>(v_ptr + qkv_offset);

        // Load K and V into SRAM
        for (int idx = threadIdx.x; idx < num_vecs_kv; idx += blockDim.x)
        {
            int row = idx / (d / 4);
            int col_vec = idx % (d / 4);
            int global_row = j + row;
            
            if (global_row < N)
            {
                int2 k_val = k_ptr_vec[(global_row * (d / 4)) + col_vec];
                int2 v_val = v_ptr_vec[(global_row * (d / 4)) + col_vec];
                reinterpret_cast<int2 *>(&Kj[row * d_padded + col_vec * 4])[0] = k_val;
                reinterpret_cast<int2 *>(&Vj[row * d_padded + col_vec * 4])[0] = v_val;
            }
            else
            {
                int2 zero = {0, 0};
                reinterpret_cast<int2 *>(&Kj[row * d_padded + col_vec * 4])[0] = zero;
                reinterpret_cast<int2 *>(&Vj[row * d_padded + col_vec * 4])[0] = zero;
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
                
                // Vectorized Dot Product (Half input -> Float Accum)
                for (int jj = 0; jj < d; jj += 4)
                {
                    // Load 4 halves (64 bits)
                    int2 q_int2 = FETCH_HALF4(Qi[(threadIdx.x * d_padded) + jj]);
                    int2 k_int2 = FETCH_HALF4(Kj[(ii * d_padded) + jj]);

                    // Reinterpret raw bits as __half
                    const __half* q_h = reinterpret_cast<const __half*>(&q_int2);
                    const __half* k_h = reinterpret_cast<const __half*>(&k_int2);

                    // Convert to float for math
                    float4 q_f4, k_f4;
                    q_f4.x = __half2float(q_h[0]); q_f4.y = __half2float(q_h[1]);
                    q_f4.z = __half2float(q_h[2]); q_f4.w = __half2float(q_h[3]);
                    
                    k_f4.x = __half2float(k_h[0]); k_f4.y = __half2float(k_h[1]);
                    k_f4.z = __half2float(k_h[2]); k_f4.w = __half2float(k_h[3]);

                    Sij += q_f4.x * k_f4.x;
                    Sij += q_f4.y * k_f4.y;
                    Sij += q_f4.z * k_f4.z;
                    Sij += q_f4.w * k_f4.w;
                }
                Sij *= softmax_scale;

                // Online Softmax
                float old_mi = mi;
                mi = max(old_mi, Sij);

                float alpha = expf(old_mi - mi);
                float beta = expf(Sij - mi);

                li = li * alpha + beta;

                // Update Output (O = O * alpha + Attention * V)
                for (int k = 0; k < d; k += 4)
                {
                    int2 v_int2 = FETCH_HALF4(Vj[ii * d_padded + k]);
                    const __half* v_h = reinterpret_cast<const __half*>(&v_int2);

                    float4 v_f4;
                    v_f4.x = __half2float(v_h[0]);
                    v_f4.y = __half2float(v_h[1]);
                    v_f4.z = __half2float(v_h[2]);
                    v_f4.w = __half2float(v_h[3]);

                    Oi[k]     = Oi[k]     * alpha + beta * v_f4.x;
                    Oi[k + 1] = Oi[k + 1] * alpha + beta * v_f4.y;
                    Oi[k + 2] = Oi[k + 2] * alpha + beta * v_f4.z;
                    Oi[k + 3] = Oi[k + 3] * alpha + beta * v_f4.w;
                }
            }
        }
        __syncthreads();
    }

    // --- Final Output Scaling and Write ---
    
    // Write statistics
    if (row_idx < N)
    {
        l[lm_offset + row_idx] = li;
        m[lm_offset + row_idx] = mi;
    }

    // Scale O and write to global memory
    int2 *out_ptr_vec = reinterpret_cast<int2 *>(out + qkv_offset);
    
    // To minimize register pressure/divergence, we write directly from Oi
    // But since the write loop matches the load loop (tiled), we temporarily store back to Qi (SRAM)
    // or we can write directly if we iterate correctly. 
    // The original code re-used Qi to stage output. We will do the same.

    if (row_idx < N)
    {
        for (int k = 0; k < d; k += 4)
        {
            // Convert float accumulators back to half
            __half h0 = __float2half(Oi[k] / li);
            __half h1 = __float2half(Oi[k + 1] / li);
            __half h2 = __float2half(Oi[k + 2] / li);
            __half h3 = __float2half(Oi[k + 3] / li);

            // Pack 4 halves into int2
            __half h_arr[4] = {h0, h1, h2, h3};
            int2 packed_val = *reinterpret_cast<int2*>(h_arr);
            
            // Store to Shared Memory (as a buffer)
            reinterpret_cast<int2 *>(&Qi[threadIdx.x * d_padded + k])[0] = packed_val;
        }
    }

    __syncthreads();

    // Write from Shared Memory to Global Memory
    for (int idx = threadIdx.x; idx < num_vecs; idx += blockDim.x)
    {
        int row = idx / (d / 4);
        int col_vec = idx % (d / 4);
        int global_row = i + row;

        if (global_row < N)
        {
            int2 val = reinterpret_cast<int2 *>(&Qi[row * d_padded + col_vec * 4])[0];
            out_ptr_vec[global_row * (d / 4) + col_vec] = val;
        }
    }
}

void launch_flash_attn_kernel(const torch::Tensor &q,
                              const torch::Tensor &k,
                              const torch::Tensor &v,
                              torch::Tensor &out,
                              torch::Tensor &l,
                              torch::Tensor &m,
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
    // Extract data pointers
    // NOTE: q, k, v, out are FP16 (__half). l, m are FP32 (float).
    const __half *q_ptr = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __half *k_ptr = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const __half *v_ptr = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    __half *out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float *l_ptr = l.data_ptr<float>();
    float *m_ptr = m.data_ptr<float>();

    if (d <= 32)
    {
        flash_attn_kernel<32><<<grid, block, smem>>>(q_ptr, k_ptr, v_ptr, out_ptr, l_ptr, m_ptr, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 64)
    {
        flash_attn_kernel<64><<<grid, block, smem>>>(q_ptr, k_ptr, v_ptr, out_ptr, l_ptr, m_ptr, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 128)
    {
        flash_attn_kernel<128><<<grid, block, smem>>>(q_ptr, k_ptr, v_ptr, out_ptr, l_ptr, m_ptr, B, nh, N, d, Bc, Br, scale);
    }
    else
    {
        throw std::runtime_error("Unsupported head dimension");
    }
}

torch::Tensor flash_attn_cuda_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    // Check inputs
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be Half (FP16)");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be Half (FP16)");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be Half (FP16)");
    TORCH_CHECK(q.is_cuda(), "Input tensors must be on CUDA");

    const int B = q.size(0);
    const int nh = q.size(1);
    const int N = q.size(2);
    const int d = q.size(3);

    float softmax_scale = 1.0 / sqrt(d);

    // l and m must be float32 for stability
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    size_t max_sram = props.sharedMemPerBlock;

    // Tuning params
    int Br = 64; 
    int d_padded = d + 8; // align padding

    // Calculate Bc based on FP16 size (2 bytes)
    // Shared mem needed: Q block + K block + V block
    // Q: Br * d_padded * 2 bytes
    // K: Bc * d_padded * 2 bytes
    // V: Bc * d_padded * 2 bytes
    int q_size = Br * d_padded * sizeof(__half);
    int remaining_sram = max_sram - q_size;
    int kv_block_size = 2 * d_padded * sizeof(__half);
    int Bc = remaining_sram / kv_block_size;

    Bc = std::min(Bc, N);
    // Ensure Bc is reasonable (e.g. divisible by 4 or 8 if possible, though not strictly required by this logic)
    if (Bc < 1) Bc = 1;

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({B, nh, N}, opts_float);
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, opts_float);

    dim3 grid(B, nh, (N + Br - 1) / Br);
    dim3 block(Br);

    size_t smem_bytes = (Br * d_padded + 2 * Bc * d_padded) * sizeof(__half);

    launch_flash_attn_kernel(
        q, k, v, out, l, m,
        B, nh, N, d, Bc, Br,
        softmax_scale,
        grid, block, smem_bytes);

    return out;
}