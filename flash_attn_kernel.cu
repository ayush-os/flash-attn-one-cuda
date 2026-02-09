#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// helper for ceiling division
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <int HEAD_DIM>
__global__ void flash_attn_tc_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    half *__restrict__ out,
    const int B,
    const int nh,
    const int N,
    const float softmax_scale)
{
    // The number of fragments required to cover the Head Dimension (d)
    // HEAD_DIM must be a multiple of 16
    const int NUM_FRAGS_D = HEAD_DIM / 16;
    const int d = HEAD_DIM;

    // --- Indexing ---
    // Grid: (N / 16), nh, B
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x; // Each block processes 16 rows of Q

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;

    // Offsets
    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    // Base row for Q in this block
    int q_row_start = q_block_idx * 16;
    if (q_row_start >= N)
        return;

    // --- Shared Memory ---
    // We need shared memory to:
    // 1. Store S (Score) tile for Softmax [16x16]
    // 2. Store O (Output) tile for Rescaling [16 x d] (Float)
    extern __shared__ float smem[];
    half *s_smem = reinterpret_cast<half *>(smem);               // 16x16
    float *o_smem = reinterpret_cast<float *>(&s_smem[16 * 16]); // 16*d

    // Initialize Output in Shared Memory to 0
    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        o_smem[i] = 0.0f;
    }
    __syncthreads();

    // Row-wise statistics for Online Softmax (stored in registers/local array)
    // Each thread needs access to the stats for the rows it processes.
    // To simplify, we keep stats in shared memory so all threads can read.
    __shared__ float row_m[16];
    __shared__ float row_l[16];

    if (tid < 16)
    {
        row_m[tid] = -1e20f; // -inf
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // --- Load Q into Fragments ---
    // Q is constant for the inner loop.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[NUM_FRAGS_D];

#pragma unroll
    for (int i = 0; i < NUM_FRAGS_D; i++)
    {
        // Load 16x16 chunk of Q
        // Src: q_base + (row * d) + col
        const half *ptr = q_base + q_row_start * d + i * 16;
        wmma::load_matrix_sync(q_frag[i], ptr, d);
    }

    // --- Loop over K, V blocks (j) ---
    for (int j = 0; j < N; j += 16)
    {

        // 1. Compute S = Q * K^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

#pragma unroll
        for (int k_idx = 0; k_idx < NUM_FRAGS_D; k_idx++)
        {
            // Load K tile (16x16).
            // Q is 16x16 (Row Major). K tile needs to be Transposed (Col Major loading).
            // Src: k_base + (row * d) + col.
            // We want K_j (rows j..j+16).
            const half *ptr = k_base + j * d + k_idx * 16;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, ptr, d);

            wmma::mma_sync(s_acc, q_frag[k_idx], k_frag, s_acc);
        }

        // 2. Softmax Logic
        // Store S to Shared Mem
        wmma::store_matrix_sync(s_smem, s_acc, 16, wmma::mem_row_major);
        __syncthreads();

        // Compute Max and Sum for the current block
        // Only first 16 threads (1 per row) need to do this logic
        if (tid < 16)
        {
            int row = tid;
            float local_max = -1e20f;

            // Find max in row
            for (int c = 0; c < 16; c++)
            {
                float val = __half2float(s_smem[row * 16 + c]);
                val *= softmax_scale;
                if (val > local_max)
                    local_max = val;
                // Store scaled val back to simplify next step (optional, but clean)
                // s_smem[row*16+c] = __float2half(val);
            }

            // Update Global Max/Sum
            float m_prev = row_m[row];
            float l_prev = row_l[row];

            // Standard Online Softmax Update
            float m_curr = max(m_prev, local_max);

            // Calculate exponentials
            float local_sum = 0.0f;
            for (int c = 0; c < 16; c++)
            {
                float val = __half2float(s_smem[row * 16 + c]) * softmax_scale;
                float p = expf(val - m_curr);
                s_smem[row * 16 + c] = __float2half(p); // P matrix
                local_sum += p;
            }

            // Correction factor for previous output
            // O_new = O_old * exp(m_prev - m_curr) + P * V
            float o_scale = expf(m_prev - m_curr);

            // Update stats
            row_m[row] = m_curr;
            row_l[row] = l_prev * o_scale + local_sum;

            // Rescale O_old in Shared Memory immediately
            // O is stored in o_smem [16 rows, d cols]
            for (int c = 0; c < d; c++)
            {
                o_smem[row * d + c] *= o_scale;
            }
        }
        __syncthreads();

        // 3. Compute O_curr = P * V
        // Load P (16x16) from shared memory into Fragment A
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_smem, 16);

        // Load V and Accumulate into O in Shared Memory
        // Since we modified O in shared memory, we load it into fragments, accumulate, store back?
        // Or accumulating directly from fragments is cleaner.

        // Loop over column chunks of V (and O)
        for (int k_idx = 0; k_idx < NUM_FRAGS_D; k_idx++)
        {
            // Load V chunk (16x16)
            // V is [N, d]. Rows j..j+16. Cols k_idx*16.
            const half *v_ptr = v_base + j * d + k_idx * 16;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, v_ptr, d);

            // Load current O chunk from Shared Memory (accumulator)
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            float *o_ptr = o_smem + k_idx * 16;
            wmma::load_matrix_sync(o_frag, o_ptr, d, wmma::mem_row_major);

            // O = O + P * V
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

            // Store back to Shared Memory
            wmma::store_matrix_sync(o_ptr, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // --- Finalize and Write to Global Memory ---
    // O = O / L
    if (tid < 16)
    {
        int row = tid;
        float inv_l = 1.0f / row_l[row];

        for (int c = 0; c < d; c++)
        {
            o_smem[row * d + c] *= inv_l;
        }
    }
    __syncthreads();

    // Copy from Shared Memory to Global Output
    // 16 rows * d cols.
    // Coalesced write: warp writes rows.
    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        int r = i / d;
        int c = i % d;
        out_base[q_row_start * d + i] = __float2half(o_smem[i]);
    }
}

// Host launcher
void launch_flash_attn_tc(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v,
    torch::Tensor &out)
{
    const int B = q.size(0);
    const int nh = q.size(1);
    const int N = q.size(2);
    const int d = q.size(3);

    float softmax_scale = 1.0f / sqrtf(d);

    // Constraints for this specific TC implementation
    TORCH_CHECK(d % 16 == 0, "Head dimension must be multiple of 16 for WMMA");
    TORCH_CHECK(N % 16 == 0, "Sequence length must be multiple of 16 for this simple implementation");

    // Grid: (N/16 blocks for Q), nh, B
    dim3 grid(N / 16, nh, B);
    dim3 block(32); // 1 Warp per block

    // Shared Memory Calculation
    // s_smem: 16*16*sizeof(half) = 256 * 2 = 512 bytes
    // o_smem: 16*d*sizeof(float). If d=64 -> 1024 * 4 = 4096 bytes
    // row_stats: 16*4*2 = 128 bytes
    size_t smem_size = (16 * 16 * sizeof(half)) + (16 * d * sizeof(float)) + (32 * sizeof(float));

    if (d == 32)
    {
        flash_attn_tc_kernel<32><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else if (d == 64)
    {
        flash_attn_tc_kernel<64><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else if (d == 128)
    {
        flash_attn_tc_kernel<128><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else
    {
        throw std::runtime_error("Unsupported head dimension for this kernel");
    }
}

torch::Tensor flash_attn_cuda_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    TORCH_CHECK(q.scalar_type() == at::kHalf, "Q must be FP16");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "K must be FP16");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "V must be FP16");

    auto out = torch::empty_like(q);
    launch_flash_attn_tc(q, k, v, out);
    return out;
}