#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// --- Tuning Parameters ---
#define WARP_SIZE 32
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

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
    // The number of 16-wide fragments needed to cover dimension d
    const int num_frags_d = HEAD_DIM / 16;
    const int d = HEAD_DIM;

    // --- Indexing ---
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x; // Block index for Q rows

    int tid = threadIdx.x;

    // Base Offsets
    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    int q_row_start = q_block_idx * 16;
    if (q_row_start >= N)
        return;

    // --- Shared Memory ---
    // s_smem: Stores scores (S) and later probabilities (P).
    // o_smem: Stores output accumulators (O).
    extern __shared__ float smem[];
    float *s_smem_f = smem;                          // Float view for Score Accumulation
    half *s_smem_h = reinterpret_cast<half *>(smem); // Half view for P matrix (aliased)
    float *o_smem = &smem[16 * 16];                  // Float view for O Accumulation

    // Initialize O accumulator in Shared Memory
    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        o_smem[i] = 0.0f;
    }

    // Row Statistics (Max m, Sum l)
    float *row_m = &o_smem[16 * d];
    float *row_l = &row_m[16];

    if (tid < 16)
    {
        row_m[tid] = -1e20f; // -inf
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // --- Register Fragments ---
    // Q is constant for the whole loop, load once.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[HEAD_DIM / 16];

#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        const half *ptr = q_base + q_row_start * d + i * 16;
        wmma::load_matrix_sync(q_frag[i], ptr, d);
    }

    // Double Buffering Registers for K and V
    // "curr" holds data for iteration j
    // "next" holds data for iteration j+16
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag_curr[HEAD_DIM / 16];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag_next[HEAD_DIM / 16];

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag_curr[HEAD_DIM / 16];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag_next[HEAD_DIM / 16];

// --- Prologue: Load first tile (j=0) ---
#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        // Load K0 (Transposed)
        const half *k_ptr = k_base + 0 * d + i * 16;
        wmma::load_matrix_sync(k_frag_curr[i], k_ptr, d);

        // Load V0
        const half *v_ptr = v_base + 0 * d + i * 16;
        wmma::load_matrix_sync(v_frag_curr[i], v_ptr, d);
    }

    // --- Main Loop ---
    for (int j = 0; j < N; j += 16)
    {

        // 1. Prefetch Next Tile (j + 16)
        // Issue loads *before* doing the heavy math for the current tile.
        // The latency of these loads will be hidden by the MMA instructions below.
        int next_j = j + 16;
        bool next_exists = (next_j < N);

        if (next_exists)
        {
#pragma unroll
            for (int i = 0; i < num_frags_d; i++)
            {
                const half *k_ptr = k_base + next_j * d + i * 16;
                wmma::load_matrix_sync(k_frag_next[i], k_ptr, d);

                const half *v_ptr = v_base + next_j * d + i * 16;
                wmma::load_matrix_sync(v_frag_next[i], v_ptr, d);
            }
        }

        // 2. Compute S = Q * K_curr^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            wmma::mma_sync(s_acc, q_frag[k_idx], k_frag_curr[k_idx], s_acc);
        }

        // Store Scores to Shared Memory
        wmma::store_matrix_sync(s_smem_f, s_acc, 16, wmma::mem_row_major);
        __syncthreads();

        // 3. Softmax & Online Output Update
        if (tid < 16)
        {
            int row = tid;

            // Find Max
            float local_max = -1e20f;
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c] * softmax_scale;
                if (val > local_max)
                    local_max = val;
                s_smem_f[row * 16 + c] = val; // Optimization: store scaled value
            }

            // Update Stats
            float m_prev = row_m[row];
            float l_prev = row_l[row];
            float m_curr = max(m_prev, local_max);

            // Compute Exp & Sum
            float local_sum = 0.0f;
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c];
                float p = expf(val - m_curr);
                s_smem_h[row * 16 + c] = __float2half(p); // Store P
                local_sum += p;
            }

            // Rescale O if max changed
            if (m_prev != m_curr)
            {
                float o_scale = expf(m_prev - m_curr);
                for (int c = 0; c < d; c++)
                {
                    o_smem[row * d + c] *= o_scale;
                }
            }

            row_m[row] = m_curr;
            row_l[row] = l_prev * expf(m_prev - m_curr) + local_sum;
        }
        __syncthreads();

        // 4. Compute O += P * V_curr
        // Load P (Matrix A)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_smem_h, 16);

// Accumulate
#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            // Load O accumulator from Shared Memory
            float *o_ptr = o_smem + k_idx * 16;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, o_ptr, d, wmma::mem_row_major);

            // O += P * V
            wmma::mma_sync(o_frag, p_frag, v_frag_curr[k_idx], o_frag);

            // Store back
            wmma::store_matrix_sync(o_ptr, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();

        // 5. Shift "Next" to "Current" for next iteration
        if (next_exists)
        {
#pragma unroll
            for (int i = 0; i < num_frags_d; i++)
            {
                k_frag_curr[i] = k_frag_next[i];
                v_frag_curr[i] = v_frag_next[i];
            }
        }
    }

    // --- Finalize & Write Output ---
    if (tid < 16)
    {
        int row = tid;
        float inv_l = 1.0f / (row_l[row] + 1e-6f);
        for (int c = 0; c < d; c++)
        {
            o_smem[row * d + c] *= inv_l;
        }
    }
    __syncthreads();

    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        out_base[q_row_start * d + i] = __float2half(o_smem[i]);
    }
}

// Host Launcher
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

    TORCH_CHECK(d % 16 == 0, "Head dim must be multiple of 16");
    TORCH_CHECK(N % 16 == 0, "Seq len must be multiple of 16");

    dim3 grid(N / 16, nh, B);
    dim3 block(32);

    size_t smem_size = 1024 + (16 * d * sizeof(float)) + 128;

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
        throw std::runtime_error("Unsupported head dimension");
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