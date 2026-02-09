#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32

// WMMA Shape Constants
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
    // --- Setup & Indexing ---
    const int d = HEAD_DIM;
    const int num_frags_d = HEAD_DIM / 16;

    // Grid: x=Q_blocks, y=Heads, z=Batch
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x; // Processes rows [q_block_idx*16 .. +16]

    int tid = threadIdx.x;

    // Global Memory Offsets
    // q, k, v shape: [B, nh, N, d]
    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    int q_row_start = q_block_idx * 16;
    if (q_row_start >= N)
        return;

    // --- Shared Memory Allocation ---
    extern __shared__ float smem[];

    // Layout:
    // 1. S_tile: 16x16 floats (used for Scores).
    //    Later reused as 16x16 halves (for Probabilities P).
    float *s_smem_f = smem;                          // Size: 256 floats (1024 bytes)
    half *s_smem_h = reinterpret_cast<half *>(smem); // Size: 256 halves (512 bytes) - Aliased!

    // 2. O_tile: 16 rows * d cols (floats).
    //    Placed after S_tile.
    float *o_smem = &smem[16 * 16];

    // Initialize O accumulator in Shared Memory to 0.0f
    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        o_smem[i] = 0.0f;
    }

    // 3. Row Statistics (Max m, Sum l)
    //    Placed after O_tile.
    float *row_m = &o_smem[16 * d];
    float *row_l = &row_m[16];

    if (tid < 16)
    {
        row_m[tid] = -1e20f; // -inf
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // --- Load Q (Reused for all K blocks) ---
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[HEAD_DIM / 16];

#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        const half *ptr = q_base + q_row_start * d + i * 16;
        wmma::load_matrix_sync(q_frag[i], ptr, d);
    }

    // --- Main Loop: Iterate over K and V (dim N) ---
    // We process K/V in blocks of 16 (WMMA_N)
    for (int j = 0; j < N; j += 16)
    {

        // --------------------------------------------------------
        // Step 1: Compute S = Q * K^T
        // --------------------------------------------------------
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            // Load K tile (Transposed via Col_Major)
            const half *ptr = k_base + j * d + k_idx * 16;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, ptr, d);

            wmma::mma_sync(s_acc, q_frag[k_idx], k_frag, s_acc);
        }

        // Store Accumulator (float) to Shared Memory (float)
        wmma::store_matrix_sync(s_smem_f, s_acc, 16, wmma::mem_row_major);
        __syncthreads();

        // --------------------------------------------------------
        // Step 2: Softmax & Rescaling
        // --------------------------------------------------------
        // Only first 16 threads (1 per row) perform this logic
        if (tid < 16)
        {
            int row = tid;

            // A. Find Max
            float local_max = -1e20f;
            for (int c = 0; c < 16; c++)
            {
                // Read float score
                float val = s_smem_f[row * 16 + c] * softmax_scale;
                if (val > local_max)
                    local_max = val;
                // Temporarily store scaled value back (optional optimization)
                s_smem_f[row * 16 + c] = val;
            }

            // B. Update Statistics
            float m_prev = row_m[row];
            float l_prev = row_l[row];
            float m_curr = max(m_prev, local_max);

            // C. Compute P (Exp) and Sum
            float local_sum = 0.0f;
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c]; // already scaled
                float p = expf(val - m_curr);

                // Write half directly to alias buffer
                // Safe: write index (half*) <= read index (float*)
                s_smem_h[row * 16 + c] = __float2half(p);

                local_sum += p;
            }

            // D. Rescale Previous Output (O) if Max changed
            // O_new = O_old * exp(m_prev - m_curr)
            if (m_prev != m_curr)
            { // optimization
                float o_scale = expf(m_prev - m_curr);
                for (int c = 0; c < d; c++)
                {
                    o_smem[row * d + c] *= o_scale;
                }
            }

            // E. Save new stats
            // L_new = L_prev * exp(m_prev - m_curr) + local_sum
            row_l[row] = l_prev * expf(m_prev - m_curr) + local_sum;
            row_m[row] = m_curr;
        }
        __syncthreads();

        // --------------------------------------------------------
        // Step 3: Compute O += P * V
        // --------------------------------------------------------

        // Load P (Matrix A) from Shared Memory (now half)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_smem_h, 16);

        // Iterate over column blocks of V (size d)
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            // Load V chunk
            const half *v_ptr = v_base + j * d + k_idx * 16;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, v_ptr, d);

            // Load O accumulator from Shared Memory
            float *o_ptr = o_smem + k_idx * 16;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, o_ptr, d, wmma::mem_row_major);

            // MMA: O = O + P * V
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

            // Store back to Shared Memory
            wmma::store_matrix_sync(o_ptr, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // --- Finalize: O = O / L ---
    if (tid < 16)
    {
        int row = tid;
        float inv_l = 1.0f / (row_l[row] + 1e-6f); // Add epsilon for stability
        for (int c = 0; c < d; c++)
        {
            o_smem[row * d + c] *= inv_l;
        }
    }
    __syncthreads();

    // --- Write Output to Global Memory ---
    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        // Coalesced write
        out_base[q_row_start * d + i] = __float2half(o_smem[i]);
    }
}

// --------------------------------------------------------
// Host Launcher
// --------------------------------------------------------

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
    dim3 block(32); // 1 Warp per block

    // Shared Memory Calculation
    // s_smem (float): 16*16*4 = 1024 bytes
    // o_smem (float): 16*d*4 bytes
    // row_stats (float): 16*2*4 = 128 bytes
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