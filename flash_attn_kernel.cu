#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h> // Required for pipeline primitives

using namespace nvcuda;

// --- Helper for cp.async ---
// Copies 16 bytes (float4 * 2 or half * 8) from global to shared
__device__ __forceinline__ void cp_async4(void *smem_ptr, const void *glob_ptr)
{
    const int BYTES = 16;
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :
        : "r"(smem_addr), "l"(glob_ptr), "n"(BYTES)
        : "memory");
}

// --- Tuning Parameters ---
#define WARP_SIZE 32
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

template <int HEAD_DIM>
__global__ void flash_attn_cp_async_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    half *__restrict__ out,
    const int B,
    const int nh,
    const int N,
    const float softmax_scale)
{
    const int d = HEAD_DIM;
    const int num_frags_d = HEAD_DIM / 16;

    // Grid Indexing
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // --- Memory Offsets ---
    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    int q_row_start = q_block_idx * 16;
    if (q_row_start >= N)
        return;

    // --- Shared Memory Layout ---
    // 1. K_Smem: 2 buffers (Ping-Pong) x 16 rows x d cols
    // 2. V_Smem: 2 buffers (Ping-Pong) x 16 rows x d cols
    // 3. S_Smem: 1 buffer 16x16 (Float for scores, Half for P)
    // 4. O_Smem: 1 buffer 16xd (Float accumulator)
    // 5. Stats:  1 buffer 16x2 (Float)

    extern __shared__ char smem_raw[];

    // Pointers
    half *k_smem[2];
    half *v_smem[2];
    float *s_smem_f;
    half *s_smem_h;
    float *o_smem;
    float *row_m;
    float *row_l;

    // Layout Calculation
    // k_tile_sz = 16 * d * sizeof(half)
    size_t k_tile_sz = 16 * d * sizeof(half);

    k_smem[0] = (half *)(smem_raw);
    k_smem[1] = (half *)(smem_raw + k_tile_sz);
    v_smem[0] = (half *)(smem_raw + 2 * k_tile_sz);
    v_smem[1] = (half *)(smem_raw + 3 * k_tile_sz);

    size_t offset_so = 4 * k_tile_sz;

    s_smem_f = (float *)(smem_raw + offset_so);
    s_smem_h = (half *)(s_smem_f);          // Alias
    o_smem = (float *)(s_smem_f + 16 * 16); // 16*16 floats
    row_m = (float *)(o_smem + 16 * d);     // 16*d floats
    row_l = (float *)(row_m + 16);          // 16 floats

    // Initialize Output Accumulator
    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        o_smem[i] = 0.0f;
    }

    // Initialize Stats
    if (tid < 16)
    {
        row_m[tid] = -1e20f;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // --- Load Q (Once, held in registers) ---
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[HEAD_DIM / 16];
#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        const half *ptr = q_base + q_row_start * d + i * 16;
        wmma::load_matrix_sync(q_frag[i], ptr, d);
    }

    // --- Software Pipeline Prologue ---
    // Kick off load for Tile 0 into Buffer 0
    {
        const half *k_src = k_base; // j=0
        const half *v_src = v_base; // j=0
        half *k_dst = k_smem[0];
        half *v_dst = v_smem[0];

        // Each thread loads 8 halves (16 bytes) at a time
        // Total elements to load: 16 * d
        // Threads: 32. Total capacity per step: 32 * 8 = 256 elems.
        // For d=64, total = 1024 elems. Loops = 4.

        for (int i = tid * 8; i < 16 * d; i += blockDim.x * 8)
        {
            cp_async4(&k_dst[i], &k_src[i]);
            cp_async4(&v_dst[i], &v_src[i]);
        }
        // Commit group 0
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // --- Main Loop ---
    int cur_stage = 0;
    int next_stage = 1;

    for (int j = 0; j < N; j += 16)
    {

        // 1. Wait for data to arrive for `cur_stage`
        // We wait for all groups except the last N (which is 0 in prologue, but grows).
        // Actually, simpler logic: wait_group N-1? No, we commit every step.
        // We want 1 stage visible.
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads(); // Memory visible to all threads in block

        // 2. Kick off Load for `next_stage` (j + 16)
        int next_j = j + 16;
        if (next_j < N)
        {
            const half *k_src = k_base + next_j * d;
            const half *v_src = v_base + next_j * d;
            half *k_dst = k_smem[next_stage];
            half *v_dst = v_smem[next_stage];

            for (int i = tid * 8; i < 16 * d; i += blockDim.x * 8)
            {
                cp_async4(&k_dst[i], &k_src[i]);
                cp_async4(&v_dst[i], &v_src[i]);
            }
            // Commit this new fetch as a new group
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // 3. Compute using `cur_stage` (Data is in Shared Mem)

        // A. Compute S = Q * K_cur^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            // Load K fragment from Shared Mem (Buffer `cur_stage`)
            // K needs to be transposed. WMMA Col Major load does this.
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;

            // Ptr to the specific 16x16 tile in the 16xd K matrix
            half *ptr = k_smem[cur_stage] + k_idx * 16;

            // Note: Stride is 'd' because K_smem is [16, d]
            wmma::load_matrix_sync(k_frag, ptr, d);

            wmma::mma_sync(s_acc, q_frag[k_idx], k_frag, s_acc);
        }

        // Store S to Shared Mem (Float)
        wmma::store_matrix_sync(s_smem_f, s_acc, 16, wmma::mem_row_major);
        __syncthreads();

        // B. Softmax Update (Standard)
        if (tid < 16)
        {
            int row = tid;
            float local_max = -1e20f;
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c] * softmax_scale;
                if (val > local_max)
                    local_max = val;
                s_smem_f[row * 16 + c] = val;
            }

            float m_prev = row_m[row];
            float l_prev = row_l[row];
            float m_curr = max(m_prev, local_max);

            float local_sum = 0.0f;
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c];
                float p = expf(val - m_curr);
                s_smem_h[row * 16 + c] = __float2half(p);
                local_sum += p;
            }

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

        // C. Compute O += P * V_cur
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_smem_h, 16);

#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            // Load V fragment from Shared Mem (Buffer `cur_stage`)
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;

            half *v_ptr = v_smem[cur_stage] + k_idx * 16;
            wmma::load_matrix_sync(v_frag, v_ptr, d);

            float *o_ptr = o_smem + k_idx * 16;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, o_ptr, d, wmma::mem_row_major);

            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            wmma::store_matrix_sync(o_ptr, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();

        // 4. Swap Stages
        // If we are at the last step, we don't need to swap, but logic holds.
        cur_stage = 1 - cur_stage;
        next_stage = 1 - next_stage;
    }

    // --- Finalize ---
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

    // Write Output
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

    // Calculate Shared Mem Size
    // K: 2 * 16 * d * 2 bytes
    // V: 2 * 16 * d * 2 bytes
    // S: 16 * 16 * 4 bytes
    // O: 16 * d * 4 bytes
    // Stats: 16 * 2 * 4 bytes
    size_t k_sz = 2 * 16 * d * sizeof(half);
    size_t v_sz = 2 * 16 * d * sizeof(half);
    size_t s_sz = 16 * 16 * sizeof(float);
    size_t o_sz = 16 * d * sizeof(float);
    size_t stats_sz = 16 * 2 * sizeof(float);

    size_t smem_size = k_sz + v_sz + s_sz + o_sz + stats_sz;

    if (d == 32)
    {
        flash_attn_cp_async_kernel<32><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else if (d == 64)
    {
        flash_attn_cp_async_kernel<64><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else if (d == 128)
    {
        flash_attn_cp_async_kernel<128><<<grid, block, smem_size>>>(
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