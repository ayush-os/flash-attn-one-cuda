#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA Shape Constants
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

template <int HEAD_DIM, int WARPS_PER_BLOCK>
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
    // Shared memory layout: S (256 floats), P (256 halves), O (16*d floats), Stats (32 floats)
    // We calculate the offset per warp to ensure no overlap.
    const int d = HEAD_DIM;
    const int num_frags_d = d / 16;

    // Calculate smem requirement per warp (in floats)
    // S: 16*16 = 256
    // P: 16*16 / 2 = 128 (since P is half, but we use float pointer for indexing smem)
    // O: 16*d
    // Stats: 16 + 16 = 32
    const int smem_per_warp = 256 + 128 + (16 * d) + 32;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Global memory offsets
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;

    // Each block processes (WARPS_PER_BLOCK * 16) rows of Q
    int q_row_start = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * 16;

    if (q_row_start >= N)
        return;

    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    // Partition shared memory for this specific warp
    extern __shared__ float smem_all[];
    float *s_smem = &smem_all[warp_id * smem_per_warp];
    half *p_smem = (half *)&s_smem[256];
    float *o_smem = (float *)&p_smem[256]; // p_smem is 256 halves
    float *row_m = &o_smem[16 * d];
    float *row_l = &row_m[16];

// Initialize output and stats
#pragma unroll
    for (int i = lane_id; i < 16 * d; i += 32)
    {
        o_smem[i] = 0.0f;
    }
    if (lane_id < 16)
    {
        row_m[lane_id] = -1e20f;
        row_l[lane_id] = 0.0f;
    }
    // No syncthreads needed here yet as only this warp touches its smem partition

    // Load Q fragment (per warp)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[HEAD_DIM / 16];
#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        wmma::load_matrix_sync(q_frag[i], q_base + q_row_start * d + i * 16, d);
    }

    // Outer loop over KV blocks
    for (int j = 0; j < N; j += 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

// Compute QK^T
#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, k_base + j * d + k_idx * 16, d);
            wmma::mma_sync(s_acc, q_frag[k_idx], k_frag, s_acc);
        }

        // Store to shared memory to perform softmax
        wmma::store_matrix_sync(s_smem, s_acc, 16, wmma::mem_row_major);

        // Softmax (Warp-local as each warp has its own row set)
        if (lane_id < 16)
        {
            int row = lane_id;
            float local_max = -1e20f;
#pragma unroll
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem[row * 16 + c] * softmax_scale;
                if (val > local_max)
                    local_max = val;
                s_smem[row * 16 + c] = val;
            }

            float m_prev = row_m[row];
            float l_prev = row_l[row];
            float m_curr = max(m_prev, local_max);

            float local_sum = 0.0f;
#pragma unroll
            for (int c = 0; c < 16; c++)
            {
                float p = expf(s_smem[row * 16 + c] - m_curr);
                p_smem[row * 16 + c] = __float2half(p);
                local_sum += p;
            }

            // Rescale previous accumulator
            float o_scale = expf(m_prev - m_curr);
#pragma unroll
            for (int c = 0; c < d; c++)
            {
                o_smem[row * d + c] *= o_scale;
            }

            row_l[row] = l_prev * o_scale + local_sum;
            row_m[row] = m_curr;
        }

        // Compute PV
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, p_smem, 16);

#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, v_base + j * d + k_idx * 16, d);

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, o_smem + k_idx * 16, d, wmma::mem_row_major);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            wmma::store_matrix_sync(o_smem + k_idx * 16, o_frag, d, wmma::mem_row_major);
        }
    }

    // Final normalization and write out
    if (lane_id < 16)
    {
        float inv_l = 1.0f / (row_l[lane_id] + 1e-6f);
#pragma unroll
        for (int c = 0; c < d; c++)
        {
            out_base[(q_row_start + lane_id) * d + c] = __float2half(o_smem[lane_id * d + c] * inv_l);
        }
    }
}

template <int HEAD_DIM>
void launch_helper(int B, int nh, int N, float scale, const half *q, const half *k, const half *v, half *out)
{
    // These values are tuned via the -DWARPS_PER_BLOCK flag in your compiler
    const int warps = WARPS_PER_BLOCK;
    const int threads = warps * 32;

    // Grid: Total rows / rows_per_block
    dim3 grid((N + (warps * 16) - 1) / (warps * 16), nh, B);
    dim3 block(threads);

    // Dynamic Smem calculation
    size_t smem_per_warp_bytes = (256 * 4) + (256 * 2) + (16 * HEAD_DIM * 4) + (32 * 4);
    size_t total_smem = smem_per_warp_bytes * warps;

    flash_attn_tc_kernel<HEAD_DIM, WARPS_PER_BLOCK><<<grid, block, total_smem>>>(
        q, k, v, out, B, nh, N, scale);
}

void launch_flash_attn_tc(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v, torch::Tensor &out)
{
    const int B = q.size(0);
    const int nh = q.size(1);
    const int N = q.size(2);
    const int d = q.size(3);
    float scale = 1.0f / sqrtf(d);

    if (d == 32)
        launch_helper<32>(B, nh, N, scale, (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr());
    else if (d == 64)
        launch_helper<64>(B, nh, N, scale, (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr());
    else if (d == 128)
        launch_helper<128>(B, nh, N, scale, (half *)q.data_ptr(), (half *)k.data_ptr(), (half *)v.data_ptr(), (half *)out.data_ptr());
}

torch::Tensor flash_attn_cuda_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    auto out = torch::empty_like(q);
    launch_flash_attn_tc(q, k, v, out);
    return out;
}