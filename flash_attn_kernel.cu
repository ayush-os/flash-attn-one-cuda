#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Br=64 (4 warps × 16 rows each), Bc=16
// K/V loaded cooperatively into shared memory (padded to avoid bank conflicts)
// O accumulators kept in WMMA register fragments throughout KV loop
// Rescaling uses known sm_70+ WMMA 16×16×16 accumulator fragment layout

template <int HEAD_DIM, int Br = 64, int Bc = 16>
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
    const int d = HEAD_DIM;
    const int num_frags_d = HEAD_DIM / WMMA_K;
    // Pad shared memory stride to avoid bank conflicts on col-major K loads
    const int KV_PAD = 8;
    const int KV_STRIDE = d + KV_PAD;

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    int q_row_start = q_block_idx * Br;
    if (q_row_start >= N)
        return;

    // ============ Shared Memory Layout ============
    // kv_smem:   Bc × KV_STRIDE halfs  (K or V tile, padded)
    // s_smem:    Br × Bc floats         (QK^T scores)
    // p_smem:    Br × Bc halfs          (softmax output P)
    // row_m:     Br floats              (running row max)
    // row_l:     Br floats              (running row sum)
    // row_scale: Br floats              (rescale factors)
    extern __shared__ char smem_raw[];

    half *kv_smem = (half *)smem_raw;
    float *s_smem = (float *)(kv_smem + Bc * KV_STRIDE);
    half *p_smem = (half *)(s_smem + Br * Bc);
    float *row_m = (float *)(p_smem + Br * Bc);
    float *row_l = row_m + Br;
    float *row_scale = row_l + Br;

    // Initialize running statistics
    for (int i = tid; i < Br; i += blockDim.x)
    {
        row_m[i] = -1e20f;
        row_l[i] = 0.0f;
    }
    __syncthreads();

    // ============ Load Q into Register Fragments ============
    // Each warp loads its own 16 rows of Q (once, reused for all KV blocks)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[HEAD_DIM / WMMA_K];
#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        const half *ptr = q_base + (q_row_start + warp_id * 16) * d + i * 16;
        wmma::load_matrix_sync(q_frag[i], ptr, d);
    }

    // ============ O Accumulators in Registers ============
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[HEAD_DIM / WMMA_K];
#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        wmma::fill_fragment(o_frag[i], 0.0f);
    }

    // ============ Fragment Row Mapping ============
    // For sm_70+ WMMA 16×16×16 half→float accumulator:
    //   octet_id = lane / 4 (0-7)
    //   Elements 0,1,4,5 → row = octet_id
    //   Elements 2,3,6,7 → row = octet_id + 8
    int octet_id = lane_id / 4;
    int frag_row[8];
    frag_row[0] = frag_row[1] = frag_row[4] = frag_row[5] = warp_id * 16 + octet_id;
    frag_row[2] = frag_row[3] = frag_row[6] = frag_row[7] = warp_id * 16 + octet_id + 8;

    // ============ KV Loop ============
    for (int j = 0; j < N; j += Bc)
    {
        // ---- Step 1: Load K tile [Bc × d] into shared memory ----
        for (int idx = tid; idx < Bc * d; idx += blockDim.x)
        {
            int r = idx / d;
            int c = idx % d;
            kv_smem[r * KV_STRIDE + c] = k_base[(j + r) * d + c];
        }
        __syncthreads();

        // ---- Step 2: Compute S = Q @ K^T ----
        // Each warp computes its 16×Bc tile of scores
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

#pragma unroll
        for (int ki = 0; ki < num_frags_d; ki++)
        {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, kv_smem + ki * 16, KV_STRIDE);
            wmma::mma_sync(s_acc, q_frag[ki], k_frag, s_acc);
        }

        // ---- Step 3: Store S to shared memory ----
        wmma::store_matrix_sync(s_smem + warp_id * 16 * Bc, s_acc, Bc, wmma::mem_row_major);
        __syncthreads();

        // ---- Step 4: Online Softmax ----
        // 128 threads, 64 rows → each of first 64 threads handles one row
        for (int i = tid; i < Br; i += blockDim.x)
        {
            // Scale and find row max
            float local_max = -1e20f;
            for (int c = 0; c < Bc; c++)
            {
                float val = s_smem[i * Bc + c] * softmax_scale;
                s_smem[i * Bc + c] = val;
                if (val > local_max)
                    local_max = val;
            }

            float m_prev = row_m[i];
            float m_curr = max(m_prev, local_max);

            // Exponentiate and sum
            float local_sum = 0.0f;
            for (int c = 0; c < Bc; c++)
            {
                float p = expf(s_smem[i * Bc + c] - m_curr);
                p_smem[i * Bc + c] = __float2half(p);
                local_sum += p;
            }

            // Compute rescale factor and update running stats
            float scale = expf(m_prev - m_curr);
            row_scale[i] = scale;
            row_l[i] = row_l[i] * scale + local_sum;
            row_m[i] = m_curr;
        }
        __syncthreads();

        // ---- Step 5: Rescale O accumulators in registers ----
#pragma unroll
        for (int f = 0; f < num_frags_d; f++)
        {
#pragma unroll
            for (int e = 0; e < 8; e++)
            {
                o_frag[f].x[e] *= row_scale[frag_row[e]];
            }
        }

        // ---- Step 6: Load V tile [Bc × d] into shared memory (reuse kv_smem) ----
        for (int idx = tid; idx < Bc * d; idx += blockDim.x)
        {
            int r = idx / d;
            int c = idx % d;
            kv_smem[r * KV_STRIDE + c] = v_base[(j + r) * d + c];
        }
        __syncthreads();

        // ---- Step 7: Compute O += P @ V ----
        // Each warp loads its P tile (16×Bc) and multiplies by V fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, p_smem + warp_id * 16 * Bc, Bc);

#pragma unroll
        for (int ki = 0; ki < num_frags_d; ki++)
        {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, kv_smem + ki * 16, KV_STRIDE);
            wmma::mma_sync(o_frag[ki], p_frag, v_frag, o_frag[ki]);
        }
        __syncthreads();
    }

    // ============ Finalize: Divide O by row_l ============
    for (int i = tid; i < Br; i += blockDim.x)
    {
        row_scale[i] = 1.0f / (row_l[i] + 1e-6f);
    }
    __syncthreads();

#pragma unroll
    for (int f = 0; f < num_frags_d; f++)
    {
#pragma unroll
        for (int e = 0; e < 8; e++)
        {
            o_frag[f].x[e] *= row_scale[frag_row[e]];
        }
    }

    // ============ Store Output ============
    // Reuse smem as staging buffer: all 4 warps store their 16×16 fragments
    // simultaneously to non-overlapping regions, then cooperatively write to global
    float *out_smem = (float *)smem_raw; // needs Br * 16 = 4096 bytes, fits

    for (int f = 0; f < num_frags_d; f++)
    {
        // Each warp stores its 16×16 tile to its section of out_smem
        wmma::store_matrix_sync(out_smem + warp_id * 16 * 16, o_frag[f], 16, wmma::mem_row_major);
        __syncthreads();

        // All 128 threads cooperatively convert float→half and write to global
        for (int i = tid; i < Br * 16; i += blockDim.x)
        {
            int r = i / 16;
            int c = i % 16;
            out_base[(q_row_start + r) * d + f * 16 + c] =
                __float2half(out_smem[r * 16 + c]);
        }
        __syncthreads();
    }
}

// ============ Host Launcher ============

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

    const int Br = 64;
    const int Bc = 16;
    const int NUM_WARPS = Br / 16; // 4
    const int KV_PAD = 8;
    const int KV_STRIDE = d + KV_PAD;

    TORCH_CHECK(d % 16 == 0, "Head dim must be multiple of 16");
    TORCH_CHECK(N % Br == 0, "Seq len must be multiple of ", Br);

    dim3 grid(N / Br, nh, B);
    dim3 block(NUM_WARPS * 32); // 128 threads

    // Shared memory: kv_smem + s_smem + p_smem + 3 stat arrays
    size_t smem_size = Bc * KV_STRIDE * sizeof(half) // kv tile (padded)
                       + Br * Bc * sizeof(float)     // S scores
                       + Br * Bc * sizeof(half)      // P after softmax
                       + 3 * Br * sizeof(float);     // row_m, row_l, row_scale

    // Ensure smem is large enough for output staging (Br * 16 floats = 4096 bytes)
    size_t out_stage = Br * 16 * sizeof(float);
    if (out_stage > smem_size)
        smem_size = out_stage;

    if (d == 32)
    {
        flash_attn_tc_kernel<32><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(),
            (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else if (d == 64)
    {
        flash_attn_tc_kernel<64><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(),
            (half *)v.data_ptr(), (half *)out.data_ptr(),
            B, nh, N, softmax_scale);
    }
    else if (d == 128)
    {
        flash_attn_tc_kernel<128><<<grid, block, smem_size>>>(
            (half *)q.data_ptr(), (half *)k.data_ptr(),
            (half *)v.data_ptr(), (half *)out.data_ptr(),
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
