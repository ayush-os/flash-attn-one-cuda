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
    // init constants
    const int d = HEAD_DIM;
    const int num_frags_d = HEAD_DIM / 16;

    // get the grid
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;

    int tid = threadIdx.x;

    // global memory offsets
    long long qkv_offset = (long long)batch_idx * nh * N * d + (long long)head_idx * N * d;
    const half *q_base = q + qkv_offset;
    const half *k_base = k + qkv_offset;
    const half *v_base = v + qkv_offset;
    half *out_base = out + qkv_offset;

    int q_row_start = q_block_idx * 16;
    if (q_row_start >= N)
        return;

    // shared memory initialization
    extern __shared__ float s[];
    float *s_smem_f = s;
    half *s_smem_h = reinterpret_cast<half *>(s);
    float *o_smem = &s[16 * 16];

    for (int i = tid; i < 16 * d; i += blockDim.x)
    {
        o_smem[i] = 0.0f;
    }

    float *row_m = &o_smem[16 * d];
    float *row_l = &row_m[16];

    if (tid < 16)
    {
        row_m[tid] = -1e20f;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // Load in Q
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag[HEAD_DIM / 16];

#pragma unroll
    for (int i = 0; i < num_frags_d; i++)
    {
        const half *ptr = q_base + q_row_start * d + i * 16;
        wmma::load_matrix_sync(q_frag[i], ptr, d);
    }

    // iterate over KV
    for (int j = 0; j < N; j += 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
        wmma::fill_fragment(s_acc, 0.0f);

#pragma unroll
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            const half *ptr = k_base + j * d + k_idx * 16;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, ptr, d);

            wmma::mma_sync(s_acc, q_frag[k_idx], k_frag, s_acc);
        }

        // store to S
        wmma::store_matrix_sync(s_smem_f, s_acc, 16, wmma::mem_row_major);
        __syncthreads();

        // softmax
        if (tid < 16)
        {
            int row = tid;

            // find max
            float local_max = -1e20f;

            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c] * softmax_scale;
                if (val > local_max)
                    local_max = val;
                s_smem_f[row * 16 + c] = val;
            }

            // update stats
            float m_prev = row_m[row];
            float l_prev = row_l[row];
            float m_curr = max(m_prev, local_max);

            // compute p (exp) and sum
            float local_sum = 0.0f;
            for (int c = 0; c < 16; c++)
            {
                float val = s_smem_f[row * 16 + c];
                float p = expf(val - m_curr);

                s_smem_h[row * 16 + c] = __float2half(p);

                local_sum += p;
            }

            // correct previous output o
            if (m_prev != m_curr)
            {
                float o_scale = expf(m_prev - m_curr);
                for (int c = 0; c < d; c++)
                {
                    o_smem[row * d + c] *= o_scale;
                }
            }

            // save new stats
            row_l[row] = l_prev * expf(m_prev - m_curr) + local_sum;
            row_m[row] = m_curr;
        }
        __syncthreads();

        // multiply P by V
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_smem_h, 16);
        for (int k_idx = 0; k_idx < num_frags_d; k_idx++)
        {
            const half *v_ptr = v_base + j * d + k_idx * 16;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, v_ptr, d);

            float *o_ptr = o_smem + k_idx * 16;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag, o_ptr, d, wmma::mem_row_major);

            // compute P * V
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

            wmma::store_matrix_sync(o_ptr, o_frag, d, wmma::mem_row_major);
        }
        __syncthreads();
    }

    if (tid < 16)
    {
        int row = tid;
        float inv_l = 1.0f / (row_l[row] + 1e-6f); // epsilon for stability
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