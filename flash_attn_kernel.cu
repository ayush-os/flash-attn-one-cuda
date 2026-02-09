#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FETCH_HALF4(pointer) (reinterpret_cast<const float2 *>(&(pointer))[0])

template <int head_dim>
__global__ void flash_attn_kernel(const half *__restrict__ q_ptr,
                                  const half *__restrict__ k_ptr,
                                  const half *__restrict__ v_ptr,
                                  half *out,
                                  const int B,
                                  const int nh,
                                  const int N,
                                  const int d,
                                  const int Bc,
                                  const int Br,
                                  const float softmax_scale)
{
    const int d_padded = d + 4;

    extern __shared__ half s[];
    half *Qi = s;
    half *Kj = &Qi[Br * d_padded];
    half *Vj = &Kj[Bc * d_padded];

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int i = blockIdx.z * Br;
    int row_idx = i + threadIdx.x;

    float Oi[head_dim];
    float li = 0.0f;
    float mi = -1e20f;

    for (int idx = 0; idx < d; idx++)
        Oi[idx] = 0.0f;

    const int num_vectors = (Br * d) / 4;
    const half *q_ptr_base = q_ptr + qkv_offset;

    for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x)
    {
        int r = idx / (d / 4);
        int c_v = idx % (d / 4);
        int shared_idx = r * d_padded + c_v * 4;
        int global_row = i + r;

        if (global_row < N)
        {
            float2 vec = reinterpret_cast<const float2 *>(q_ptr_base + global_row * d)[c_v];
            reinterpret_cast<float2 *>(&Qi[shared_idx])[0] = vec;
        }
        else
        {
            reinterpret_cast<float2 *>(&Qi[shared_idx])[0] = {0.f, 0.f};
        }
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc)
    {
        const int num_vectors_kv = (Bc * d) / 4;
        const half *k_ptr_base = k_ptr + qkv_offset;
        const half *v_ptr_base = v_ptr + qkv_offset;

        for (int idx = threadIdx.x; idx < num_vectors_kv; idx += blockDim.x)
        {
            int r = idx / (d / 4);
            int c_v = idx % (d / 4);
            int global_row = j + r;
            int shared_idx = r * d_padded + c_v * 4;

            if (global_row < N)
            {
                float2 k_vec = reinterpret_cast<const float2 *>(k_ptr_base + global_row * d)[c_v];
                float2 v_vec = reinterpret_cast<const float2 *>(v_ptr_base + global_row * d)[c_v];

                reinterpret_cast<float2 *>(&Kj[shared_idx])[0] = k_vec;
                reinterpret_cast<float2 *>(&Vj[shared_idx])[0] = v_vec;
            }
            else
            {
                reinterpret_cast<float2 *>(&Kj[shared_idx])[0] = {0.f, 0.f};
                reinterpret_cast<float2 *>(&Vj[shared_idx])[0] = {0.f, 0.f};
            }
        }
        __syncthreads();

        if (row_idx < N)
        {
            for (int ii = 0; ii < Bc; ii++)
            {
                if ((j + ii) >= N)
                    break;

                float Sij = 0.f;
                for (int jj = 0; jj < d; jj += 4)
                {
                    float2 q_vec_f2 = FETCH_HALF4(Qi[(threadIdx.x * d_padded) + jj]);
                    float2 k_vec_f2 = FETCH_HALF4(Kj[(ii * d_padded) + jj]);

                    const half *q_h = reinterpret_cast<const half *>(&q_vec_f2);
                    const half *k_h = reinterpret_cast<const half *>(&k_vec_f2);

                    Sij += __half2float(q_h[0]) * __half2float(k_h[0]);
                    Sij += __half2float(q_h[1]) * __half2float(k_h[1]);
                    Sij += __half2float(q_h[2]) * __half2float(k_h[2]);
                    Sij += __half2float(q_h[3]) * __half2float(k_h[3]);
                }
                Sij *= softmax_scale;

                float mi_prev = mi;
                mi = max(mi_prev, Sij);

                float exp_prev = expf(mi_prev - mi);
                float exp_curr = expf(Sij - mi);

                li = li * exp_prev + exp_curr;

                for (int k = 0; k < d; k += 4)
                {
                    float2 v_vec_f2 = FETCH_HALF4(Vj[ii * d_padded + k]);
                    const half *v_h = reinterpret_cast<const half *>(&v_vec_f2);

                    Oi[k] = Oi[k] * exp_prev + exp_curr * __half2float(v_h[0]);
                    Oi[k + 1] = Oi[k + 1] * exp_prev + exp_curr * __half2float(v_h[1]);
                    Oi[k + 2] = Oi[k + 2] * exp_prev + exp_curr * __half2float(v_h[2]);
                    Oi[k + 3] = Oi[k + 3] * exp_prev + exp_curr * __half2float(v_h[3]);
                }
            }
        }
        __syncthreads();
    }

    if (row_idx < N)
    {
        half *out_ptr_row = out + qkv_offset + row_idx * d;
        float inv_li = 1.0f / li;

        for (int k = 0; k < d; k += 4)
        {
            float2 final_val_f2;
            half *final_val_h = reinterpret_cast<half *>(&final_val_f2);

            final_val_h[0] = __float2half(Oi[k] * inv_li);
            final_val_h[1] = __float2half(Oi[k + 1] * inv_li);
            final_val_h[2] = __float2half(Oi[k + 2] * inv_li);
            final_val_h[3] = __float2half(Oi[k + 3] * inv_li);

            reinterpret_cast<float2 *>(&out_ptr_row[k])[0] = final_val_f2;
        }
    }
}

void launch_flash_attn_kernel(const torch::Tensor &q,
                              const torch::Tensor &k,
                              const torch::Tensor &v,
                              torch::Tensor &out,
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
    const half *q_ptr = reinterpret_cast<const half *>(q.data_ptr<at::Half>());
    const half *k_ptr = reinterpret_cast<const half *>(k.data_ptr<at::Half>());
    const half *v_ptr = reinterpret_cast<const half *>(v.data_ptr<at::Half>());
    half *out_ptr = reinterpret_cast<half *>(out.data_ptr<at::Half>());

    if (d <= 32)
    {
        flash_attn_kernel<32><<<grid, block, smem>>>(q_ptr, k_ptr, v_ptr, out_ptr, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 64)
    {
        flash_attn_kernel<64><<<grid, block, smem>>>(q_ptr, k_ptr, v_ptr, out_ptr, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 128)
    {
        flash_attn_kernel<128><<<grid, block, smem>>>(q_ptr, k_ptr, v_ptr, out_ptr, B, nh, N, d, Bc, Br, scale);
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

    const int B = q.size(0);
    const int nh = q.size(1);
    const int N = q.size(2);
    const int d = q.size(3);

    float softmax_scale = 1.0 / sqrt(d);

    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    size_t max_sram = props.sharedMemPerBlock;

    int Br = 64;
    int d_padded = d + 4;

    int remaining_sram = max_sram - (Br * d_padded * sizeof(half));
    int Bc = remaining_sram / (2 * d_padded * sizeof(half));

    Bc = std::min(Bc, N);

    torch::Tensor out = torch::zeros_like(q);

    dim3 grid(B, nh, (N + Br - 1) / Br);
    dim3 block(Br);

    size_t smem_bytes = (Br * d_padded + 2 * Bc * d_padded) * sizeof(half);

    launch_flash_attn_kernel(q,
                             k,
                             v,
                             out,
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