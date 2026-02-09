#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])

template <int head_dim>
__global__ void flash_attn_kernel(const float *__restrict__ q_ptr,
                                  const float *__restrict__ k_ptr,
                                  const float *__restrict__ v_ptr,
                                  float *out,
                                  const int B,
                                  const int nh,
                                  const int N,
                                  const int d,
                                  const int Bc,
                                  const int Br,
                                  const float softmax_scale)
{
    const int d_padded = d + 4;
    extern __shared__ float s[];
    float *Qi = s;
    float *Kj = &Qi[Br * d_padded];
    float *Vj = &Kj[Bc * d_padded];

    int qkv_offset = (blockIdx.x * nh * N * d) + (blockIdx.y * N * d);
    int i = blockIdx.z * Br;
    int row_idx = i + threadIdx.x;

    float Oi[head_dim];
    float li = 0.0f;
    float mi = -1e20f;

    for (int idx = 0; idx < d; idx++)
        Oi[idx] = 0.0f;

    const int num_float4s = (Br * d) / 4;
    const float4 *q_ptr_4 = reinterpret_cast<const float4 *>(q_ptr + qkv_offset);
    for (int idx = threadIdx.x; idx < num_float4s; idx += blockDim.x)
    {
        int r = idx / (d / 4);
        int c_v = idx % (d / 4);
        int shared_idx = r * d_padded + c_v * 4;
        int global_row = i + r;
        if (global_row < N)
        {
            reinterpret_cast<float4 *>(&Qi[shared_idx])[0] = q_ptr_4[global_row * (d / 4) + c_v];
        }
        else
        {
            reinterpret_cast<float4 *>(&Qi[shared_idx])[0] = {0.f, 0.f, 0.f, 0.f};
        }
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc)
    {
        const int num_float4s_kv = (Bc * d) / 4;
        const float4 *k_ptr_4 = reinterpret_cast<const float4 *>(k_ptr + qkv_offset);
        const float4 *v_ptr_4 = reinterpret_cast<const float4 *>(v_ptr + qkv_offset);

        for (int idx = threadIdx.x; idx < num_float4s_kv; idx += blockDim.x)
        {
            int r = idx / (d / 4);
            int c_v = idx % (d / 4);
            int global_row = j + r;
            int shared_idx = r * d_padded + c_v * 4;

            if (global_row < N)
            {
                reinterpret_cast<float4 *>(&Kj[shared_idx])[0] = k_ptr_4[global_row * (d / 4) + c_v];
                reinterpret_cast<float4 *>(&Vj[shared_idx])[0] = v_ptr_4[global_row * (d / 4) + c_v];
            }
            else
            {
                reinterpret_cast<float4 *>(&Kj[shared_idx])[0] = {0.f, 0.f, 0.f, 0.f};
                reinterpret_cast<float4 *>(&Vj[shared_idx])[0] = {0.f, 0.f, 0.f, 0.f};
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
                    float4 qVal = FETCH_FLOAT4(Qi[(threadIdx.x * d_padded) + jj]);
                    float4 kVal = FETCH_FLOAT4(Kj[(ii * d_padded) + jj]);
                    Sij += (qVal.x * kVal.x) + (qVal.y * kVal.y) + (qVal.z * kVal.z) + (qVal.w * kVal.w);
                }
                Sij *= softmax_scale;

                float mi_prev = mi;
                mi = max(mi_prev, Sij);

                float exp_prev = expf(mi_prev - mi);
                float exp_curr = expf(Sij - mi);

                li = li * exp_prev + exp_curr;

                for (int k = 0; k < d; k += 4)
                {
                    float4 v_vec = FETCH_FLOAT4(Vj[ii * d_padded + k]);

                    Oi[k] = Oi[k] * exp_prev + exp_curr * v_vec.x;
                    Oi[k + 1] = Oi[k + 1] * exp_prev + exp_curr * v_vec.y;
                    Oi[k + 2] = Oi[k + 2] * exp_prev + exp_curr * v_vec.z;
                    Oi[k + 3] = Oi[k + 3] * exp_prev + exp_curr * v_vec.w;
                }
            }
        }
        __syncthreads();
    }

    if (row_idx < N)
    {
        float4 *out_ptr_4 = reinterpret_cast<float4 *>(out + qkv_offset + row_idx * d);
        float inv_li = 1.0f / li;
        for (int k = 0; k < d; k += 4)
        {
            float4 final_val;
            final_val.x = Oi[k] * inv_li;
            final_val.y = Oi[k + 1] * inv_li;
            final_val.z = Oi[k + 2] * inv_li;
            final_val.w = Oi[k + 3] * inv_li;
            out_ptr_4[k / 4] = final_val;
        }
    }
}

void launch_flash_attn_kernel(const float *q,
                              const float *k,
                              const float *v,
                              float *out,
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
        flash_attn_kernel<32><<<grid, block, smem>>>(q, k, v, out, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 64)
    {
        flash_attn_kernel<64><<<grid, block, smem>>>(q, k, v, out, B, nh, N, d, Bc, Br, scale);
    }
    else if (d <= 128)
    {
        flash_attn_kernel<128><<<grid, block, smem>>>(q, k, v, out, B, nh, N, d, Bc, Br, scale);
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

    dim3 grid(B, nh, (N + Br - 1) / Br);
    dim3 block(Br);

    size_t smem_bytes = (Br * d_padded + 2 * Bc * d_padded) * sizeof(float);

    launch_flash_attn_kernel(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
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