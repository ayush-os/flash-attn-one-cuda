#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor flash_attn_cuda_forward(
    torch::Tensor q, 
    torch::Tensor k, 
    torch::Tensor v
);

// Python Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_cuda_forward, "My Flash Attention forward (CUDA)");
}