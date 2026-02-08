cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attention LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

add_library(flash_attn SHARED flash_attn.cpp flash_attn_kernel.cu)

target_link_libraries(flash_attn PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET flash_attn PROPERTY CUDA_ARCHITECTURES 80)

target_compile_options(flash_attn PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)