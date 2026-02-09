from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attn',
    ext_modules=[
        CUDAExtension(
            name='flash_attn.flash_attn',
            sources=['src/flash_attn/flash_attn.cpp', 'src/flash_attn/flash_attn_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)