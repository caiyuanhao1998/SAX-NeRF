"""Build the original SAX-NeRF hash encoder as an inference wheel.

Example: CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST='8.6;12.0+PTX'
python demo/build_encoder.py bdist_wheel --dist-dir /path/to/wheels
"""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

source = Path(__file__).resolve().parents[1] / "src/encoder/hashencoder/src"
setup(
    name="sax_hash_encoder",
    version="1.0.0",
    description="Original SAX-NeRF CUDA hash-grid encoder",
    ext_modules=[CUDAExtension(
        name="_hash_encoder",
        sources=[str(source / "bindings.cpp"), str(source / "hashencoder.cu")],
        extra_compile_args={"cxx": ["-O3"], "nvcc": [
            "-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
