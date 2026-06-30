from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="rtgs-bvh-cuda",
    ext_modules=[
        CUDAExtension(
            name="rtgs_bvh_cuda",
            sources=[str(ROOT / "csrc" / "bvh_bindings.cpp"), str(ROOT / "csrc" / "bvh_cuda.cu")],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
