# Modified from Mamba & VMamba
import sys
import warnings
import os
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "FALSE") == "TRUE"

def get_compute_capability():
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    return int(str(capability[0]) + str(capability[1]))
    
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

MODES = ["oflex"]

def get_ext():
    cc_flag = []

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    print("\n\nCUDA_HOME = {}\n\n".format(CUDA_HOME))

    # Check if card has compute capability 8.0 or higher for BFloat16 operations
    if get_compute_capability() < 80:
        warnings.warn("This code uses BFloat16 date type, which is only supported on GPU architectures with compute capability 8.0 or higher")
        
    multi_threads = True
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        print("CUDA version: ", bare_metal_version, flush=True)
        if bare_metal_version < Version("11.6"):
            warnings.warn("CUDA version ealier than 11.6 may leads to performance mismatch.")
        if bare_metal_version < Version("11.2"):
            multi_threads = False
            
    cc_flag.append(f"-arch=sm_{get_compute_capability()}")
    
    if multi_threads:
        cc_flag.extend(["--threads", "4"])

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    sources = dict(
        oflex=[
            "src/cusoflex/selective_scan_oflex.cpp",
            "src/cusoflex/selective_scan_core_fwd.cu",
            "src/cusoflex/selective_scan_core_bwd.cu",
        ],
    )

    names = dict(
        oflex="selective_nd_scan_cuda_oflex",
    )

    ext_modules = [
        CUDAExtension(
            name=names.get(MODE, None),
            sources=sources.get(MODE, None),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                            "-O3",
                            "-std=c++17",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                            "--ptxas-options=-v",
                            "-lineinfo",
                        ]
                        + cc_flag
            },
            include_dirs=[
                Path(this_dir) / "include",
                Path(this_dir) / "include" / "cusoflex",
            ],
        )
        for MODE in MODES
    ]

    return ext_modules

ext_modules = get_ext()
setup(
    name="selective_nd_scan",
    version="0.1.0",
    packages=[],
    author="LongLong Yu",
    author_email="longlong.yu@hdu.edu.cn",
    description="selective nd scan",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension} if ext_modules else {"bdist_wheel": _bdist_wheel,},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
    ],
)
