#!/bin/sh

export CUDA_SDK_PATH=/home/vmnih/NVIDIA_GPU_Computing_SDK
#export CUDA_INSTALL_PATH=/pkgs_local/cuda-4.0
#export CUDA_SDK_PATH=/pkgs_local/cuda-sdk-3.2.16
export PYTHON_INCLUDE_PATH=/usr/include/python2.7/
export NUMPY_INCLUDE_PATH=/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/
#export ATLAS_LIB_PATH=/usr/lib/

make $*

