#!/bin/sh

export CUDA_SDK_PATH=$HOME/NVIDIA_GPU_Computing_SDK
export CUDA_INSTALL_PATH=/pkgs_local/cuda-4.2/
export PYTHON_INCLUDE_PATH=/usr/include/python2.7/
export NUMPY_INCLUDE_PATH=/usr/include/python2.7/numpy/
export ATLAS_LIB_PATH=/usr/lib/atlas-base/atlas
make $*

