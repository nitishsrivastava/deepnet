#!/bin/sh

export CUDA_SDK_PATH=/pkgs_local/cuda-sdk-4.0/
export CUDA_INSTALL_PATH=/pkgs_local/cuda-4.2
export PYTHON_INCLUDE_PATH=$HOME/epd/include/python2.7/
export NUMPY_INCLUDE_PATH=/usr/include/python2.7/numpy/

make $*

