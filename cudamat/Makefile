
all: libcudamat.so libcudamat_conv.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	nvcc -O3 \
		-v \
		-gencode=arch=compute_10,code=sm_10 \
	  -gencode=arch=compute_20,code=sm_20 \
		-gencode=arch=compute_30,code=sm_30 \
		--compiler-options '-fPIC' -o libcudamat.so \
		--shared cudamat.cu cudamat_kernels.cu -lcublas -L$(CUDA_LIB)

libcudamat_conv.so: cudamat_conv.cu cudamat_conv_kernels.cu cudamat_conv.cuh cudamat.cuh cudamat_conv_kernels.cuh
	nvcc -O3 \
		-v \
		-gencode=arch=compute_10,code=sm_10 \
	  -gencode=arch=compute_20,code=sm_20 \
		-gencode=arch=compute_30,code=sm_30 \
		--compiler-options '-fPIC' -o libcudamat_conv.so \
		--shared cudamat_conv.cu cudamat_conv_kernels.cu -lcublas -L$(CUDA_LIB)
	
clean:
	rm -rf *.so
