cudamat:
	nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared cudamat.cu cudamat_kernels.cu -lcublas -L/pkgs_local/cuda-4.2/lib64
	nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudalearn.so --shared learn.cu learn_kernels.cu -lcublas -L/pkgs_local/cuda-4.2/lib64


clean:
	rm *.linkinfo *.pyc *.so
