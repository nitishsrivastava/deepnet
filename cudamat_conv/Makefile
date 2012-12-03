

# cudamat:
# 	echo building cudamat
# 	nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat.so --shared cudamat.cu cudamat_kernels.cu -lcublas -L/pkgs_local/cuda-4.0/lib64
# 	nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudalearn.so --shared learn.cu learn_kernels.cu -lcublas -L/pkgs_local/cuda-4.0/lib64

# clean:
# 	rm *.linkinfo *.pyc *.so

MODELNAME := _ConvNet

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I./include -I./include/common -I./include/cudaconv2 -I./include/nvmatrix -I./
LIB := -lpthread -L$(ATLAS_LIB_PATH) -L$(CUDA_INSTALL_PATH)/lib64 -lcblas

USECUBLAS   := 1

PYTHON_VERSION=$(shell python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB += -lpython$(PYTHON_VERSION)

GENCODE_ARCH := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/cudamat_conv.cu  src/cudaconv2/*.cu src/nvmatrix/*.cu)
CU_DEPS				:= $(shell echo include/cudamat_conv.cuh include/cudaconv2/*.cuh include/nvmatrix/*.cuh)
CCFILES				:= $(shell echo src/common/*.cpp)
C_DEPS				:= $(shell echo include/common/*.h)


include common-gcc-cuda-3.2.mk

makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src/cudaconv2
	$(VERBOSE)mkdir -p $(OBJDIR)/src/nvmatrix
	$(VERBOSE)mkdir -p $(OBJDIR)/src/common
	$(VERBOSE)mkdir -p $(OBJDIR)/src/
	$(VERBOSE)mkdir -p $(TARGETDIR)
