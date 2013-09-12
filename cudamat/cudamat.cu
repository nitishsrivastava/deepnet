#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <math.h>
#include "cudamat_kernels.cuh"
#include "cudamat.cuh"

extern "C" {

/* ------------------------------ CUBLAS init/shutdown ------------------------------ */

inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();
    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

extern const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

extern int cublas_init() {
    cublasInit();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

extern int cublas_shutdown() {
    cublasShutdown();
    cudaThreadExit();
    return 0;
}


extern int cuda_set_device(int deviceId) {
    cudaSetDevice(deviceId);
    
    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int init_random(rnd_struct* rnd_state, int seed, char* cudamatpath) {
    unsigned int * host_mults;
    host_mults = (unsigned int*)malloc(NUM_RND_STREAMS * sizeof(unsigned int));
    FILE * pFile;

    pFile = fopen (cudamatpath,"r");

    for (int i = 0; i < NUM_RND_STREAMS; i++) {
        fscanf (pFile, "%u", &host_mults[i]);
    }
    fclose (pFile);

    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned int), (void**)&rnd_state->dev_mults);
    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned long long), (void**)&rnd_state->dev_words);
    cublasSetVector(NUM_RND_STREAMS, sizeof(unsigned int), host_mults, 1, rnd_state->dev_mults, 1);
    //cudaMalloc((void **)&rnd_state->dev_mults, NUM_RND_STREAMS * sizeof(unsigned int));
    //cudaMalloc((void **)&rnd_state->dev_words, NUM_RND_STREAMS * sizeof(unsigned long long));
    //cudaMemcpy(rnd_state->dev_mults, host_mults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, seed);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

/* ------------------------------ Utility routines ------------------------------ */

extern int get_leading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[1] : mat->size[0];
}

extern int get_nonleading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[0] : mat->size[1];
}

extern void set_transpose(cudamat* mat, int is_trans) {
    mat->is_trans = is_trans;
}

inline char get_transpose_char(cudamat* mat) {
    return mat->is_trans ? 't' : 'n';
}

extern void cuda_sync_threads() {
    cudaThreadSynchronize();
}

/* ------------------------------ Allocating/moving data ------------------------------ */

extern int allocate_device_memory(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    cublasStatus stat;

    stat = cublasAlloc(len, sizeof(mat->data_device[0]), (void**)&mat->data_device);

    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
    }

    mat->on_device = 1;
    return 0;
}

extern int allocate_device_memory_sparse(cudamat_sparse* mat) {
    int nnz = mat->nnz, rows = mat->size[0];

    cublasStatus stat;

    stat = cublasAlloc(nnz, sizeof(mat->data_device.data[0]), (void**)&mat->data_device.data);
    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
    }

    stat = cublasAlloc(nnz, sizeof(mat->data_device.indices[0]), (void**)&mat->data_device.indices);
    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
    }

    stat = cublasAlloc(rows + 1, sizeof(mat->data_device.indptr[0]), (void**)&mat->data_device.indptr);
    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
    }

    mat->on_device = 1;
    return 0;
}


extern int copy_to_host(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    if (mat->on_device) {
            cublasGetVector(len, sizeof(mat->data_host[0]), mat->data_device, 1, mat->data_host, 1);

        if (check_cublas_error())
            return CUBLAS_ERROR;
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}

extern int copy_to_device(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];
    int err_code = 0;

    //if (!mat->owns_data)
    //    return VIEW_ERROR;

    if (!mat->on_device) {
        err_code = allocate_device_memory(mat);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, sizeof(mat->data_host[0]), mat->data_host, 1, mat->data_device, 1);
    
    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

extern int copy_sparse_to_device(cudamat_sparse* mat) {
    int len = mat->nnz, rows = mat->size[0];
    int err_code = 0;

    //if (!mat->owns_data)
    //    return VIEW_ERROR;

    if (!mat->on_device) {
        err_code = allocate_device_memory_sparse(mat);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, sizeof(mat->data_host.data[0]), mat->data_host.data, 1, mat->data_device.data, 1);
    if (check_cublas_error())
        return CUBLAS_ERROR;

    cublasSetVector(len, sizeof(mat->data_host.indices[0]), mat->data_host.indices, 1, mat->data_device.indices, 1);
    if (check_cublas_error())
        return CUBLAS_ERROR;

    cublasSetVector(rows + 1, sizeof(mat->data_host.indptr[0]), mat->data_host.indptr, 1, mat->data_device.indptr, 1);
    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}


extern int copy_on_device(cudamat* mat1, cudamat* mat2) {
    int len = mat1->size[0]*mat1->size[1];

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasScopy(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

extern int get_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    int height = source->size[0];
    int width = source->size[1];

    if ((end - start) != target->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    kGetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int set_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    int height = target->size[0];
    int width = target->size[1];

    if ((end - start) != source->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    kSetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int copy_transpose(cudamat* source, cudamat* target) {
    unsigned int height = source->size[0];
    unsigned int width = source->size[1];

    if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    kTranspose<<< grid, threads >>>(target->data_device, source->data_device, height, width);

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int free_device_memory(cudamat* mat) {
    if (mat->owns_data && mat->on_device) {
        cublasStatus stat;

        stat = cublasFree(mat->data_device);
        mat->on_device = 0;

        if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
            return CUBLAS_ERROR;
    }

    return 0;
}

extern int set_shape(cudamat* mat, unsigned int m, unsigned int n) {

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}


extern int reshape(cudamat* mat, unsigned int m, unsigned int n) {
    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}

extern int get_slice(cudamat* source, cudamat* target, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_col * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = source->size[0];
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 0;

    return 0;
}

extern int get_vector_slice(cudamat* source, cudamat* target, unsigned int first_ind, unsigned int last_ind) {
    // source must be a vector.
    if (source->size[0] > 1 && source->size[1] > 1)
        return ERROR_GENERIC;

    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (first_ind >= last_ind)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_ind * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->is_trans = 0;
    target->owns_data = 0;

    if (source->size[0] > 1) {
        if (last_ind > source->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = last_ind - first_ind;
        target->size[1] = 1;
    } else {
        if (last_ind > source->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = 1;
        target->size[1] = last_ind - first_ind;
    }

    return 0;
}

/* ------------------------------ Initialization routines ------------------------------ */

extern void init_from_array(cudamat* mat, float* data, int m, int n) {
    mat->data_host = data;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
}

extern void init_from_sparse_array(cudamat_sparse* mat, float* data, int* indices, int* indptr, int m, int n, int nnz) {
    mat->data_host.data = data;
    mat->data_host.indices = indices;
    mat->data_host.indptr = indptr;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
    mat->nnz = nnz;
}


extern void set_on_device(cudamat* mat) {
  mat->on_device = 1;
}

extern int init_empty(cudamat* mat, int m, int n) {
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 0;
    mat->is_trans = 0;
    mat->owns_data = 1;

    return allocate_device_memory(mat);
}

/* ------------------------------ Random number generation ------------------------------ */
extern int fill_with_rand(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int fill_with_randn(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int sample_bernoulli(rnd_struct* rnd_state, cudamat* mat, cudamat* target) {
    int len = mat->size[0] * mat->size[1];
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kSampleBernoulli<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
extern int sample_bernoulli_tanh(rnd_struct* rnd_state, cudamat* mat, cudamat* target) {
    int len = mat->size[0] * mat->size[1];
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kSampleBernoulliTanh<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
extern int sample_poisson(rnd_struct* rnd_state, cudamat* mat, cudamat* target) {
    int len = mat->size[0] * mat->size[1];
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kSamplePoisson<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
extern int sample_gaussian(rnd_struct* rnd_state, cudamat* mat, cudamat* target, float mult) {
    int len = mat->size[0] * mat->size[1];
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kSampleGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, target->data_device, len, mult);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int perturb_energy(rnd_struct* rnd_state, cudamat* mat, cudamat* target) {
    int len = mat->size[0] * mat->size[1];
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kPerturbEnergy<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int perturb_prob(rnd_struct* rnd_state, cudamat* mat, cudamat* target) {
    int len = mat->size[0] * mat->size[1];
    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kPerturbProb<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int dropout(rnd_struct* rnd_state, cudamat* mat, float dropprob, float val) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kRandomDropout<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len, dropprob, val);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

/* ------------------------------ Algebraic operations ------------------------------ */

extern int add_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

extern int add_col_mult(cudamat* mat, cudamat* vec, cudamat* target, float mult) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddColMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, mult, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_diagonal_scalar(cudamat* mat, float val, cudamat* target) {
    unsigned int w = mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultDiagonalScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, w);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int add_diagonal_scalar(cudamat* mat, float val, cudamat* target) {
    unsigned int w = mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddDiagonalScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, w);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int mult_diagonal(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[1] * vec->size[0] ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultDiagonal<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int add_diagonal(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[1] * vec->size[0] ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddDiagonal<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int add_row_mult(cudamat* mat, cudamat* vec, cudamat* target, float mult) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddRowMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, mult, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int add_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultByColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultByRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int div_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivByColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int div_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivByRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int less_than_eq(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThanEq<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int less_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThan<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int less_than_eq_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThanEqScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int less_than_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int greater_than_eq(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThanEq<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int greater_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThan<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int upper_bound(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kUpperBound<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int lower_bound(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLowerBound<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int greater_than_eq_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThanEqScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int greater_than_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int upper_bound_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kUpperBoundScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int lower_bound_scalar(cudamat* mat, float val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLowerBoundScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int max_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        int shared_mem_size = 32 * sizeof(float) ;
        int w1 = floor(sqrt(w));
        int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
        dim3 gridDim(w1, w2, 1);
        kMaxColumnwise<<<gridDim, 32, shared_mem_size>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int choose_max_and_accumulate(cudamat* mat, cudamat* acc) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !acc->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (acc->size[0] != mat->size[0] || acc->size[1] != mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

        int w1 = floor(sqrt(w));
        int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
        dim3 gridDim(w1, w2, 1);
    kChooseMaxAndAccumulate<<<gridDim,32>>>(mat->data_device, acc->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int choose_max_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != mat->size[0] || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        int shared_mem_size = 32 * sizeof(float) ;
        int w1 = floor(sqrt(w));
        int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
        dim3 gridDim(w1, w2, 1);
        kChooseMaxColumnwise<<<gridDim, 32, shared_mem_size>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int argmax_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        int w1 = floor(sqrt(w));
        int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
        dim3 gridDim(w1, w2, 1);
        kArgMaxColumnwise<<<gridDim,32>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int sqsum_by_axis(cudamat* mat, cudamat* target, int axis, float mult, float p) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        int shared_mem_size = 32 * sizeof(float) ;
        int w1 = floor(sqrt(w));
        int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
        dim3 gridDim(w1, w2, 1);
        kSqSumColumnwise<<<gridDim, 32, shared_mem_size>>>(mat->data_device, target->data_device, w, h, mult, p);
        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int normlimit_by_axis(cudamat* mat, cudamat* target, int axis,
                                   float norm) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != mat->size[0] || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        int shared_mem_size = 32 * sizeof(float) ;
        int w1 = floor(sqrt(w));
        int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
        dim3 gridDim(w1, w2, 1);
        kNormLimitColumnwise<<<gridDim,32, shared_mem_size>>>(mat->data_device, target->data_device, norm, w, h);
        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;
    if (checkCUDAError())
        return CUDA_ERROR;
    return 0;
}


extern int sign(cudamat* mat, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSign<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_cos(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyCos<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_sin(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplySin<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_sigmoid(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplySigmoid<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_tanh(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyTanh<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_abs(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyAbs<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_log_1_plus_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyLog1PlusExp<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_log(cudamat* mat, cudamat* target, float tiny) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLog<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len, tiny);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kExp<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_ceil(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCeil<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_floor(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kFloor<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}



extern int apply_sqrt(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSqrt<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_pow(cudamat* mat, float pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kPow<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_pow_matrix(cudamat* mat, cudamat* pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kPowMatrix<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int compute_cross_entropy(cudamat* mat, cudamat* pow, cudamat* target, float tiny) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCrossEntropy<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow->data_device, target->data_device, len, tiny);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int compute_cross_entropy_bernoulli(cudamat* mat, cudamat* pow, cudamat* target, float tiny) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCrossEntropyBernoulli<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow->data_device, target->data_device, len, tiny);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int correct_preds(cudamat* mat, cudamat* pow, cudamat* target, float cutoff) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCorrectPreds<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow->data_device, target->data_device, len, cutoff);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int reciprocal(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kReciprocal<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int dot(cudamat* mat1, cudamat* mat2, cudamat* target, float beta, float alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
        get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
        get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    int m = get_leading_dimension(mat1),
        k = get_leading_dimension(mat2),
        n = get_nonleading_dimension(mat2);

    cublasSgemm(get_transpose_char(mat1), get_transpose_char(mat2), 
                m, n, k,
                alpha, mat1->data_device, mat1->size[0],
                mat2->data_device, mat2->size[0],
                beta, target->data_device, target->size[0]);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    cudaThreadSynchronize();

    return 0;
}

extern int sparse_dot(cudamat_sparse* mat1, cudamat* mat2, cudamat* target, float beta, float alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    int m = mat1->size[0],
        k = mat1->size[1],
        k2 = mat2->size[0],
        n = mat2->size[1];

    if (k != k2) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    unsigned int grid_x = m / COPY_BLOCK_SIZE;
    if (m % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = n / COPY_BLOCK_SIZE;
    if (n % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    kSparseDot<<<grid, threads>>>(m, n, k, mat1->data_device.data,
        mat1->data_device.indptr,
        mat1->data_device.indices,
        mat2->data_device, target->data_device, beta, alpha);
    if (check_cublas_error())
        return CUBLAS_ERROR;

    cudaThreadSynchronize();

    return 0;
}


extern float vdot(cudamat* mat1, cudamat* mat2, int* err_code) {
    int len = mat1->size[0]*mat1->size[1];
    float res;

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans) {
        *err_code = ERROR_TRANSPOSEDNESS;
        return 0;
    }

    if (mat2->size[0] * mat2->size[1] != len) {
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
        return 0;
    }

    res = cublasSdot(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

/* Perform the operation mat1 = mat1 + alpha * mat2. mat1 and mat2 must
   have the same transposedness. */
extern int add_mult(cudamat* mat1, cudamat* mat2, float alpha) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasSaxpy(len, alpha, mat2->data_device, 1, mat1->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}
extern int add_mult_sign(cudamat* mat1, cudamat* mat2, float mult) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddMultSign<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, len, mult);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}


extern int add_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAdd<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int subtract_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSubtract<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int divide_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivide<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Elementwise multiplication of 2 matrices */
extern int mult_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_sin_deriv(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSinDeriv<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_cos_deriv(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCosDeriv<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_logistic_deriv(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLogisticDeriv<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_tanh_deriv(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kTanhDeriv<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_rectified_linear_deriv(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kRectifiedLinearDeriv<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int apply_rectified_linear_smooth_deriv(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kRectifiedLinearSmoothDeriv<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int assign_scalar(cudamat* mat, float alpha) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kAssignScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int mult_by_scalar(cudamat* mat, float alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int divide_by_scalar(cudamat* mat, float alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivideScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int add_scalar(cudamat* mat, float alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern float euclid_norm(cudamat* mat, int* err_code) {
    int len = mat->size[0]*mat->size[1];

    float res =  cublasSnrm2(len, mat->data_device, 1);

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}
extern int selectRows(cudamat* source, cudamat* target, cudamat* indices){
    const int nRetRows = indices->size[1];

    if (nRetRows==0) return 0;

    dim3 gridDim((nRetRows+31)/32);
    dim3 blockDim(32);

    kSelectRows<<<gridDim, blockDim>>>(source->data_device, target->data_device, indices->data_device, nRetRows, source->size[0], source->size[1]);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}


extern int swapColumns(cudamat* source, cudamat* target, cudamat* indices1, cudamat* indices2){
    const int cols = indices1->size[1]*indices1->size[0],
                 h = source->size[0],
                 w = source->size[1];

    kSwapColumns<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(source->data_device, target->data_device, indices1->data_device, indices2->data_device, cols, w, h);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int setSelectedRows(cudamat* target, cudamat* source, cudamat* indices){
    const int nSetRows = indices->size[1];

    if (nSetRows==0)
        return 0;

    dim3 gridDim((nSetRows+31)/32);
    dim3 blockDim(32);

    kSetSelectedRows<<<gridDim, blockDim>>>(target->data_device, source->data_device, indices->data_device, nSetRows, target->size[0], target->size[1]);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int generate_translations_big_var_off(cudamat* source, cudamat* target, cudamat* off_x, cudamat* off_y, int source_w, int target_w, int num_channels) {
    dim3 kernelBlockGrid(source->size[1], 1, 1);
    dim3 kernelBlockDim(512, 1, 1);

    kGenerateTranslationsBigVarOff<<<kernelBlockGrid, kernelBlockDim>>>(source->data_device, target->data_device, off_x->data_device, off_y->data_device, source_w, target_w, num_channels);

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int blockify(cudamat* source, cudamat* target, int blocksize) {
    dim3 kernelBlockGrid(source->size[1], 1, 1);
    dim3 kernelBlockDim(512, 1, 1);
    kBlockify<<<kernelBlockGrid, kernelBlockDim>>>(source->data_device, target->data_device, source->size[0], blocksize);
    if (checkCUDAError())
        return CUDA_ERROR;
    return 0;
}


extern int softmax(cudamat* mat, cudamat* target) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    
    int shared_mem_size = 32 * sizeof(float) ;

    int w1 = floor(sqrt(w));
    int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
    dim3 gridDim(w1, w2, 1);
    kSoftMax<<<gridDim, 32, shared_mem_size>>>(mat->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int softmax_overwrite(cudamat* mat) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    int shared_mem_size = 32 * sizeof(float) ; 
    int w1 = floor(sqrt(w));
    int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
    dim3 gridDim(w1, w2, 1);
    kSoftMaxOverwrite<<<gridDim, 32, shared_mem_size>>>(mat->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_softmax_grad(cudamat* mat, cudamat* labels, cudamat* target) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (labels->size[0] != 1 || labels->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    
    kSoftMaxGrad<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, labels->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int get_softmax_correct(cudamat* mat, cudamat* labels, cudamat* target) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != 1 || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (labels->size[0] != 1 || labels->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    
    int w1 = floor(sqrt(w));
    int w2 = w / w1 + (w % w1 == 0 ? 0 : 1);
    dim3 gridDim(w1, w2, 1);
    kSoftMaxCorrect<<<gridDim, 32>>>(mat->data_device, labels->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int accumulate_columns(cudamat* mat, cudamat* indices, cudamat* target, float mult, int avg) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1],
                 w2 = target->size[1];

    if (!mat->on_device || !indices->on_device|| !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (indices->size[0] != 1 || indices->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (NUM_VECTOR_OP_THREADS_PER_BLOCK < w2)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    
    kAccumulateColumns<<<h, NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, indices->data_device, target->data_device, w, w2, h, mult, avg);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int get_softmax_cross_entropy(cudamat* mat, cudamat* labels, cudamat* target, float tiny) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != 1 || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (labels->size[0] != 1 || labels->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    
    kSoftMaxCrossEntropy<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, labels->data_device, target->data_device, w, h, tiny);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int expand(cudamat* source, cudamat* indices, cudamat* target){
    unsigned int h = source->size[0],
                 w = source->size[1],
                 w2 = target->size[1];

    if (!source->on_device || !indices->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (indices->size[0] != 1 || indices->size[1] != w2)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kExpand<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(source->data_device, indices->data_device, target->data_device, h, w, w2);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}


extern int expand_and_add(cudamat* source, cudamat* mat, cudamat* indices, cudamat* target, float mult){
    unsigned int h = source->size[0],
                 w = source->size[1],
                 w2 = mat->size[1];

    if (!source->on_device || !mat->on_device || !indices->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (indices->size[0] != 1 || indices->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (mat->size[0] != h)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kExpandAndAdd<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(source->data_device, mat->data_device, indices->data_device, target->data_device, w, h, mult, w2);
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}



}
