#include "cudamat_kernels.cuh"
#include "float.h"
const int NUM_THREADS = 32;

__device__ void reduceToMax(float* sdata, unsigned int tid){

  //Synchronize threads to share shared memory data
  __syncthreads();

  float mySum = sdata[tid];

  // do reduction in shared mem
  if (NUM_THREADS >= 512) { if (tid < 256) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 256]); } __syncthreads(); }
  if (NUM_THREADS >= 256) { if (tid < 128) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 128]); } __syncthreads(); }
  if (NUM_THREADS >= 128) { if (tid <  64) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid +  64]); } __syncthreads(); }

  if (NUM_THREADS == 32){
    if (tid < 16)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); }
    }
  }
  else
  {
    if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  64) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 32]); }
      if (NUM_THREADS >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); }
    }
  }
}

__device__ void reduceToSumLocal(float* sdata, unsigned int tid)
{

  //Synchronize threads to share shared memory data
  __syncthreads();

  float mySum = sdata[tid];

  // do reduction in shared mem
  if (NUM_THREADS >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
  if (NUM_THREADS >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
  if (NUM_THREADS >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

  if (NUM_THREADS == 32){
    if (tid < 16)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
  }
  else
  {
    if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
      if (NUM_THREADS >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
      if (NUM_THREADS >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
      if (NUM_THREADS >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
      if (NUM_THREADS >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
      if (NUM_THREADS >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
  }
}

/* ------------------------- Random number generation ------------------------- */

__global__ void kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // The initial x is the seed and the initial carry is 1
  unsigned long long rndWord = ((unsigned long long)seed << 32) + 1;
  const unsigned int rndMult = rndMults[idx];
  /*
   * Run the chain for a few steps so that all the streams have a chance
   * to differentiate. They start out generating similar random numbers
   * because all the multipliers are similar.
   */
  for(unsigned int i = 0; i < NUM_RND_BURNIN; i++) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
  }
  rndWords[idx] = rndWord;
}

__global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    gData[i] = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
  }
  rndWords[idx] = rndWord;
}

__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  float rnd1, rnd2, R, T;
  for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    T = 2 * PI * rnd2;
    R = sqrtf(-2 * __logf(rnd1));
    gData[i] = R * __cosf(T);
    if (i + NUM_RND_STREAMS < numElements)
      gData[i + NUM_RND_STREAMS] = R * __sinf(T);
  }
  rndWords[idx] = rndWord;
}

__global__ void kRandomDropout(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements, float dropprob, float val) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    gData[i] = ((__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f) > dropprob ? gData[i]:val;
  }
  rndWords[idx] = rndWord;
}

__global__ void kSampleBernoulli(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float* target, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    target[i] = ((__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f) < gData[i] ? 1:0;
  }
  rndWords[idx] = rndWord;
}
__global__ void kSampleBernoulliTanh(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float* target, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    target[i] = ((__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f) < (1.0 + gData[i]) / 2.0 ? 1:0;
  }
  rndWords[idx] = rndWord;
}

__global__ void kSamplePoisson(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float* target, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    target[i] = gData[i];
  }
  rndWords[idx] = rndWord;
}

__global__ void kSampleGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float* target, unsigned int numElements, float mult) {

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];

  float rnd1, rnd2, R, T;
  for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    T = 2 * PI * rnd2;
    R = sqrtf(-2 * __logf(rnd1));
    target[i] = gData[i] + mult * R * __cosf(T);
    if (i + NUM_RND_STREAMS < numElements)
      target[i + NUM_RND_STREAMS] = gData[i + NUM_RND_STREAMS] + mult * R * __sinf(T);
  }
  rndWords[idx] = rndWord;
}

__global__ void kPerturbEnergy(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float* target, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];
  float rnd;

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    rnd = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    target[i] = gData[i] - __logf( - __logf(rnd));
  }
  rndWords[idx] = rndWord;
}

__global__ void kPerturbProb(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float* target, unsigned int numElements) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long rndWord = rndWords[idx];
  const unsigned int rndMult = rndMults[idx];
  float rnd;

  for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
    rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    rnd = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    target[i] = - gData[i] /  __logf(rnd);
  }
  rndWords[idx] = rndWord;
}


/* ------------------------- Data copying ------------------------- */

/*
   Copy row slice from source to target. There is a block for every 32x32 chunk being copied.
 */
__global__ void kGetRowSlice(float* source, float* target, int start, int end, int width, int height) {
  const int row = start + blockIdx.x * 32 + threadIdx.x;
  const int start_col = blockIdx.y * 32;
  const int end_col = (start_col + 32 < width) ? start_col + 32: width;
  const int target_height = end - start;
  if (row < end) {
    for (int cur_col = start_col; cur_col < end_col; cur_col++)
      target[cur_col * target_height + row - start] = source[cur_col * height + row];
  }
}

__global__ void kSetRowSlice(float* source, float* target, int start, int end, int width, int height) {
  const int row = start + blockIdx.x * 32 + threadIdx.x;
  const int start_col = blockIdx.y * 32;
  const int end_col = (start_col + 32 < width) ? start_col + 32: width;
  const int source_height = end - start;
  if (row < end) {
    for (int cur_col = start_col; cur_col < end_col; cur_col++)
      target[cur_col * height + row] = source[cur_col * source_height + row - start];
    //source[cur_col * height + row - start] = target[cur_col * target_height + row];
  }
}

__global__ void kTranspose(float *odata, float *idata, int width, int height) {
  __shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
  unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

  if((xIndex < width) && (yIndex < height)) {
    unsigned int index_in = yIndex * width + xIndex;

    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
  yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

  if((xIndex < height) && (yIndex < width)) {
    unsigned int index_out = yIndex * height + xIndex;

    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

/* ------------------------- Mathematical operations ------------------------- */

__global__ void kLessThan(float* mat1, float* mat2, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat1[i] < mat2[i];
}

__global__ void kLessThanEq(float* mat1, float* mat2, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat1[i] <= mat2[i];
}

__global__ void kLessThanScalar(float* mat, float val, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] < val;
}

__global__ void kLessThanEqScalar(float* mat, float val, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] <= val;
}

__global__ void kGreaterThan(float* mat1, float* mat2, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat1[i] > mat2[i];
}

__global__ void kGreaterThanEq(float* mat1, float* mat2, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat1[i] >= mat2[i];
}

__global__ void kGreaterThanScalar(float* mat, float val, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] > val;
}

__global__ void kGreaterThanEqScalar(float* mat, float val, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] >= val;
}

__global__ void kUpperBound(float* mat1, float* mat2, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat1[i] > mat2[i] ? mat2[i] : mat1[i];
}

__global__ void kLowerBound(float* mat1, float* mat2, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat1[i] < mat2[i] ? mat2[i] : mat1[i];
}

__global__ void kUpperBoundScalar(float* mat, float val, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] > val ? val:mat[i];
}

__global__ void kLowerBoundScalar(float* mat, float val, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] < val ? val:mat[i];
}

__global__ void kSparseDot(int m, int n, int k, float *data, int* indptr, int* indices, float *dense_data, float* target, float beta, float alpha) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    const int start = indptr[row];
    const int end = indptr[row + 1];
    float sum = 0.f;
    for (int i = start; i < end; i++) {
      sum += data[i]  * dense_data[col * k + indices[i]];
    }
    const int pos = col * m + row;
    target[pos] = alpha * sum + ((beta == 0) ? 0 : beta * target[pos]);
  }
}

__global__ void kSign(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] ? copysignf(1., mat[i]) : 0;
}

__global__ void kApplySin(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = __sinf(mat[i]);
}

__global__ void kApplyCos(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = __cosf(mat[i]);
}

__global__ void kApplySigmoid(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = 1 / (1 + __expf(-mat[i]));
}

__global__ void kApplyTanh(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  float mat_i, exp2x;
  for (unsigned int i = idx; i < len; i += numThreads) {
    mat_i = mat[i];
    exp2x = __expf(2 * mat_i);
    target[i] = 1 - 2 / (exp2x + 1);
  }
}

__global__ void kApplyAbs(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] * ((mat[i] > 0) - (mat[i] < 0));
}

__global__ void kApplyLog1PlusExp(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  float mat_i;
  for (unsigned int i = idx; i < len; i += numThreads) {
    mat_i = mat[i];
    if (mat_i > 0)
      target[i] = (__logf(1 + __expf(-mat_i)) + mat_i);
    else
      target[i] = __logf(1 + __expf(mat_i));
  }
}

__global__ void kLog(float* mat, float* target, unsigned int len, float tiny) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = __logf(mat[i] + tiny);
}

__global__ void kExp(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = __expf(mat[i]);
}

__global__ void kCeil(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = ceil(mat[i]);
}

__global__ void kFloor(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = floor(mat[i]);
}

__global__ void kSqrt(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = sqrt(mat[i]);
}

__global__ void kPow(float* mat, float pow, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = powf(mat[i], pow);
}

__global__ void kPowMatrix(float* mat, float* pow, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = powf(mat[i], pow[i]);
}

__global__ void kCrossEntropy(float* mat, float* p, float* target, unsigned int len, float tiny) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = -mat[i] * __logf(p[i] + tiny);
}

__global__ void kCrossEntropyBernoulli(float* mat, float* p, float* target, unsigned int len, float tiny) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads)
    target[i] = -mat[i] * __logf(p[i] + tiny) - (1 - mat[i]) * __logf(1 - p[i] + tiny);
}

__global__ void kCorrectPreds(float* mat, float* p, float* target, unsigned int len, float cutoff) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads)
    target[i] = mat[i] * (p[i] >= cutoff) + (1 - mat[i]) * (p[i] < cutoff);
}

__global__ void kReciprocal(float* mat, float* target, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) target[i] = 1. / mat[i];
}

__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] + vec[i % height];
  }
}
__global__ void kAddDiagonalScalar(float* mat, float val, float* tgtMat, unsigned int width) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width; i += numThreads) {
    tgtMat[width*i + i] = mat[width*i + i] + val;
  }
}

__global__ void kAddDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width; i += numThreads) {
    tgtMat[width*i + i] = mat[width*i + i] + vec[i];
  }
}

__global__ void kMultDiagonalScalar(float* mat, float val, float* tgtMat, unsigned int width) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width; i += numThreads) {
    tgtMat[width*i + i] = mat[width*i + i] * val;
  }
}

__global__ void kMultDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width; i += numThreads) {
    tgtMat[width*i + i] = mat[width*i + i] * vec[i];
  }
}
__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] + vec[i / height];
  }
}

__global__ void kAddColMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] + mult * vec[i % height];
  }
}

__global__ void kAddRowMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] + mult * vec[i / height];
  }
}
__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] * vec[i % height];
  }
}

__global__ void kDivByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] / vec[i / height];
  }
}

__global__ void kDivByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] / vec[i % height];
  }
}

__global__ void kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width * height; i += numThreads) {
    tgtMat[i] = mat[i] * vec[i / height];
  }
}
__global__ void kAddMultSign(float* a, float* b, unsigned int numEls, float mult) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    a[i] = a[i] + ((b[i] > 0) ? mult : ((b[i] < 0) ? -mult : 0));
  }
}
__global__ void kAdd(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] + b[i];
  }
}

__global__ void kSubtract(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] - b[i];
  }
}

__global__ void kDivide(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] / b[i];
  }
}

__global__ void kMult(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] * b[i];
  }
}

__global__ void kCosDeriv(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = -a[i] * __sinf(b[i]);
  }
}

__global__ void kSinDeriv(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] * __cosf(b[i]);
  }
}

__global__ void kLogisticDeriv(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] * b[i] * (1.0 - b[i]);
  }
}

__global__ void kTanhDeriv(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] * (1.0 + b[i]) * (1.0 - b[i]) * 0.5;
  }
}

__global__ void kRectifiedLinearDeriv(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] * (b[i] > 0 ? 1 : 0);
  }
}

__global__ void kRectifiedLinearSmoothDeriv(float* a, float* b, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] * (1 - __expf(-b[i]));
  }
}

__global__ void kMultScalar(float* mat, float alpha, float* dest, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) {
    dest[i] = alpha * mat[i];
  }
}

__global__ void kAssignScalar(float* dest, float alpha, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) {
    dest[i] = alpha;
  }
}

__global__ void kDivideScalar(float* mat, float alpha, float* dest, unsigned int len) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < len; i += numThreads) {
    dest[i] = mat[i] / alpha;
  }
}

__global__ void kAddScalar(float* a, float alpha, float* dest, unsigned int numEls) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < numEls; i += numThreads) {
    dest[i] = a[i] + alpha;
  }
}


__global__ void kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows){
  __shared__ int sourceRowIndices[32];
  const int startTargetRowI = blockIdx.x * 32;
  const int tid = threadIdx.x;
  const int localNRowIs = min(32, nRowIs-startTargetRowI);

  // cooperatively load 32 row indices
  if (tid < localNRowIs){
    sourceRowIndices[tid] = int(indices[startTargetRowI + tid]);
    if (sourceRowIndices[tid]<0)
      sourceRowIndices[tid] += nSourceRows;
    if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nSourceRows)
      sourceRowIndices[tid] = -1;
  }
  __syncthreads();

  // copy 32 rows
  for (int i=0; i<localNRowIs; i++){
    const int targetRowI = startTargetRowI + i, sourceRowI = sourceRowIndices[i];
    for (int colI=tid; colI<nCols; colI+=32)
      target[targetRowI * nCols + colI] = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
  }
}

__global__ void kSwapColumns(float* source, float* target, float* indices1, float* indices2, int cols, int width, int height){
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  float temp;
  unsigned int column, row, source_pos, target_pos;
  for (unsigned int i = idx; i < height * cols; i += numThreads) {
    column = i / height;
    row = i % height;
    source_pos = height * (int)indices1[column] + row;
    target_pos = height * (int)indices2[column] + row;
    temp = source[source_pos];
    source[source_pos] = target[target_pos];
    target[target_pos] = temp;
  }
}

__global__ void kSetSelectedRows(float* target, float* source, float* indices, int nRowIs, int nCols, int nTargetRows){
  __shared__ int targetRowIndices[32];
  const int startSourceRowI = blockIdx.x * 32;
  const int tid = threadIdx.x;
  const int localNRowIs = min(32, nRowIs-startSourceRowI);

  // cooperatively load 32 row indices
  if (tid < localNRowIs){
    targetRowIndices[tid] = int(indices[startSourceRowI + tid]);
    if (targetRowIndices[tid]<0)
      targetRowIndices[tid] += nTargetRows;
    if (targetRowIndices[tid]<0 || targetRowIndices[tid]>=nTargetRows)
      targetRowIndices[tid] = -1;
  }
  __syncthreads();

  // copy 32 rows
  for (int i=0; i<localNRowIs; i++){
    const int sourceRowI = startSourceRowI + i, targetRowI = targetRowIndices[i];
    for (int colI=tid; colI<nCols; colI+=32)
      target[targetRowI * nCols + colI] = targetRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
  }
}

__global__ void kBlockify(float* source, float* target, int numdims, int blocksize) {
  const unsigned int idx = threadIdx.x;
  const unsigned int numThreads = blockDim.x;
  const int off = blockIdx.x * numdims;

  for (unsigned int target_ind = idx; target_ind < numdims; target_ind += numThreads) {
    const int block = target_ind / blocksize;
    target[off + target_ind] = source[off + block * blocksize];
  }
}

__global__ void kGenerateTranslationsBigVarOff(float* source, float* target, float* off_x_arr, float* off_y_arr, int source_w, int target_w, int num_channels) {
  const unsigned int idx = threadIdx.x;
  const unsigned int numThreads = blockDim.x;

  int target_x, target_y;
  int pad = (source_w - target_w)/2;
  int target_tile_size = target_w * target_w;
  int source_tile_size = source_w * source_w;

  int off_x = off_x_arr[blockIdx.x];
  int off_y = off_y_arr[blockIdx.x];
  int target_off = blockIdx.x * target_tile_size;
  int source_off = blockIdx.x * source_tile_size + (pad + off_x) * source_w + (pad + off_y);

  for (unsigned int target_ind = idx; target_ind < target_tile_size; target_ind += numThreads) {
    target_x = target_ind / target_w;
    target_y = target_ind - target_x * target_w;

    for (unsigned int ch = 0; ch < num_channels; ch += 1) {
      target[num_channels*(target_off + target_x * target_w + target_y) + ch] = source[num_channels*(source_off + target_x * source_w + target_y) + ch];
    }
  }
}

__global__ void kSoftMaxGrad(float* mat, float* labels, float* target, unsigned int width, unsigned int height) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width*height; i += numThreads) {
    target[i] = mat[i] - ((int)labels[i / height] == i % height ? 1 : 0);
  }
}

__global__ void kSoftMaxCrossEntropy(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float tiny) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width; i += numThreads) {
    target[i] = -__logf(mat[height * i + (int)labels[i]] + tiny);
  }
}

__global__ void kSoftMaxCorrect(float* mat, float* labels, float* target, unsigned int width, unsigned int height) {
  __shared__ float max_vals[32];
  __shared__ unsigned int max_val_args[32];
  float cur_max = -FLT_MAX;
  unsigned int cur_argmax = 0;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
        cur_argmax = i;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    max_val_args[threadIdx.x] = cur_argmax;
    __syncthreads();
    if (threadIdx.x == 0) {
      cur_max = -FLT_MAX;
      cur_argmax = 0;
      for (unsigned int i = 0; i < blockDim.x; i++)
        if (max_vals[i] > cur_max) {
          cur_max = max_vals[i];
          cur_argmax = max_val_args[i];
        }   
      target[column] = (cur_argmax == (int)labels[column]) ? 1 : 0;
    }
  }
}


__global__ void kSoftMax(float* mat, float* target, unsigned int width, unsigned int height) {
  extern __shared__ float max_vals[] ;
  float cur_max = -FLT_MAX;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    max_vals[threadIdx.x]=-FLT_MAX;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    reduceToMax(max_vals, threadIdx.x);
    __syncthreads();
    cur_max = max_vals[0] ; 
    __syncthreads();
    val = 0;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val += __expf(cur_data[i]-cur_max);
    }
    max_vals[threadIdx.x] = val;
    reduceToSumLocal(max_vals, threadIdx.x);
    __syncthreads();
    float norm = max_vals[0] ; 
    float *cur_target = &target[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      cur_target[i] = __expf(cur_data[i]-cur_max) / norm ;
    }
  }
}

__global__ void kSoftMaxOverwrite(float* mat, unsigned int width, unsigned int height) {
  extern __shared__ float max_vals[] ;
  float cur_max = -FLT_MAX;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    max_vals[threadIdx.x]=-FLT_MAX;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    reduceToMax(max_vals, threadIdx.x);
    __syncthreads();
    cur_max = max_vals[0] ;
    __syncthreads();
    val = 0;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      cur_data[i] = __expf(cur_data[i]-cur_max);
      val += cur_data[i];
    }
    max_vals[threadIdx.x] = val;
    reduceToSumLocal(max_vals, threadIdx.x);
    __syncthreads();
    float norm = max_vals[0] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      cur_data[i] /= norm;
    }
  }
}

__global__ void kChooseMaxAndAccumulate(float* mat, float* target, unsigned int width, unsigned int height) {
  __shared__ float max_vals[32];
  __shared__ unsigned int max_val_args[32];
  float cur_max = -FLT_MAX;
  unsigned int cur_argmax = 0;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    float *target_data = &target[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
        cur_argmax = i;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    max_val_args[threadIdx.x] = cur_argmax;
    __syncthreads();
    if (threadIdx.x == 0) {
      cur_max = -FLT_MAX;
      cur_argmax = 0;
      for (unsigned int i = 0; i < blockDim.x; i++)
        if (max_vals[i] > cur_max) {
          cur_max = max_vals[i];
          cur_argmax = max_val_args[i];
        }   
      target_data[cur_argmax] += 1;
    }
  }
}
__global__ void kChooseMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
  __shared__ float max_vals[32];
  __shared__ unsigned int max_val_args[32];
  float cur_max = -FLT_MAX;
  unsigned int cur_argmax = 0;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    float *target_data = &target[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      target[i] = 0;
      if (val > cur_max) {
        cur_max = val;
        cur_argmax = i;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    max_val_args[threadIdx.x] = cur_argmax;
    __syncthreads();
    if (threadIdx.x == 0) {
      cur_max = -FLT_MAX;
      cur_argmax = 0;
      for (unsigned int i = 0; i < blockDim.x; i++)
        if (max_vals[i] > cur_max) {
          cur_max = max_vals[i];
          cur_argmax = max_val_args[i];
        }   
      target_data[cur_argmax] = 1;
    }
  }
}

__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
  extern __shared__ float max_vals[] ;
  float cur_max = -FLT_MAX;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) cur_max = val;
    }
    max_vals[threadIdx.x] = cur_max;
    reduceToMax(max_vals, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) target[column] = max_vals[0];
  }
}

__global__ void kArgMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
  __shared__ float max_vals[32];
  __shared__ unsigned int max_val_args[32];
  float cur_max = -FLT_MAX;
  unsigned int cur_argmax = 0;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) {
        cur_max = val;
        cur_argmax = i;
      }
    }
    max_vals[threadIdx.x] = cur_max;
    max_val_args[threadIdx.x] = cur_argmax;
    __syncthreads();
    if (threadIdx.x == 0) {
      cur_max = -FLT_MAX;
      cur_argmax = 0;
      for (unsigned int i = 0; i < blockDim.x; i++)
        if (max_vals[i] > cur_max) {
          cur_max = max_vals[i];
          cur_argmax = max_val_args[i];
        }   
      target[column] = cur_argmax;
    }
  }
}

__global__ void kSqSumColumnwise(float* mat, float* target, unsigned int width, unsigned int height, float mult, float p) {
  extern __shared__ float sum_vals[];
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float cur_sum = 0;
    float *cur_data = &mat[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      cur_sum += cur_data[i]*cur_data[i];
    }
    sum_vals[threadIdx.x] = cur_sum;
    reduceToSumLocal(sum_vals, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) target[column] = p * target[column] + mult * sum_vals[0];
  }
}

__global__ void kNormLimitColumnwise(float* mat, float* target, float norm, unsigned int width, unsigned int height) {
  extern __shared__ float sum_vals[];
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float cur_sum = 0;
    float *cur_data = &mat[column * height] ; 
    float *target_data = &target[column * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      cur_sum += cur_data[i]*cur_data[i];
    }
    sum_vals[threadIdx.x] = cur_sum;
    reduceToSumLocal(sum_vals, threadIdx.x);
    __syncthreads();
    cur_sum = sqrt(sum_vals[0]);
    cur_sum = (cur_sum < norm) ? 1: (norm / cur_sum);
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      target_data[i] = cur_data[i] * cur_sum;
    }
    __syncthreads();
  }
}

__global__ void kExpand(float* source, float* indices, float* target, int height, int width, int target_width){
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < target_width*height; i += numThreads) {
    const int pos = height * (int)indices[i / height] + i % height;
    target[i] = (pos < height * width)? source[pos] : 1.0/0.0 - 1.0/0.0;
  }
}


__global__ void kExpandAndAdd(float* source, float* mat, float* indices, float* target, int width, int height, float mult, int width2){
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numThreads = blockDim.x * gridDim.x;
  for (unsigned int i = idx; i < width*height; i += numThreads) {
    const int pos = height * (int)indices[i / height] + i % height;
    target[i] = (pos < height * width2)? source[i] + mult * mat[pos] : 1.0/0.0 - 1.0/0.0;
  }
}

__global__ void kAccumulateColumns(float* mat, float* indices, float* target, int mat_width, int target_width, int height, float mult, int avg){
  const int row = gridDim.x * blockIdx.y + blockIdx.x;
  const int column = threadIdx.x;
  if (row < height && column < target_width) {
    float cur_sum = 0.0;
    unsigned int count = 0;
    for (unsigned int i = 0; i < mat_width; i ++) {
      count += ((int)indices[i] == column) ? 1 : 0 ;
      cur_sum += ((int)indices[i] == column) ? mat[row + i * height] : 0 ;
    }
    target[row + height * column] = mult * cur_sum / ((avg == 1 && count > 0) ? count : 1);
  }
}
