#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
#define NUM_RND_BURNIN                      100

#define COPY_BLOCK_SIZE                     16
#
#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512

#define PI 3.1415926535897932f

__device__ float device_val;
__global__ void kSeedRandom(unsigned int* randMults, unsigned long long* randWords, unsigned int seed);
__global__ void kRandomUniform(unsigned int* randMults, unsigned long long* randWords, float* gData, unsigned int numElements);
__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements);
__global__ void kRandomDropout(unsigned int* randMults, unsigned long long* randWords, float* gData, unsigned int numElements, float dropprob, float val);
__global__ void kSampleBernoulli(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kSampleBernoulliTanh(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kSamplePoisson(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kSampleGaussian(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements, float mult);
__global__ void kPerturb(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);

__global__ void kGetRowSlice(float* source, float* target, int start, int end, int width, int height);
__global__ void kTranspose(float *odata, float *idata, int width, int height);
__global__ void kSetRowSlice(float* source, float* target, int start, int end, int width, int height);

__global__ void kLessThan(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kLessThanScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kGreaterThan(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kUpperBound(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kLowerBound(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kGreaterThanScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kUpperBoundScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kLowerBoundScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kArgMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kSqSumColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kNormLimitColumnwise(float* mat, float* target, float norm, unsigned int width, unsigned int height);
__global__ void kSumAll(float* mat, unsigned int len);
__global__ void kSign(float* mat, float* target, unsigned int len);
__global__ void kApplySigmoid(float* mat, float* target, unsigned int len);
__global__ void kApplyTanh(float* mat, float* target, unsigned int len);
__global__ void kApplyAbs(float* mat, float* target, unsigned int len);
__global__ void kApplyLog1PlusExp(float* mat, float* target, unsigned int len);
__global__ void kApplyLog1PlusExpExact(float* mat, float* target, unsigned int len);
__global__ void kLog(float* mat, float* target, unsigned int len, float tiny);
__global__ void kExp(float* mat, float* target, unsigned int len);
__global__ void kCeil(float* mat, float* target, unsigned int len);
__global__ void kFloor(float* mat, float* target, unsigned int len);
__global__ void kSqrt(float* mat, float* target, unsigned int len);
__global__ void kPow(float* mat, float pow, float* target, unsigned int len);
__global__ void kPowMatrix(float* mat, float* pow, float* target, unsigned int len);
__global__ void kCrossEntropy(float* mat, float* p, float* target, unsigned int len, float tiny);
__global__ void kCrossEntropyBernoulli(float* mat, float* p, float* target, unsigned int len, float tiny);
__global__ void kCorrectPreds(float* mat, float* p, float* target, unsigned int len, float cutoff);
__global__ void kReciprocal(float* mat, float* target, unsigned int len);
__global__ void kAddDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width);
__global__ void kAddDiagonalScalar(float* mat, float val, float* tgtMat, unsigned int width);
__global__ void kMultDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width);
__global__ void kMultDiagonalScalar(float* mat, float val, float* tgtMat, unsigned int width);
__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddColMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height);
__global__ void kAddRowMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height);
__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kAdd(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kSubtract(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kMult(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kLogisticDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kTanhDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kRectifiedLinearDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kRectifiedLinearSmoothDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kDivide(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kMultScalar(float* mat, float alpha, float* dest, unsigned int len);
__global__ void kAssignScalar(float* dest, float alpha, unsigned int len);
__global__ void kDivideScalar(float* mat, float alpha, float* dest, unsigned int len);
__global__ void kAddScalar(float* a, float alpha, float* dest, unsigned int numEls);
__global__ void kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows);
__global__ void kSetSelectedRows(float* target, float* source, float* indices, int nRowIs, int nCols, int nTargetRows);
__global__ void kSwapRows(float* target, float* source, float* indices1, float* indices2, int nRowIs, int nCols, int nRows);
__global__ void kGenerateTranslationsBigVarOff(float* source, float* target, float* off_x_arr, float* off_y_arr, int source_w, int target_w, int num_channels);
__global__ void kBlockify(float* source, float* target, int numdims, int blocksize);
__global__ void kCumsum(float *mat, float *target, float *temp, unsigned int height);
__global__ void kChooseMaxColumnwise(float* mat, float* target, unsigned int height);
__global__ void kChooseMaxAndAccumulate(float* mat, float* acc, unsigned int height);
#endif
