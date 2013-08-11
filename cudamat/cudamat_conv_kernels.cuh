#include "cudamat.cuh"

#define LO16(x)   ((x) & 0x0000FFFF)
#define HI16(x)   ((x) >> 16)

#define getLastCudaError(msg)   __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
 cudaError_t err = cudaGetLastError();
 if (cudaSuccess != err) {
  fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString(err));
  exit(EXIT_FAILURE);
 }
}

void convContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms,
           cudamat* target, int numFilters, int sizeX,
           float addScale, float powScale);

void convResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs,
             cudamat* acts, cudamat* target, int numFilters,
             int sizeX, float addScale, float powScale,
             float scaleTargets, float scaleOutput);

void convContrastNormUndoCu(cudamat* outGrads, cudamat* denoms,
              cudamat* meanDiffs, cudamat* acts, cudamat* target,
              int numFilters, int sizeX, float addScale,
              float powScale, float scaleTargets,
              float scaleOutput);

class MaxPooler {
public:
  __device__ inline float operator()(const float a, const float b) const {
    return a > b ? a : b;
  }
  __device__ inline float getBaseValue() const {
    return -2e38; 
  }
  __device__ inline float output(const float a) const {
    return a;
  }
};

class ProbMaxPooler {
public:
  __device__ inline float operator()(const float a, const float b, const float r1, const float r2) const {
    return a * r1 > b * r2 ? 0 : 1;
  }
  __device__ inline float getBaseValue() const {
    return -2e38; 
  }
  __device__ inline float output(const float a) const {
    return a;
  }
};

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:   (numColors, imgPixels, numImages) with stride given
 * filters:   (numColors, filterPixels, numFilters) if conv
 *       (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:   (numFilters, numModules, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors,
     bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
                  const int numImages, const int numFilters,
                  const int imgSize, const int filterSize, const int paddingStart,
                  const int moduleStride,
                  const int numModulesX, const int imgStride,
                  const float scaleTargets, const float scaleOutputs,
                  const bool conv) {
  __shared__ float shFilters[B_Y*numColors][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
  __shared__ float shImages[B_Y*numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
  const int imgPixels = imgSize * imgSize;
  const int filterPixels = filterSize * filterSize;

  const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
  const int moduleIdx = blockIdx.y / blocksPerModule;
  const int blockFilterIdx = blockIdx.y % blocksPerModule;

  const int tidx = threadIdx.y * B_X + threadIdx.x;

  const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
  const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

  const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
  const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
  const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
  images += myImgIdx;
  filters += filtersPerThread * B_Y * blockFilterIdx
       + shFilterLoadY * numFilters + shFilterLoadX;
  if (!conv) {
    filters += moduleIdx * numColors * filterPixels * numFilters;
  }

  targets += moduleIdx * numImages
      + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesX * numModulesX
      + myImgIdx;


  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for(int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for(int g = 0; g < imgsPerThread; g++) {
      prod[f][g] = 0;
    }
  }

  for (int p = 0; p < filterPixels; p += B_Y) {
    /*
     * Load B_Y pixels from B_Y*filtersPerThread filters
     */
    if (shFilterLoadY < B_Y) {
      #pragma unroll
      for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
        if (p + p2 + shFilterLoadY < filterPixels) {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
          }
        } else {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
          }
        }
      }
    }

    /*
     * Load B_Y pixels from B_X*imgsPerThread images
     */
    const int pixIdx = p + threadIdx.y;
    if (pixIdx < filterPixels) {
      const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
      const int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
      if (y >= 0 && y< imgSize && x >= 0 && x < imgSize) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
              shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSize + x) + i * B_X];
            }
          } else {
            #pragma unroll
            for (int c = 0; c < numColors; c++) {
              shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
            }
          }
        }
      } else { // Padding
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < B_Y*numColors; i++) {
      #pragma unroll
      for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
          prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
        }
      }

    }
    __syncthreads();
  }
  
  if (scale) {
    #pragma unroll
    for (int g = 0; g < imgsPerThread; g++) {
      if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] + scaleOutputs * prod[f][g];
        }
      }
    }
  } else {
    #pragma unroll
    for (int g = 0; g < imgsPerThread; g++) {
      if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          targets[g * B_X + f * B_Y * numImages * numModulesX * numModulesX] = scaleOutputs * prod[f][g];
        }
      }
    }
  }
}

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:   (numImgColors, imgPixels, numImages) with stride given
 * filters:   (numFilterColors, filterPixels, numFilters) if conv
 *       (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:   (numFilters, numModules, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
     bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
                    const int numImages, const int numFilters,
                    const int imgSize, const int filterSize, const int paddingStart,
                    const int moduleStride,
                    const int numModulesX, const int imgStride, const int numImgColors,
                    const int numGroups, 
                    const float scaleTargets, const float scaleOutputs,
                    const bool conv) {
  __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
  __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
  const int imgPixels = imgSize * imgSize;
  const int filterPixels = filterSize * filterSize;
  const int numFilterColors = numImgColors / numGroups;
  const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
  const int moduleIdx = blockIdx.y / blocksPerModule;
  const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
  const int numFiltersPerGroup = numFilters / numGroups;
  const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

  const int numModules = numModulesX * numModulesX;
  const int blockColorIdx = numFilterColors * blockGroupIdx;

  const int tidx = threadIdx.y * B_X + threadIdx.x;

  const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
  const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

  const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
  const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
  const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

  images += blockColorIdx * imgPixels * imgStride + myImgIdx;
  filters +=blockFilterIdx
      + shFilterLoadY * numFilters + shFilterLoadX;
  if (!conv) {
    filters += moduleIdx * numFilterColors * filterPixels * numFilters;
  }

  targets += moduleIdx * numImages
      + (blockFilterIdx + threadIdx.y) * numImages * numModules
      + myImgIdx;

  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for(int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for(int g = 0; g < imgsPerThread; g++) {
      prod[f][g] = 0;
    }
  }
//  __shared__ int imgPos[]
  for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
    for (int p = 0; p < filterPixels; p += B_Y) {
      /*
       * Load B_Y pixels from B_Y*filtersPerThread filters
       */
      if (shFilterLoadY < B_Y) {
        #pragma unroll
        for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
          if (p + p2 + shFilterLoadY < filterPixels) {
            #pragma unroll
            for (int c = 0; c < colorCache; c++) {
              shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
            }
          } else {
            #pragma unroll
            for (int c = 0; c < colorCache; c++) {
              shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
            }
          }
        }
      }

      /*
       * Load B_Y pixels from B_X*imgsPerThread images
       */
      const int pixIdx = p + threadIdx.y;
      if (pixIdx < filterPixels) {
        const int x = imgLoadModPosX + pixIdx % filterSize;
        const int y = imgLoadModPosY + pixIdx / filterSize;
        if (y >= 0 && y < imgSize && x >= 0 && x < imgSize) {
          float* m = &images[imgStride * (oc * imgPixels + y * imgSize + x)];
          #pragma unroll
          for (int i = 0; i < imgsPerThread; i++) {
            if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
              #pragma unroll
              for (int c = 0; c < colorCache; c++) {
                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
              }
            } else {
              #pragma unroll
              for (int c = 0; c < colorCache; c++) {
                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
              }
            }
          }
        } else { // Padding
          #pragma unroll
          for (int i = 0; i < imgsPerThread; i++) {
            #pragma unroll
            for (int c = 0; c < colorCache; c++) {
              shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
            }
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int i = 0; i < B_Y*colorCache; i++) {
        #pragma unroll
        for(int f = 0; f < filtersPerThread; f++) {
          #pragma unroll
          for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
          }
        }

      }
      __syncthreads();
    }
  }

  if (scale) {
    #pragma unroll
    for (int g = 0; g < imgsPerThread; g++) {
      if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
        }
      }
    }
  } else {
    #pragma unroll
    for (int g = 0; g < imgsPerThread; g++) {
      if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
        }
      }
    }
  }
}

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread.
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:   (numFilters, numModules, numImages)
 * filters:   (numColors, filterPixels, numFilters)        if conv
 *       (numModules, numColors, filterPixels, numFilters)  otherwise
 * targets:   (numColors, imgPixels, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of filters must be divisible by 16.
 * Number of images must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 */
template <int imgsPerThread, int numColors, bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_color(const float* hidActs, const float* filters, float* targets,
                  const int numModulesX, const int numImages, const int numFilters,
                  const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                  const float scaleTargets, const float scaleOutputs) {
  __shared__ float shFilters[numColors*16][16 + 1];
  __shared__ float shHidActs[16][16*imgsPerThread];

  const int blockCaseIdx = blockIdx.x * 16*imgsPerThread;
  const int numRegionsX = DIVUP(imgSize, 4);
  const int blockRegionIdx = blockIdx.y;
  const int blockRegionIdxX = blockRegionIdx % numRegionsX;
  const int blockRegionIdxY = blockRegionIdx / numRegionsX;
  const int blockRegionLeft = blockRegionIdxX * 4;
  const int blockRegionTop = blockRegionIdxY * 4;
  const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
  const int pxY = blockRegionTop + pxYInRegion;
  const int pxX = blockRegionLeft + pxXInRegion;
  const int pxIdx = pxY * imgSize + pxX;
  const bool isPxInImg = pxY < imgSize && pxX < imgSize;
  const uint numModules = numModulesX * numModulesX;
  const int filterPixels = filterSize * filterSize;
  const int imgPixels = imgSize * imgSize;
  const int tidx = threadIdx.y * 16 + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;

  hidActs += blockCaseIdx + loadY * numImages * numModulesX * numModulesX + loadX;
  filters += threadIdx.x;
  targets += pxIdx * numImages + blockCaseIdx + threadIdx.x;


  float prod[numColors][imgsPerThread];
  #pragma unroll
  for (int c = 0; c < numColors; c++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[c][i] = 0;
    }
  }
  const int startY = blockRegionTop - paddingStart < filterSize ? 0
            : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
  const int endY = MIN(numModulesX, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
  const int startX = blockRegionLeft - paddingStart < filterSize ? 0
            : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
  const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);
  
  float* shilterLoad = &shFilters[threadIdx.y][threadIdx.x];
  float* shHidActLoad = &shHidActs[loadY][loadX];

  for (int my = startY; my < endY; my++) {
    const int moduleTop = paddingStart + my * moduleStride;
    const int pxInModuleY = pxY - moduleTop;

    for (int mx = startX; mx < endX; mx++) {
      const int moduleIdx = my * numModulesX + mx;
      const int moduleLeft = paddingStart + mx * moduleStride;
      const int pxInModuleX = pxX - moduleLeft;

      const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
      const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

      for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time
        // Now the threads split up into half-warps, and each half-warp decides if it's interested.
        const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
        #pragma unroll
        for (int i = 0; i < imgsPerThread * 16; i += 32) {
          if (!checkCaseBounds || blockCaseIdx + i + loadX < numImages) {
            #pragma unroll
            for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
              shHidActLoad[j * 16 * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
            }
          } else {
            #pragma unroll
            for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
              shHidActLoad[j * 16 * imgsPerThread + i] = 0;
            }
          }
        }
        
        if (isPxInImg && isPxInModule) {
          // This half-warp is interested, so it's going to load the weights from this module to its pixel.
          // Not fully coalesced read :(
          // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
          const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
                       : &filters[(moduleIdx * numColors * filterPixels + pxIdxInModule) * numFilters + f];
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            shilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
          }

          
        }

        __syncthreads();
        // Do some actual computation
        if (isPxInImg && isPxInModule) {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            #pragma unroll
            for (int w = 0; w < 16; w++) {
              #pragma unroll
              for (int i = 0; i < imgsPerThread; i++) {
                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
  // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
  if (isPxInImg) {
    if (scale) {
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
          }
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
          }
        }
      }
    }
  }
}

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread, also color in batches of colorsPerThread.
 * In essence, blockIdx.x.x = 1..numImages/(16*imgsPerThread)
 *       blockIdx.x.y = 1..numImgColors/colorsPerThread
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:   (numFilters, numModules, numImages)
 * filters:   (numFilterColors, filterPixels, numFilters)        if conv
 *       (numModules, numFilterColors, filterPixels, numFilters)  otherwise
 * targets:   (numImageColors, imgPixels, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * numImages must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 * numImageColors/numGroups must be divisible by colorsPerThread.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are 4-16 color channels.
 */
template <int imgsPerThread, int colorsPerThread, bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_mediumcolor(const float* hidActs, const float* filters, float* targets,
                    const int numModulesX, const int numImages, const int numFilters,
                    const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                    const int numImgColors, const int numGroups,
                    const float scaleTargets, const float scaleOutputs) {
  __shared__ float shFilters[colorsPerThread*16][16 + 1];
  __shared__ float shHidActs[16][16*imgsPerThread];

  const int numImgBlocks = DIVUP(numImages,16*imgsPerThread);
  const int blockCaseIdx = (blockIdx.x % numImgBlocks) * 16*imgsPerThread;

  const int imgColorIdx = (blockIdx.x / numImgBlocks) * colorsPerThread; // color idx globally
  const int numFilterColors = numImgColors / numGroups;
  const int blockGroupIdx = imgColorIdx / numFilterColors;
  const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
  const int numFiltersPerGroup = numFilters / numGroups;
  const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;
  
  const int numRegionsX = DIVUP(imgSize, 4);
  const int blockRegionIdx = blockIdx.y;
  const int blockRegionIdxX = blockRegionIdx % numRegionsX;
  const int blockRegionIdxY = blockRegionIdx / numRegionsX;
  const int blockRegionLeft = blockRegionIdxX * 4;
  const int blockRegionTop = blockRegionIdxY * 4;
  const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
  const int pxY = blockRegionTop + pxYInRegion;
  const int pxX = blockRegionLeft + pxXInRegion;
  const int pxIdx = pxY * imgSize + pxX;
  const bool isPxInImg = pxY < imgSize && pxX < imgSize;
//  const uint numModules = numModulesX * numModulesX;
  const int filterPixels = filterSize * filterSize;
  const int imgPixels = imgSize * imgSize;
  const int tidx = threadIdx.y * 16 + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;

  hidActs += blockCaseIdx + (blockFilterIdx + loadY) * numImages * numModulesX * numModulesX + loadX;
  filters += blockFilterIdx + filterColorIdx * filterPixels * numFilters + threadIdx.x;
  targets += imgColorIdx * imgPixels * numImages + pxIdx * numImages + blockCaseIdx + threadIdx.x;

  float prod[colorsPerThread][imgsPerThread];
  #pragma unroll
  for (int c = 0; c < colorsPerThread; c++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[c][i] = 0;
    }
  }
  const int startY = blockRegionTop - paddingStart < filterSize ? 0
            : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
  const int endY = MIN(numModulesX, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
  const int startX = blockRegionLeft - paddingStart < filterSize ? 0
            : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
  const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

  float* shFilterLoad = &shFilters[threadIdx.y][threadIdx.x];
  float* shHidActLoad = &shHidActs[loadY][loadX];

  for (int my = startY; my < endY; my++) {
    const int moduleTop = paddingStart + my * moduleStride;
    const int pxInModuleY = pxY - moduleTop;

    for (int mx = startX; mx < endX; mx++) {
      const int moduleIdx = my * numModulesX + mx;
      const int moduleLeft = paddingStart + mx * moduleStride;
      const int pxInModuleX = pxX - moduleLeft;

      const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
      const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

      for (int f = 0; f < numFiltersPerGroup; f += 16) { // multipply with 16 filters at a time
        // Now the threads split up into half-warps, and each half-warp decides if it's interested.
        const float* hLoad = &hidActs[(moduleIdx + f * numModulesX * numModulesX) * numImages];
        #pragma unroll
        for (int i = 0; i < imgsPerThread * 16; i += 32) {
          if (!checkCaseBounds || blockCaseIdx + loadX + i < numImages) {
            #pragma unroll
            for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
              shHidActLoad[j * 16 * imgsPerThread + i] = hLoad[j * numModulesX * numModulesX * numImages + i];
            }
          } else {
            #pragma unroll
            for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
              shHidActLoad[j * 16 * imgsPerThread + i] = 0;
            }
          }
        }

        if (isPxInImg && isPxInModule) {
          // This half-warp is interested, so it's going to load the weights from this module to its pixel.
     
          // Not fully coalesced read :(
          // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
          const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
                       : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInModule * numFilters + f];
          #pragma unroll
          for (int c = 0; c < colorsPerThread; c++) {
            shFilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
          }
        }

        __syncthreads();
        // Do some actual computation
        if (isPxInImg && isPxInModule) {
          #pragma unroll
          for (int c = 0; c < colorsPerThread; c++) {
            #pragma unroll
            for (int w = 0; w < 16; w++) {
              #pragma unroll
              for (int i = 0; i < imgsPerThread; i++) {
                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
  // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
  if (isPxInImg) {
    if (scale) {
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
          #pragma unroll
          for (int c = 0; c < colorsPerThread; c++) {
            targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
          }
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
          #pragma unroll
          for (int c = 0; c < colorsPerThread; c++) {
            targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
          }
        }
      }
    }
  }
}

/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 * In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *       blockIdx.x.y = 1..numImgColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:   (numFilters, numModules, numImages)
 * filters:   (numFilterColors, filterPixels, numFilters)        if conv
 *       (numModules, numFilterColors, filterPixels, numFilters)  otherwise
 * targets:   (numImageColors, imgPixels, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * numFiltersPerGroup must be divisible by 16.
 * 
 * B_X * imgsPerThread must be divisible by 32.
 * numFilterColors must be divisible by B_Y*colorsPerThread.
 * B_X*B_Y must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are >= 16 color channels.
 */
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, bool scale, bool checkCaseBounds, bool conv>
__global__ void conv_img_acts_manycolor(const float* hidActs, const float* filters, float* targets,
                     const int numModulesX, const int numImages, const int numFilters,
                     const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                     const int numImgColors, const int numGroups,
                     const float scaleTargets, const float scaleOutputs) {
  __shared__ float shFilters[colorsPerThread*B_Y][16 + 1]; // TODO: perhaps reconsider this 16
  __shared__ float shHidActs[16][B_X*imgsPerThread];

  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;
  
  const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
  const int numFilterColors = numImgColors / numGroups;
  const int blockGroupIdx = imgColorIdx / numFilterColors;
  const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
  const int numFiltersPerGroup = numFilters / numGroups;
  const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

  const int blockPixelIdx = blockIdx.y;
  const int blockPixelIdxX = blockPixelIdx % imgSize;
  const int blockPixelIdxY = blockPixelIdx / imgSize;

  const int filterPixels = filterSize * filterSize;
  const int imgPixels = imgSize * imgSize;
  const int tidx = threadIdx.y * B_X + threadIdx.x;
  const int hidActLoadY = tidx / 32, hidActLoadX = tidx % 32;
  const int filtersLoadY = tidx / 16, filtersLoadX = tidx % 16;
  const int numModules = numModulesX * numModulesX;

  hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
  filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
  targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;

  float prod[colorsPerThread][imgsPerThread];
  #pragma unroll
  for (int c = 0; c < colorsPerThread; c++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[c][i] = 0;
    }
  }

  const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
            : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
  const int endY = MIN(numModulesX, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
  const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
            : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
  const int endX = MIN(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

  float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
  float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];

  for (int my = startY; my < endY; my++) {
    const int moduleTop = paddingStart + my * moduleStride;
    const int pxInFilterY = blockPixelIdxY - moduleTop;

    for (int mx = startX; mx < endX; mx++) {
      const int moduleIdx = my * numModulesX + mx;
      const int moduleLeft = paddingStart + mx * moduleStride;
      const int pxInFilterX = blockPixelIdxX - moduleLeft;
      
      const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

      for (int f = 0; f < numFiltersPerGroup; f += 16) { // multiply with 16 filters at a time
        const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
        #pragma unroll
        for (int i = 0; i < imgsPerThread * B_X; i += 32) {
          if (!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages) {
            #pragma unroll
            for (int j = 0; j < 16; j += B_X*B_Y/32) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
              shHidActLoad[j * B_X * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
            }
          } else {
            #pragma unroll
            for (int j = 0; j < 16; j += B_X*B_Y/32) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
              shHidActLoad[j * B_X * imgsPerThread + i] = 0;
            }
          }
        }
        const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + f]
                     : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f];
        #pragma unroll
        for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/16) {
          if ((colorsPerThread*B_Y) % (B_X*B_Y/16) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
            shFilterLoad[i * (16 + 1)] = fLoad[i * filterPixels * numFilters];
          }
        }
        
        __syncthreads();
        // Do some actual computation
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
          #pragma unroll
          for (int w = 0; w < 16; w++) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
              prod[c][i] += shFilters[c * B_Y + threadIdx.y][w] * shHidActs[w][threadIdx.x + i * B_X];
            }
          }
        }
        __syncthreads();
      }
    }
  }
  if (scale) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
          targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
        }
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
          targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
        }
      }
    }
  }
}

/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:   (numColors, imgPixels, numImages), with stride given
 * hidActs:   (numFilters, numModules, numImages)
 *
 * targets:   (numModules/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelsPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_c(float* images, float* hidActs, float* targets,
                  const int numImages, const int numFilters,
                  const int numModulesX,
                  const int imgSize, const int filterSize,
                  const int paddingStart, const int moduleStride, const int imgStride,
                  const int partialSum,
                  const float scaleTargets, const float scaleOutputs) {
  __shared__ float shImages[pixelsPerThread * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
  __shared__ float shHidActs[B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

  const int tidx = B_X * threadIdx.y + threadIdx.x;
  const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

  const int filterPixels = filterSize * filterSize;
  const int imgPixels = imgSize * imgSize;

  const int filterBlocksPerModule = numFilters / B_X;
  const int outputModuleIdx = blockIdx.x / filterBlocksPerModule;
  const int moduleIdx = partialSum * outputModuleIdx;
  const int blockFilterIdx = B_X * (blockIdx.x % filterBlocksPerModule);

//  const int moduleStride = (imgSize - filterSize + 1) / numModulesX; 
  const int numModules = numModulesX * numModulesX;

  const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;

  images += loadX;
  hidActs += moduleIdx * numImages
      + blockFilterIdx * numImages * numModules
      + loadY * numImages * numModules
      + loadX;
  
  targets += (outputModuleIdx * numFilters) * filterPixels * numColors
      + blockPixelOffset * numFilters
      + blockFilterIdx
      + threadIdx.y * numFilters + threadIdx.x;

  float* shImgLoad = &shImages[loadY][loadX];
  float* shHidActLoad = &shHidActs[loadY][loadX];

  float prod[numColors][pixelsPerThread];
  #pragma unroll
  for (int c = 0; c < numColors; c++) {
    #pragma unroll
    for (int p = 0; p < pixelsPerThread; p++) {
      prod[c][p] = 0;
    }
  }
  
  __shared__ int pxDivs[B_Y*pixelsPerThread];
  if (tidx < B_Y * pixelsPerThread) {
    pxDivs[tidx] = (((blockPixelOffset + tidx) / filterSize) << 16) + ((blockPixelOffset + tidx) % filterSize);
  }
  __syncthreads();
  for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
    const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
    for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
      if (loadY < B_Y * pixelsPerThread) {
        /*
         * As long as B_Y * B_X is divisible by preloadCases this will loop the right
         * number of times.
         *
         * This will load some imgGrads from filter pixels that don't exit (it'll set those to 0),
         * but the code does not produce any output for those pixels (see last lines).
         */
  //      #pragma unroll
        for (int y = 0; y < B_Y * pixelsPerThread; y += (B_X * B_Y) / preloadCases) {
          // Make sure number of rows in the array is divisible by number of rows filled per iteration
          if ((B_Y * pixelsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelsPerThread) {
            const int pxIdx = loadY + y; // pixel idx in filter

            if (pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
              const int pxY = imgLoadModPosY + HI16(pxDivs[pxIdx]); // pixel x,y coords in image
              const int pxX = imgLoadModPosX + LO16(pxDivs[pxIdx]);
              if (pxY >= 0 && pxY < imgSize && pxX >= 0 && pxX < imgSize) {
                const int pixIdx = (pxY * imgSize + pxX) * imgStride;
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                  shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                }
              } else {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                  shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                }
              }
            } else {
              #pragma unroll
              for (int c = 0; c < numColors; c++) {
                shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
              }
            }
          }
        }
      }
      if (loadY < B_X && (!checkCaseBounds || caseIdx + loadX < numImages)) {
        #pragma unroll
        for (int y = 0; y < B_X; y += (B_X * B_Y) / preloadCases) {
          // Make sure number of rows in the array is divisible by number of rows filled per iteration
          if (B_X % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X) {
            shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules];
          }
        }
      }

      __syncthreads();
      #pragma unroll
      for (int p = 0; p < pixelsPerThread; p++) {
        #pragma unroll
        for (int i = 0; i < preloadCases; i++) {
          #pragma unroll
          for (int c = 0; c < numColors; c++) {
            prod[c][p] += shImages[threadIdx.y + p * B_Y + c * pixelsPerThread * B_Y][i] * shHidActs[threadIdx.x][i];
          }
        }
      }
      __syncthreads();
    }
    hidActs += numImages;
  }
  
  if (scale) {
    #pragma unroll
    for (int p = 0; p < pixelsPerThread; p++) {
      if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
        #pragma unroll
        for (int c = 0; c < numColors; c++) {
          targets[p * B_Y * numFilters + c * filterPixels * numFilters] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters] + scaleOutputs * prod[c][p];
        }
      }
    }
  } else {
    #pragma unroll
    for (int p = 0; p < pixelsPerThread; p++) {
      if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
        #pragma unroll
        for (int c = 0; c < numColors; c++) {
          targets[p * B_Y * numFilters + c * filterPixels * numFilters] = scaleOutputs * prod[c][p];
        }
      }
    }
  }
}


/*
 * Each block computes weight gradients for B_Y pixels and B_X * filtersPerThread filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel, color batch of B_Y * colorsPerThread
 *   In essence, blockIdx.y.x = 0...numFilterColors / colorsPerThread
 *         blockIdx.y.y = 0...DIVUP(numPixels, B_Y)
 * ============
 * CONSTRAINTS:
 * ============
 * numFilters/numGroups must be divisible by B_X * filtersPerThread
 * numImgColors/numGroups must be divisible by colorsPerThread
 * numFilters must be divisible by numGroups
 * numImgColors must be divisible by numGroups
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:   (numImgColors, imgPixels, numImages), with stride given
 * hidActs:   (numFilters, numModules, numImages)
 *
 * targets:   (numModules/partialSum, numFilterColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * 
 * This routine is especially fast when numFilters >= 32. That's when it should be used.
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale, bool checkCaseBounds>
__global__ void conv_weight_acts_mc_mf(float* images, float* hidActs, float* targets,
                    const int numImages, const int numFilters,
                    const int numModulesX,
                    const int imgSize, const int filterSize,
                    const int paddingStart, const int moduleStride, const int imgStride,
                    const int numImgColors, const int numGroups, const int partialSum,
                    const float scaleTargets, const float scaleOutputs) {
  __shared__ float shImages[colorsPerThread * B_Y][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
  __shared__ float shHidActs[filtersPerThread * B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts

  const int tidx = B_X * threadIdx.y + threadIdx.x;
  const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

  const int filterPixels = filterSize * filterSize;
  const int imgPixels = imgSize * imgSize;

  const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
  const int outputModuleIdx = blockIdx.x / numFilterBlocks;
  const int moduleIdx = partialSum * outputModuleIdx;
  const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
  const int numModules = numModulesX * numModulesX;
  
  const int numFiltersPerGroup = numFilters / numGroups;
  const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
  const int numFilterColors = numImgColors / numGroups;
  
  const int blockPixelOffset = (blockIdx.y / (numFilterColors/colorsPerThread)) * B_Y;
  const int filterColorIdx = (blockIdx.y % (numFilterColors/colorsPerThread)) * colorsPerThread;
  const int imgColorIdx = filterColorIdx + blockGroupIdx * numFilterColors;

  images += imgColorIdx * imgPixels * imgStride + loadX;

  hidActs += moduleIdx * numImages
      + blockFilterIdx * numImages * numModules
      + loadY * numImages * numModules
      + loadX;
  
  targets += outputModuleIdx * numFilters * filterPixels * numFilterColors
      + filterColorIdx * filterPixels * numFilters
      + blockPixelOffset * numFilters
      + blockFilterIdx
      + threadIdx.y * numFilters + threadIdx.x;

  float* shHidActLoad = &shHidActs[loadY][loadX];
  float* shImgLoad = &shImages[loadY][loadX];
  float prod[colorsPerThread][filtersPerThread];
  #pragma unroll
  for (int c = 0; c < colorsPerThread; c++) {
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
      prod[c][f] = 0;
    }
  }
  // This avoids doing a division in an inner loop
  __shared__ int pxDivs[B_Y];
  if (tidx < B_Y) {
    pxDivs[tidx] = (((blockPixelOffset + tidx) / filterSize) << 16) + (blockPixelOffset + tidx) % filterSize;
  }
  __syncthreads();
  for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
    const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
    for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
      if (loadY < B_Y) {
        /*
         * As long as B_Y * B_X is divisible by preloadCases this will loop the right
         * number of times.
         *
         * This will load some images from filter pixels that don't exist (it'll set those to 0),
         * but the code does not produce any output for those pixels (see last lines).
         */
  //      #pragma unroll
        for (int y = 0; y < B_Y; y += (B_X * B_Y) / preloadCases) {
          // Make sure number of rows in the array is divisible by number of rows filled per iteration
          if (B_Y % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y) {
            const int pxIdx = loadY + y; // pixel idx in filter

            if (pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
              const int pxY = imgLoadModPosY + HI16(pxDivs[pxIdx]);//pxIdx / filterSize; // pixel x,y coords in image
              const int pxX = imgLoadModPosX + LO16(pxDivs[pxIdx]);
              if (pxY >= 0 && pxY < imgSize && pxX >= 0 && pxX < imgSize) {
                const int pixIdx = (pxY * imgSize + pxX) * imgStride; // pixel idx in image
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                  shImgLoad[(y + c * B_Y) * preloadCases] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                }
              } else {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                  shImgLoad[(y + c * B_Y) * preloadCases] = 0;
                }
              }
            } else {
              #pragma unroll
              for (int c = 0; c < colorsPerThread; c++) {
                shImgLoad[(y + c * B_Y) * preloadCases] = 0;
              }
            }
          }
        }
      }
      if (loadY < B_X * filtersPerThread && (!checkCaseBounds || caseIdx + loadX < numImages)) {
        #pragma unroll
        for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
          // Make sure number of rows in the array is divisible by number of rows filled per iteration
          if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
            shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules];
          }
        }
      }

      __syncthreads();

      #pragma unroll
      for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < preloadCases; i++) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            prod[c][f] += shImages[threadIdx.y + c * B_Y][i] * shHidActs[threadIdx.x + f * B_X][i];
          }
        }
      }
      __syncthreads();
    }
    hidActs += numImages;
  }
  if (blockPixelOffset + threadIdx.y < filterPixels) {
    if (scale) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
          targets[c * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][f];
        }
      }
    } else {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int c = 0; c < colorsPerThread; c++) {
          targets[c * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][f];
        }
      }
    }
  }
}

template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool2(float* imgs, float* target, const int imgSize, const int numFilters,
              const int numImages, const int subsX, const int startX,
              const int outputsX, Agg agg) {
  __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(filtersPerThread);
  const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
  const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
  
//  const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
  const int numOutputs = outputsX * outputsX;
  const int imgPixels = imgSize * imgSize;
  
  const int tidx = threadIdx.y * B_X + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;
  
  const int myX = threadIdx.y % 4;
  const int myY = threadIdx.y / 4;
  
  const int myOutputIdxY = blockOutputY + myY;
  const int myOutputIdxX = blockOutputX + myX;
  const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;
  
  const int startImgPxX = startX + blockOutputX;
  const int startImgPxY = startX + blockOutputY;
  const int endImgPxX = startImgPxX + subsX;
  const int endImgPxY = startImgPxY + subsX;
  
  const int myStartImgPxY = startImgPxY + myY;
  const int myStartImgPxX = startImgPxX + myX;
  const int myEndImgPxY = endImgPxY + myY;
  const int myEndImgPxX = endImgPxX + myX;

  const int loopStartY = MAX(startImgPxY, 0);
  const int loopStartX = MAX(startImgPxX, 0);
  const int loopEndY = MIN(imgSize, endImgPxY + 3);
  const int loopEndX = MIN(imgSize, endImgPxX + 3);

  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
  target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = agg.getBaseValue(); 
    }
  }

  for (int y = loopStartY; y < loopEndY; y++) {
    const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
    for (int x = loopStartX; x < loopEndX; x++) {
      // Load a pixel
      const int px = y * imgSize + x;
      #pragma unroll
      for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
        if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
          #pragma unroll
          for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
            if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
              shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
            }
          }
        }
      }
      __syncthreads();

      // Is this pixel in my region?
      if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
              prod[f][i] = agg(prod[f][i], shImgs[f][threadIdx.x + i * B_X]);
            }
          }
        }
      }
      __syncthreads();

    }
  }
  if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i]); 
        }
      }
    }
  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * target:   (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool(float* imgs, float* target, const int imgSize, const int numFilters,
              const int numImages, const int subsX, const int startX, const int strideX,
              const int outputsX, Agg agg) {
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
  const int outputIdxX = blockIdx.x / numImgBlocks;
  const int outputIdxY = blockIdx.y / numFilterBlocks;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
  const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
  if (myFilterIdx >= numFilters) {
    return;
  }
  
  const int outputIdx = outputIdxY * outputsX + outputIdxX;
  const int numOutputs = outputsX * outputsX;
  const int imgPixels = imgSize * imgSize;
  
  const int startImgPxX = startX + outputIdxX * strideX;
  const int startImgPxY = startX + outputIdxY * strideX;
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += myFilterIdx * imgPixels * numImages + imgIdx;
  target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = agg.getBaseValue(); 
    }
  }
  
  const int loopStartY = MAX(0, startImgPxY);
  const int loopStartX = MAX(0, startImgPxX);
  const int loopEndY = MIN(imgSize, startImgPxY + subsX);
  const int loopEndX = MIN(imgSize, startImgPxX + subsX);
  for (int y = loopStartY; y < loopEndY; y++) {
    for (int x = loopStartX; x < loopEndX; x++) {
      const int imgPx = y * imgSize + x;
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
          }
        }
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i]); 
      }
    }
  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * maxGrads:  (numFilters, numOutputs, numImages)
 * rMaxActs:  (numFilters, numOutputs, numImages)
 * target:   (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalMaxUndo(float* imgs, float* maxGrads, float* maxActs, float* target, const int imgSize, const int numFilters,
               const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
               const float scaleTargets, const float scaleOutputs) {
  __shared__ float shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int blockPxX = blockIdx.x / numImgBlocks;
  const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
  
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
  
  const int blockPx = blockPxY * imgSize + blockPxX;
  const int numOutputs = outputsX * outputsX;
  const int imgPixels = imgSize * imgSize;

  const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
  const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
  const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
  const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
  
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
  maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
      + imgIdx;
  maxActs += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
      + imgIdx;
  
  target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = 0;
    }
  }
  
  if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
     && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
        }
      }
    }
    for (int my = startOutputY; my < endOutputY; my++) {
      for (int mx = startOutputX; mx < endOutputX; mx++) {
        const int outputIdx = my * outputsX + mx;
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
              const float ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X]; 
              const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
              const float img = shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i];

              prod[f][i] += (img == ma) * mg;
            }
          }
        }
      }
    }
  }
  if (!add) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
        }
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
        }
      }
    }
  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * rnd :    (numFilters, imgPixels, numImages)
 * target:   (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalProbPool(float* imgs, float* rnd, float* target, const int imgSize, const int numFilters,
              const int numImages, const int subsX, const int startX, const int strideX,
              const int outputsX, Agg agg) {
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
  const int outputIdxX = blockIdx.x / numImgBlocks;
  const int outputIdxY = blockIdx.y / numFilterBlocks;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
  const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
  if (myFilterIdx >= numFilters) {
    return;
  }
  
  const int outputIdx = outputIdxY * outputsX + outputIdxX;
  const int numOutputs = outputsX * outputsX;
  const int imgPixels = imgSize * imgSize;
  
  const int startImgPxX = startX + outputIdxX * strideX;
  const int startImgPxY = startX + outputIdxY * strideX;
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += myFilterIdx * imgPixels * numImages + imgIdx;
  rnd += myFilterIdx * imgPixels * numImages + imgIdx;
  target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  float rnd_used[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = agg.getBaseValue(); 
      rnd_used[f][i] = agg.getBaseValue(); 
    }
  }
  
  const int loopStartY = MAX(0, startImgPxY);
  const int loopStartX = MAX(0, startImgPxX);
  const int loopEndY = MIN(imgSize, startImgPxY + subsX);
  const int loopEndX = MIN(imgSize, startImgPxX + subsX);
  for (int y = loopStartY; y < loopEndY; y++) {
    for (int x = loopStartX; x < loopEndX; x++) {
      const int imgPx = y * imgSize + x;
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            const int loc = (f * imgPixels + imgPx) * numImages + i * B_X;
            const int res = agg(prod[f][i], imgs[loc], rnd_used[f][i], rnd[loc]);
            prod[f][i] = res == 0 ? prod[f][i] : imgs[loc];
            rnd_used[f][i] = res == 0 ? rnd_used[f][i] : rnd[loc];
          }
        }
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i]); 
      }
    }
  }
}

/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does a 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * target:   (numFilters, numOutputs, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 * 
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalProbPool2(float* imgs, float* rnd, float* target, const int imgSize, const int numFilters,
              const int numImages, const int subsX, const int startX,
              const int outputsX, Agg agg) {
  __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
  __shared__ float shRnd[filtersPerThread][B_X*imgsPerThread];
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(filtersPerThread);
  const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
  const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
  
//  const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
  const int numOutputs = outputsX * outputsX;
  const int imgPixels = imgSize * imgSize;
  
  const int tidx = threadIdx.y * B_X + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;
  
  const int myX = threadIdx.y % 4;
  const int myY = threadIdx.y / 4;
  
  const int myOutputIdxY = blockOutputY + myY;
  const int myOutputIdxX = blockOutputX + myX;
  const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;
  
  const int startImgPxX = startX + blockOutputX;
  const int startImgPxY = startX + blockOutputY;
  const int endImgPxX = startImgPxX + subsX;
  const int endImgPxY = startImgPxY + subsX;
  
  const int myStartImgPxY = startImgPxY + myY;
  const int myStartImgPxX = startImgPxX + myX;
  const int myEndImgPxY = endImgPxY + myY;
  const int myEndImgPxX = endImgPxX + myX;

  const int loopStartY = MAX(startImgPxY, 0);
  const int loopStartX = MAX(startImgPxX, 0);
  const int loopEndY = MIN(imgSize, endImgPxY + 3);
  const int loopEndX = MIN(imgSize, endImgPxX + 3);

  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
  rnd += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
  target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  float rnd_used[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = agg.getBaseValue(); 
      rnd_used[f][i] = agg.getBaseValue(); 
    }
  }

  for (int y = loopStartY; y < loopEndY; y++) {
    const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
    for (int x = loopStartX; x < loopEndX; x++) {
      // Load a pixel
      const int px = y * imgSize + x;
      #pragma unroll
      for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
        if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
          #pragma unroll
          for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
            if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
              shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
              shRnd[ly + loadY][lx + loadX] = rnd[(ly * imgPixels + px) * numImages + lx];
            }
          }
        }
      }
      __syncthreads();

      // Is this pixel in my region?
      if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
              const int loc = threadIdx.x + i * B_X;
              const int res = agg(prod[f][i], shImgs[f][loc], rnd_used[f][i], shRnd[f][loc]);
              prod[f][i] = res == 0 ? prod[f][i] : shImgs[f][loc];
              rnd_used[f][i] = res == 0 ? rnd_used[f][i] : shRnd[f][loc];
            }
          }
        }
      }
      __syncthreads();

    }
  }
  if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i]); 
        }
      }
    }
  }
}

/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:     (numFilters, imgPixels, numImages)
 * inputs:     (numFilters, imgPixels, numImages)
 * acts:      (numFilters, imgPixels, numImages)
 * target:     (numFilters, imgPixels, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kRNormUndo2(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
              const int numImages, const int sizeX, const float powScale, const float scaleTargets, const float scaleOutputs) {
  __shared__ float shActs[filtersPerThread][B_X*imgsPerThread];
  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(filtersPerThread);
  const int blockPxX = 4*(blockIdx.x / numImgBlocks);
  const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
  
  const int tidx = threadIdx.y * B_X + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;
  
  const int startPxX = MAX(0, -DIVUP(sizeX,2) + blockPxX + 1);
  const int startPxY = MAX(0, -DIVUP(sizeX,2) + blockPxY + 1);
  const int endPxX = MIN(imgSize, blockPxX + sizeX/2 + 4);
  const int endPxY = MIN(imgSize, blockPxY + sizeX/2 + 4);
  
  const int myPxX = blockPxX + threadIdx.y % 4;
  const int myPxY = blockPxY + threadIdx.y / 4;
  const int myPxIdx = myPxY * imgSize + myPxX;
//  const bool doWork = myPxX < imgSize && myPxY < imgSize;
  const int myStartPxY = -DIVUP(sizeX,2) + myPxY + 1;
  const int myStartPxX = -DIVUP(sizeX,2) + myPxX + 1;
  const int myEndPxY = myPxY + sizeX/2 + 1;
  const int myEndPxX = myPxX + sizeX/2 + 1;
  
  const int imgIdx = blockImgIdx + threadIdx.x;
    
  acts    += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
  denoms   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  inputs   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  outGrads  += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  target   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = 0; 
    }
  }

  for (int y = startPxY; y < endPxY; y++) {
    const bool isInY = y >= myStartPxY && y < myEndPxY;
    for (int x = startPxX; x < endPxX; x++) {
      const int px = y * imgSize + x;
      // All the threads load a pixel from memory
      #pragma unroll
      for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
        if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
          #pragma unroll
          for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
            if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
              shActs[ly + loadY][lx + loadX] = acts[(ly * imgPixels + px) * numImages + lx];
            }
          }
        }
      }
      __syncthreads();
      
      // Each row of threads decides if it's interested in this pixel
      if (isInY && x >= myStartPxX && x < myEndPxX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
              prod[f][i] += shActs[f][threadIdx.x + i * B_X];
            }
          }
        }
      }
      __syncthreads();
    }
  }
  acts -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
  acts += threadIdx.x;
  if (myPxX < imgSize && myPxY < imgSize) {
    if (!add) {
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            const float out = outGrads[f * imgPixels * numImages + i * B_X];
            const float den = denoms[f * imgPixels * numImages + i * B_X];
            const float inp = inputs[f * imgPixels * numImages + i * B_X];
            prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
            target[f * imgPixels * numImages + i * B_X] = prod[f][i];
          }
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            const float out = outGrads[f * imgPixels * numImages + i * B_X];
            const float den = denoms[f * imgPixels * numImages + i * B_X];
            const float inp = inputs[f * imgPixels * numImages + i * B_X];
            prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
            target[f * imgPixels * numImages + i * B_X] = scaleTargets * target[f * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
          }
        }
      }
    }

  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:     (numFilters, imgPixels, numImages)
 * inputs:     (numFilters, imgPixels, numImages)
 * acts:      (numFilters, imgPixels, numImages)
 * target:     (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 * 
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kRNormUndo(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
               const int numImages, const int sizeX, const float powScale, const float scaleTargets, const float scaleOutputs) {
  const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);
  
  const int blockPxX = blockIdx.x / numImgBlocks;
  const int blockPxY = blockIdx.y / numFilterBlocks;
  
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
  
  const int blockPx = blockPxY * imgSize + blockPxX;
  const int imgPixels = imgSize * imgSize;

  const int startY = MAX(0, blockPxY + sizeX/2 - sizeX + 1);
  const int startX = MAX(0, blockPxX + sizeX/2 - sizeX + 1);
  const int endY = MIN(imgSize, blockPxY + sizeX/2 + 1);
  const int endX = MIN(imgSize, blockPxX + sizeX/2 + 1);

  const int imgIdx = blockImgIdx + threadIdx.x;
  
  acts    += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
  inputs   += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
  denoms   += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
  outGrads  += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
  target   += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int f = 0; f < filtersPerThread; f++) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      prod[f][i] = 0;
    }
  }
  
  for (int sy = startY; sy < endY; sy++) {
    for (int sx = startX; sx < endX; sx++) {
      const int outPx = sy * imgSize + sx;

      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            prod[f][i] += acts[(f * B_Y * imgPixels + outPx) * numImages + i * B_X];
          }
        }
      }
    }
  }
//  outGrads += blockPx * numImages;
  if (!add) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          const float inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
          const float out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
          const float den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
          prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
          target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
        }
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          const float inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
          const float out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
          const float den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
          prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
          target[f * B_Y * imgPixels * numImages + i * B_X] = 
                        scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] 
                        + scaleOutputs * prod[f][i];
        }
      }
    }
  }
}


/*
 * acts := -2 x scale x acts x outGrads / denoms
 */
template<int B_X, int eltsPerThread>
__global__ void kRNormUndoPrelims(float* acts, float* denoms, float* outGrads,
                 const uint numElements, const float scale) {
  const uint e = B_X * blockIdx.x * eltsPerThread + threadIdx.x;
  const uint numThreads = B_X * gridDim.x;
  for (uint i = e; i < numElements; i += numThreads*eltsPerThread) {
    #pragma unroll
    for (uint k = 0; k < eltsPerThread; k++) {
      if (i + k * B_X < numElements) {
        acts[i + k * B_X] = __fdividef(scale*outGrads[i + k * B_X] * acts[i + k * B_X], denoms[i + k * B_X]);
      }
    }
  }
}

