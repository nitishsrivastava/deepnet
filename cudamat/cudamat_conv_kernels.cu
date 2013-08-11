#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <math.h>
#include <assert.h>
#include "cudamat_conv_kernels.cuh"

__device__ float square(const float a) {
  return a*a;
}

/*
 * Block size 1x128
 * blockIdx.x determines pixel.x, image idx in batches of 128*imgsPerThread
 * blockIdx.y determines pixel.y
 * 
 * So each block does one output for some number of images and all the fliters.
 * 
 * threadIdx.x determines img idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * meanDiffs:  (numFilters, imgPixels, numImages)
 * denoms:   (numFilters, imgPixels, numImages) (out)
 * target:   (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int imgsPerThread, int numFilters, bool checkCaseBounds>
__global__ void kCNorm_fewfilter(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                 const int numImages, const int sizeX, const float addScale, const float powScale) {

  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, 128*imgsPerThread);
  const int pxIdxX = blockIdx.x / numImgBlocks;
  const int pxIdxY = blockIdx.y;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * 128 * imgsPerThread;
  
  const int pxIdx = pxIdxY * imgSize + pxIdxX;
  
  const int startPxX = -sizeX/2 + pxIdxX;
  const int startPxY = -sizeX/2 + pxIdxY;
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += pxIdx * numImages + imgIdx;
  denoms += pxIdx * numImages + imgIdx;
  meanDiffs += imgIdx;
  target += pxIdx * numImages + imgIdx;
  
  float prod[numFilters][imgsPerThread];
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
      #pragma unroll
      for (int f = 0; f < numFilters; f++) {
        prod[f][i] = 0; 
      }
    }
  }
  const int loopStartY = MAX(0, startPxY);
  const int loopStartX = MAX(0, startPxX);
  const int loopEndY = MIN(imgSize, startPxY + sizeX);
  const int loopEndX = MIN(imgSize, startPxX + sizeX);
    
  for (int y = loopStartY; y < loopEndY; y++) {
    for (int x = loopStartX; x < loopEndX; x++) {
      const int imgPx = y * imgSize + x;
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
          #pragma unroll
          for (int f = 0; f < numFilters; f++) {
            prod[f][i] += square(meanDiffs[(f * imgPixels + imgPx) * numImages + i * 128]);
          }
        }
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
      #pragma unroll
      for (int f = 0; f < numFilters; f++) {
        prod[f][i] = 1 + addScale * prod[f][i];
        denoms[f * imgPixels * numImages + i * 128] = prod[f][i];
        target[f * imgPixels * numImages + i * 128] = imgs[f * imgPixels * numImages + i * 128] * __powf(prod[f][i], -powScale);
      }
    }
  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * means:    (numFilters, imgPixels, numImages)
 * denoms:   (numFilters, imgPixels, numImages) (out)
 * target:   (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm_manyfilter(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                 const int numFilters, const int numImages, const int sizeX, const float addScale, const float powScale) {
  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);
  const int pxIdxX = blockIdx.x / numImgBlocks;
  const int pxIdxY = blockIdx.y / numFilterBlocks;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
  
  const int pxIdx = pxIdxY * imgSize + pxIdxX;
  
  const int startPxX = -sizeX/2 + pxIdxX;
  const int startPxY = -sizeX/2 + pxIdxY;
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
  meanDiffs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + imgIdx;
  denoms += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
  target += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        prod[f][i] = 0;
      }
    }
  }

  const int loopStartY = MAX(0, startPxY);
  const int loopStartX = MAX(0, startPxX);
  const int loopEndY = MIN(imgSize, startPxY + sizeX);
  const int loopEndX = MIN(imgSize, startPxX + sizeX);
  
  for (int y = loopStartY; y < loopEndY; y++) {
    for (int x = loopStartX; x < loopEndX; x++) {
      const int imgPx = y * imgSize + x;
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            prod[f][i] += square(meanDiffs[(f * B_Y * imgPixels + imgPx) * numImages + i * B_X]);
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
        prod[f][i] = 1 + addScale * prod[f][i];
        denoms[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
        target[f * B_Y * imgPixels * numImages + i * B_X] = imgs[f * B_Y * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
      }
    }
  }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does 4x4 region of pixels for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * means:    (numFilters, imgPixels, numImages)
 * denoms:   (numFilters, imgPixels, numImages) (out)
 * target:   (numFilters, imgPixels, numImages) (out)
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
template<int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm2(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
             const int numFilters, const int numImages, const int sizeX, const float addScale, const float powScale) {
  __shared__ float shDiffs[filtersPerThread][B_X*imgsPerThread];
  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(filtersPerThread);
  const int blockPxX = 4*(blockIdx.x / numImgBlocks);
  const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
  
  const int tidx = threadIdx.y * B_X + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;
  
  const int startPxX = MAX(0, -sizeX/2 + blockPxX);
  const int startPxY = MAX(0, -sizeX/2 + blockPxY);
  const int endPxX = MIN(imgSize, blockPxX + DIVUP(sizeX, 2) + 3);
  const int endPxY = MIN(imgSize, blockPxY + DIVUP(sizeX, 2) + 3);
  
  const int myPxX = blockPxX + threadIdx.y % 4;
  const int myPxY = blockPxY + threadIdx.y / 4;
  const int myPxIdx = myPxY * imgSize + myPxX;
//  const bool doWork = myPxX < imgSize && myPxY < imgSize;
  const int myStartPxY = -sizeX/2 + myPxY;
  const int myStartPxX = -sizeX/2 + myPxX;
  const int myEndPxY = myPxY + DIVUP(sizeX, 2);
  const int myEndPxX = myPxX + DIVUP(sizeX, 2);
  
  const int imgIdx = blockImgIdx + threadIdx.x;
    
  imgs    += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  meanDiffs  += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
  denoms   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  target   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        prod[f][i] = 0;
      }
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
              shDiffs[ly + loadY][lx + loadX] = meanDiffs[(ly * imgPixels + px) * numImages + lx];
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
              prod[f][i] += square(shDiffs[f][threadIdx.x + i * B_X]);
            }
          }
        }
      }
      __syncthreads();
    }
  }
//  imgs -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
//  imgs += threadIdx.x;
  if (myPxX < imgSize && myPxY < imgSize) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          prod[f][i] = 1 + addScale * prod[f][i];
          denoms[f * imgPixels * numImages + i * B_X] = prod[f][i];
          target[f * imgPixels * numImages + i * B_X] = imgs[f * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
        }
      }
    }
  }
}

void convContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* target, int numFilters, int sizeX, float addScale, float powScale) {
  int numImages = images->size[0];
  int imgPixels = images->size[1] / numFilters;
  assert(images->size[1] == numFilters * imgPixels);
  int imgSize = int(sqrt(imgPixels));
  assert(imgSize * imgSize == imgPixels);
  //assert(meanDiffs.isSameDims(images));
  
  //assert(!meanDiffs.isTrans());
  //assert(!images.isTrans());
  //assert(images.isContiguous());
  //assert(meanDiffs.isContiguous());
  assert(numFilters % 16 == 0 || numFilters <= 8);

  //target.resize(images);
  //denoms.resize(images);

  if (sizeX >= 6 && numFilters % 4 == 0) {
    // This one is faster for large regions (my tests show regions >= 6...)
    int imgsPerThread = 8;
    int filtersPerThread = 4;
    int bx = 8;
    bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
    assert((imgsPerThread * bx) % 32 == 0);
    assert(numFilters % filtersPerThread == 0);
    dim3 threads(bx, 16);
    dim3 blocks(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);

    if (checkCaseBounds) {
      cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, true>, cudaFuncCachePreferL1); // L1 faster here
      kCNorm2<8, 8, 4, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                            imgSize, numFilters, numImages, sizeX, addScale, powScale);
    } else {
      cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, false>, cudaFuncCachePreferL1); // L1 faster here
      kCNorm2<8, 8, 4, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                             imgSize, numFilters, numImages, sizeX, addScale, powScale);
    }
  } else {
    bool checkCaseBounds = numImages % 128 != 0;
    if (numFilters <= 8) {
      dim3 threads(128);
      dim3 blocks(DIVUP(numImages,128) * imgSize, imgSize);
      if (numFilters == 1) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 1, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 1, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 2) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 2, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 2, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 3) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 3, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 3, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 4) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 4, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 4, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 5) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 5, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 5, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 6) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 6, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 6, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 7) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 7, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 7, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 8) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 8, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 8, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } 
    } else {
      dim3 threads(32, 4);
      dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
      if (checkCaseBounds) {
        cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
        kCNorm_manyfilter<4, 32, 4, 2, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                 imgSize, numFilters, numImages, sizeX, addScale, powScale);
      } else {
        cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
        kCNorm_manyfilter<4, 32, 4, 2, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                 imgSize, numFilters, numImages, sizeX, addScale, powScale);
      }
    }
  }
  getLastCudaError("convResponseNorm: kernel execution failed");
}


