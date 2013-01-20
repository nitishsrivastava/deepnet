/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h> //iostream>
#include <assert.h>
#include <nvmatrix_kernels.cuh>
#include <nvmatrix.cuh>
#include <conv_util.cuh>
#include <iostream>

using namespace std;

__device__ float square(const float a) {
    return a*a;
}

/*
 * Block size B_YxB_X.
 * B_X*imgsPerThread*blockIdx.x + threadIdx.x determines img idx 
 * B_Y*blockIdx.y + threadIdx.y determines img row (col if !horiz), channel idx
 * 
 * imgs:        (numChannels, imgPixels, numImages) with given imgStride
 * filter:      (1, 2*radius + 1)
 * target:      (numChannels, imgPixels, numImages)
 * 
 * target can be the same matrix as imgs.
 * radius must be one of 3, 5, 7, 9.
 * 
 * Tried imgsPerThread, slower.
 */
template<int B_Y, int B_X, int radius>
__global__ void kGaussianBlur(float* imgs, float* filter, float* target, const int imgSize,
                              const int numImages, const int imgStride,
                              const bool horiz,
                              const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilter[radius];
    
    const int imgPixels = imgSize * imgSize;
    const int ty = B_Y * blockIdx.y + threadIdx.y;
    const int channelIdx = ty / imgSize;
    const int rowIdx = ty % imgSize;
    const int imgIdx = B_X*blockIdx.x + threadIdx.x;
    const int filterWidth = 2*radius+1;
//    const int tidx = B_Y * threadIdx.y + threadIdx.x;
    if (horiz) {
        imgs += channelIdx * imgPixels * imgStride + rowIdx * imgSize * imgStride + imgIdx;
        target += channelIdx * imgPixels * numImages + rowIdx * imgSize * numImages + imgIdx;
    } else {
        imgs += channelIdx * imgPixels * imgStride + rowIdx * imgStride + imgIdx;
        target += channelIdx * imgPixels * numImages + rowIdx * numImages + imgIdx;
    }
    float outputs[filterWidth-1];
    #pragma unroll
    for (int r = 0; r < filterWidth-1; r++) {
        outputs[r] = 0;
    }
    if (threadIdx.x < filterWidth-1) {
        shFilter[threadIdx.x] = filter[threadIdx.x];
    }
    __syncthreads();

    if (imgIdx < numImages) {
        // This writes radius*2 = filterWidth - 1 values to outputs 
        #pragma unroll
        for (int col = 0; col < radius; col++) {
            float px = imgs[0];
            #pragma unroll
            for (int r = 0; r < radius + 1 + col; r++) {
                outputs[r] += px * shFilter[radius + col - r];
            }
            imgs += horiz ? imgStride : imgStride * imgSize;
        }

        // Unfortunately this has to be at this level of granularity
        if (scaleTargets != 0) {
            for (int col = radius; col < imgSize ; col++) { // loop over img columns
                float px = imgs[0];
                target[0] = scaleTargets * target[0] + scaleOutputs * (outputs[0] + px * shFilter[0]);

                #pragma unroll
                for (int r = 1; r < radius*2; r++) {
                    outputs[r-1] = outputs[r] + px * shFilter[r];
                }
                outputs[filterWidth - 2] = px * shFilter[0];

                imgs += horiz ? imgStride : imgStride * imgSize;
                target += horiz ? numImages : numImages * imgSize;
            }

            #pragma unroll
            for (int r = 0; r < radius; r++) {
                float* t = &target[0];
                t[0] = scaleTargets * t[0] + scaleOutputs * outputs[r];
                target += horiz ? numImages : numImages * imgSize;
            }
        } else {
            for (int col = radius; col < imgSize ; col++) { // loop over img columns
                float px = imgs[0];
                target[0] = scaleOutputs * (outputs[0] + px * shFilter[0]);
                #pragma unroll
                for (int r = 1; r < radius*2; r++) {
                    outputs[r-1] = outputs[r] + px * shFilter[r];
                }
                outputs[filterWidth - 2] = px * shFilter[0];

                imgs += horiz ? imgStride : imgStride * imgSize;
                target += horiz ? numImages : numImages * imgSize;
            }

            #pragma unroll
            for (int r = 0; r < radius; r++) {
                target[0] = scaleOutputs * outputs[r];
                target += horiz ? numImages : numImages * imgSize;
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
 * imgs:        (numChannels, imgPixels, numImages)
 * target:      (numChannels, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int chansPerThread, bool checkCaseBounds>
__global__ void kBedOfNails(float* imgs, float* target, const int imgSize, const int numChannels,
                           const int numImages, const int startX, const int strideX, const int outputsX,
                           const bool reverse, const float scaleTargets, const float scaleOutput) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numChanBlocks = DIVUP(numChannels, B_Y*chansPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numChanBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockChanIdx = (blockIdx.y % numChanBlocks) * B_Y * chansPerThread;
    const int myChanIdx = (blockChanIdx + threadIdx.y*chansPerThread);
    if (myChanIdx >= numChannels) {
        return;
    }
//    if (blockIdx.x != 0 || blockIdx.y != 0) {
//        return;
//    }
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    const int imgPx = startImgPxY * imgSize + startImgPxX;
    
    imgs += myChanIdx * imgPixels * numImages + imgPx * numImages + imgIdx;
    target += (myChanIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    if (scaleTargets != 0) {
        if (!reverse) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        target[c * numOutputs * numImages + i * B_X] = scaleTargets * target[c * numOutputs * numImages + i * B_X] + scaleOutput * imgs[c * imgPixels * numImages + i * B_X]; 
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        imgs[c * imgPixels * numImages + i * B_X] = scaleTargets * imgs[c * imgPixels * numImages + i * B_X] + scaleOutput * target[c * numOutputs * numImages + i * B_X]; 
                    }
                }
            }
        }
    } else {
        if (!reverse) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        target[c * numOutputs * numImages + i * B_X] = scaleOutput * imgs[c * imgPixels * numImages + i * B_X]; 
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        imgs[c * imgPixels * numImages + i * B_X] = scaleOutput * target[c * numOutputs * numImages + i * B_X]; 
                    }
                }
            }
        }
    }

}

/*
 * imgs:        (numChannels, imgPixels, numImages)
 * target:      (numChannels, outputs, numImages)
 */
void _convBedOfNails(NVMatrix& images, NVMatrix& target, int numChannels, int imgSize, int startX, int strideX,
                     bool reverse, float scaleTargets, float scaleOutput) {
    int numImages = reverse ? target.getNumCols() : images.getNumCols();
    int imgPixels = imgSize * imgSize;

    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
    assert(target.isContiguous());
    assert(strideX > 1);
    
    int outputsX = DIVUP(imgSize, strideX);
    int outputs = outputsX * outputsX;
    if (reverse) {
        assert(target.getNumRows() == numChannels * outputs);
    } else  {
        assert(images.getNumRows() == numChannels * imgPixels);
    }
    
    if (scaleTargets == 0) {
        if (reverse) {
            images.resize(numChannels * imgPixels, numImages);
            images.apply(NVMatrixOps::Zero());
        } else {
            target.resize(numChannels*outputs, numImages);
        }
    } else {
        if (reverse) {
            assert(images.getNumRows() == numChannels * outputs);
            assert(images.getNumCols() == numImages);
        } else {
            assert(target.getNumRows() == numChannels * outputs);
            assert(target.getNumCols() == numImages);
        }
    }
    
    bool checkCaseBounds = numImages % 128 != 0;
    
    int chansPerThread = numChannels % 8 == 0 ? 2 : 1;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * outputsX, DIVUP(numChannels, 4 * chansPerThread) * outputsX);
    if (chansPerThread == 1) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 1, true>, cudaFuncCachePreferL1);
            kBedOfNails<4, 32, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                reverse, scaleTargets, scaleOutput);
        } else {
            cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 1, false>, cudaFuncCachePreferL1);
            kBedOfNails<4, 32, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                 imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                 reverse, scaleTargets, scaleOutput);
        }
    } else {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
            kBedOfNails<4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                reverse, scaleTargets, scaleOutput);
        } else {
            cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
            kBedOfNails<4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                 imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                 reverse, scaleTargets, scaleOutput);
        }
    }
}

void convBedOfNails(NVMatrix& images, NVMatrix& target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput) {
    _convBedOfNails(images, target, numChannels, imgSize, startX, strideX, false, scaleTargets, scaleOutput);
}

void convBedOfNailsUndo(NVMatrix& actsGrad, NVMatrix& target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput) {

    _convBedOfNails(target, actsGrad, numChannels, imgSize, startX, strideX, true, scaleTargets, scaleOutput);
}
    

/*
 * imgs:        (numChannels, imgPixels, numImages) with given imgStride
 * filter:      (1, 2*radius + 1)
 * target:      (numChannels, imgPixels, numImages)
 */
void convGaussianBlur(NVMatrix& images, NVMatrix& filter, NVMatrix& target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs) {
    int numImages = images.getNumCols();
    int radius = filter.getNumCols() / 2;
    int imgPixels = images.getNumRows() / numChannels;
    int imgSize = int(sqrt(imgPixels));
    
    assert(imgPixels == imgSize * imgSize);
    assert(radius >= 1 && radius <= 4);
    assert(imgSize >= 2 * radius + 1);
    assert(filter.getNumRows() == 1);
    assert(images.getNumRows() == numChannels * imgPixels);
    assert(!images.isTrans());
    assert(!filter.isTrans());
    assert(!target.isTrans());
    assert(target.isContiguous());
    if (scaleTargets == 0) {
        target.resize(images);
    } else {
        assert(target.isSameDims(images));
    }

    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages, threads.x), DIVUP(numChannels*imgSize, threads.y));

    if (radius == 1) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 1>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 1><<<blocks, threads>>>(images.getDevData(), filter.getDevData(), target.getDevData(),
                                                           imgSize, numImages, images.getStride(), horiz, scaleTargets, scaleOutputs);

    } else if (radius == 2) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 2>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 2><<<blocks, threads>>>(images.getDevData(), filter.getDevData(), target.getDevData(),
                                                           imgSize, numImages, images.getStride(), horiz, scaleTargets, scaleOutputs);

    } else if (radius == 3) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 3>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 3><<<blocks, threads>>>(images.getDevData(), filter.getDevData(), target.getDevData(),
                                                           imgSize, numImages, images.getStride(), horiz, scaleTargets, scaleOutputs);
    } else if (radius == 4) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 4>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 4><<<blocks, threads>>>(images.getDevData(), filter.getDevData(), target.getDevData(),
                                                           imgSize, numImages, images.getStride(), horiz, scaleTargets, scaleOutputs);
    }
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
 * imgs:        (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
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
    meanDiffs  += imgIdx;
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
 * imgs:        (numFilters, imgPixels, numImages)
 * means:       (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
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
 * imgs:        (numFilters, imgPixels, numImages)
 * means:       (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
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
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -sizeX/2 + myPxY;
    const int myStartPxX = -sizeX/2 + myPxX;
    const int myEndPxY = myPxY + DIVUP(sizeX, 2);
    const int myEndPxX = myPxX + DIVUP(sizeX, 2);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
        
    imgs        += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    meanDiffs   += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    
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
//    imgs -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
//    imgs += threadIdx.x;
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
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalAvgUndo(float* avgGrads, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
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
    
    avgGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
            && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += avgGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
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
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i] / (subsX * subsX);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i] / (subsX * subsX);
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
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
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
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
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
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
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
    
    acts        += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
    inputs      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    denoms      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    outGrads    += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    target      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
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
//    outGrads += blockPx * numImages;
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
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
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
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -DIVUP(sizeX,2) + myPxY + 1;
    const int myStartPxX = -DIVUP(sizeX,2) + myPxX + 1;
    const int myEndPxY = myPxY + sizeX/2 + 1;
    const int myEndPxX = myPxX + sizeX/2 + 1;
    
    const int imgIdx = blockImgIdx + threadIdx.x;
        
    acts        += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    inputs      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    outGrads    += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    
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

void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX) {
    convLocalMaxUndo(images, maxGrads, maxActs, target, subsX, startX, strideX, outputsX, 0, 1);
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput) {
    int outputs = outputsX * outputsX;
    int numImages = images.getNumCols();
    int numFilters = maxGrads.getNumRows() / outputs;
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    
    assert(imgSize * imgSize == imgPixels);
    assert(maxGrads.getNumRows() == numFilters * outputs);
    assert(maxGrads.getNumCols() == numImages);
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(!maxGrads.isTrans());
    assert(!maxActs.isTrans());
    assert(images.isContiguous());
    assert(maxGrads.isContiguous());
    assert(maxActs.isContiguous());
    assert(maxGrads.isSameDims(maxActs));
    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(images);
    
    int checkCaseBounds = numImages % 128 != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
    
    if  (checkCaseBounds) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalMaxUndo<4, 32, 4, 2, false, true><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalMaxUndo<4, 32, 4, 2, true, true><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalMaxUndo<4, 32, 4, 2, false, false><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalMaxUndo<4, 32, 4, 2, true, false><<<blocks, threads>>>(images.getDevData(), maxGrads.getDevData(), maxActs.getDevData(), target.getDevData(),
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("convLocalMaxUndo: kernel execution failed");
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalMaxUndoCu(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput) {
    int outputs = outputsX * outputsX;
    int numImages = images->size[0];
    int numFilters = maxGrads->size[1] / outputs;
    int imgPixels = images->size[1] / numFilters;
    assert(images->size[1] == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    
    assert(imgSize * imgSize == imgPixels);
    assert(maxGrads->size[1] == numFilters * outputs);
    assert(maxGrads->size[0] == numImages);

    /*
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(!maxGrads.isTrans());
    assert(!maxActs.isTrans());
    assert(images.isContiguous());
    assert(maxGrads.isContiguous());
    assert(maxActs.isContiguous());
    assert(maxGrads.isSameDims(maxActs));
    */

    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    //target.resize(images);
    
    int checkCaseBounds = numImages % 128 != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
    
    if  (checkCaseBounds) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalMaxUndo<4, 32, 4, 2, false, true><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalMaxUndo<4, 32, 4, 2, true, true><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalMaxUndo<4, 32, 4, 2, false, false><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalMaxUndo<4, 32, 4, 2, true, false><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                            imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("convLocalMaxUndo: kernel execution failed");
}

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target, int subsX, int startX, int strideX, int outputsX, int imgSize) {
    convLocalAvgUndo(avgGrads, target, subsX, startX, strideX, outputsX, imgSize, 0, 1);
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput) {
    int numImages = avgGrads.getNumCols();

    int outputs = outputsX * outputsX;
    int imgPixels = imgSize * imgSize;
    int numFilters = avgGrads.getNumRows() / outputs;
    assert(avgGrads.getNumRows() == numFilters * outputs);

    assert(!target.isTrans());
    assert(!avgGrads.isTrans());
    assert(avgGrads.isContiguous());
    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(numFilters * imgPixels, numImages);
    
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 4)) * imgSize);
    int checkCaseBounds = numImages % 128 != 0;
    
    if (checkCaseBounds) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalAvgUndo<4, 32, 4, 4, false, true><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                   imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                   outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalAvgUndo<4, 32, 4, 4, true, true><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                  outputsX, scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalAvgUndo<4, 32, 4, 4, false, false><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                   imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                   outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalAvgUndo<4, 32, 4, 4, true, false><<<blocks, threads>>>(avgGrads.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                  outputsX, scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("convLocalAvgUndo: kernel execution failed");
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalAvgUndoCu(cudamat* avgGrads, cudamat* target,
                        int subsX, int startX, int strideX, int outputsX, int imgSize,
                        float scaleTargets, float scaleOutput) {
    int numImages = avgGrads->size[0];

    int outputs = outputsX * outputsX;
    int imgPixels = imgSize * imgSize;
    int numFilters = avgGrads->size[1] / outputs;
    cout << numImages << " " << outputs << " " << imgPixels << " " << numFilters << endl;
    assert(avgGrads->size[1] == numFilters * outputs);

    /*
    assert(!target.isTrans());
    assert(!avgGrads.isTrans());
    assert(avgGrads.isContiguous());
    */

    assert(numFilters % 16 == 0);
    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    //target.resize(numFilters * imgPixels, numImages);
    
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 4)) * imgSize);
    int checkCaseBounds = numImages % 128 != 0;
    
    if (checkCaseBounds) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalAvgUndo<4, 32, 4, 4, false, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                   imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                   outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalAvgUndo<4, 32, 4, 4, true, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                  outputsX, scaleTargets, scaleOutput);
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            kLocalAvgUndo<4, 32, 4, 4, false, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                   imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                   outputsX, scaleTargets, scaleOutput);
        } else {
            kLocalAvgUndo<4, 32, 4, 4, true, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                  outputsX, scaleTargets, scaleOutput);
        }
    }

    cutilCheckMsg("convLocalAvgUndo: kernel execution failed");
}

void convResponseNormCu(cudamat* images, cudamat* denoms, cudamat* target, int numFilters, int sizeX, float addScale, float powScale) {
    convContrastNormCu(images, images, denoms, target, numFilters, sizeX, addScale, powScale);
}

void convResponseNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale) {
    convContrastNorm(images, images, denoms, target, numFilters, sizeX, addScale, powScale);
}

/*
 * images:      (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 */
void convContrastNorm(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(meanDiffs.isSameDims(images));
    
    assert(!meanDiffs.isTrans());
    assert(!images.isTrans());
    assert(images.isContiguous());
    assert(meanDiffs.isContiguous());
    assert(numFilters % 16 == 0 || numFilters <= 8);

    target.resize(images);
    denoms.resize(images);

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
            kCNorm2<8, 8, 4, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                           imgSize, numFilters, numImages, sizeX, addScale, powScale);
        } else {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, false>, cudaFuncCachePreferL1); // L1 faster here
            kCNorm2<8, 8, 4, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
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
                    kCNorm_fewfilter<1, 1, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 5) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 6) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 7) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 8) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } 
        } else {
            dim3 threads(32, 4);
            dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, sizeX, addScale, powScale);
            } else {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), meanDiffs.getDevData(), denoms.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, sizeX, addScale, powScale);
            }
        }
    }
    cutilCheckMsg("convResponseNorm: kernel execution failed");
}


void convContrastNormCu(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* target, int numFilters, int sizeX, float addScale, float powScale) {
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
            } else  if (numFilters == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 5) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 6) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 7) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 8) {
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
    cutilCheckMsg("convResponseNorm: kernel execution failed");
}



void convContrastNormUndoCu(cudamat* outGrads, cudamat* denoms, cudamat* meanDiffs, cudamat* acts, cudamat* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput) {
    convResponseNormUndoCu(outGrads, denoms, meanDiffs, acts, target, numFilters, sizeX, addScale, powScale, scaleTargets, scaleOutput);
}
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& meanDiffs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput) {
    convResponseNormUndo(outGrads, denoms, meanDiffs, acts, target, numFilters, sizeX, addScale, powScale, scaleTargets, scaleOutput);
}

/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
void convResponseNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput) {
    int numImages = outGrads.getNumCols();
    int imgPixels = outGrads.getNumRows() / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);

    assert(outGrads.getNumRows() == numFilters * imgPixels);
    
    assert(denoms.isSameDims(outGrads));
    assert(acts.isSameDims(denoms));
    assert(!denoms.isTrans());
    assert(!outGrads.isTrans());
    assert(!acts.isTrans());
    assert(!target.isTrans());
    assert(outGrads.isContiguous());
    
    assert(numFilters % 16 == 0);
    
    target.resize(outGrads);
    
    // First do acts := -2 x scale x acts x outGrads / denoms
    // so that the main routine only has to do an addition in its inner loop.
    int prelimEltsPerThread = 4;
    dim3 threads(128);
    dim3 blocks(MIN(512, DIVUP(outGrads.getNumElements(),(threads.x * prelimEltsPerThread))));
    kRNormUndoPrelims<128, 4><<<blocks, threads>>>(acts.getDevData(), denoms.getDevData(), outGrads.getDevData(), outGrads.getNumElements(), -2*addScale*powScale);
   
    // Now the main routine
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = 8;
        int filtersPerThread = 4;
        int bx = 16;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);

        threads = dim3(bx, 16);
        blocks = dim3(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, true>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, true, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, true>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, false, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, false>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, true, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, false>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, false, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                              target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            }
        }
    } else {
        bool checkCaseBounds = numImages % 128 != 0;
        threads = dim3(32, 4);
        blocks = dim3(DIVUP(numImages,32*2) * imgSize, (numFilters / (4 * 2)) * imgSize);
        if (checkCaseBounds) { 
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, true>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, false, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, true>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, true, true><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, false>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, false, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, false>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, true, false><<<blocks, threads>>>(outGrads.getDevData(), denoms.getDevData(), inputs.getDevData(), acts.getDevData(),
                                                                          target.getDevData(), imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            }
        }
    }


    cutilCheckMsg("kRNormUndo: kernel execution failed");
}
void convResponseNormUndoCu(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput) {
    int numImages = outGrads->size[0];
    int imgPixels = outGrads->size[1] / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);

    assert(outGrads->size[1] == numFilters * imgPixels);
    
    //assert(denoms.isSameDims(outGrads));
    //assert(acts.isSameDims(denoms));
    //assert(!denoms.isTrans());
    //assert(!outGrads.isTrans());
    //assert(!acts.isTrans());
    //assert(!target.isTrans());
    //assert(outGrads.isContiguous());
    
    assert(numFilters % 16 == 0);
    
    //target.resize(outGrads);
    
    // First do acts := -2 x scale x acts x outGrads / denoms
    // so that the main routine only has to do an addition in its inner loop.
    int prelimEltsPerThread = 4;
    dim3 threads(128);
    dim3 blocks(MIN(512, DIVUP(outGrads->size[0]*outGrads->size[1],(threads.x * prelimEltsPerThread))));
    kRNormUndoPrelims<128, 4><<<blocks, threads>>>(acts->data_device, denoms->data_device, outGrads->data_device, outGrads->size[0]*outGrads->size[1], -2*addScale*powScale);
   
    // Now the main routine
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = 8;
        int filtersPerThread = 4;
        int bx = 16;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);

        threads = dim3(bx, 16);
        blocks = dim3(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, true>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, true, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                              target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, true>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, false, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                              target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, false>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, true, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                              target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, false>, cudaFuncCachePreferL1);
                kRNormUndo2<16, 8, 4, false, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                              target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
            }
        }
    } else {
        bool checkCaseBounds = numImages % 128 != 0;
        threads = dim3(32, 4);
        blocks = dim3(DIVUP(numImages,32*2) * imgSize, (numFilters / (4 * 2)) * imgSize);
        if (checkCaseBounds) { 
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, true>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, false, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, true>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, true, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, false>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, false, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, false>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 2, 2, true, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            }
        }
    }


    cutilCheckMsg("kRNormUndo: kernel execution failed");
}
