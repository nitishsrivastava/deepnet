#include "cudamat.cuh"
#include "cudaconv2/conv_util.cuh"
#include "cudaconv2/cudaconv2.cuh"
#include "nvmatrix/nvmatrix.cuh"

/*
 * images:      (numImgColors, imgPixels, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)
 * targets:     (numFilters, numModules, numImages)
 */

// Convolutions.
extern "C" void convUp(cudamat* images, cudamat* filters, cudamat* targets, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups){
  _filterActsCu(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, true);
}
extern "C" void convDown(cudamat* images, cudamat* filters, cudamat* targets, int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups){
  _imgActsCu(images, filters, targets, imgSize, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, true);
}
extern "C" void convOutp(cudamat* images, cudamat* hidSums, cudamat* targets, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum){
  _weightActsCu(images, hidSums, targets, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, partialSum, 0, 1);
}

// Local Connections.
extern "C" void localUp(cudamat* images, cudamat* filters, cudamat* targets, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups){
  _filterActsCu(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, false);
}
extern "C" void localDown(cudamat* images, cudamat* filters, cudamat* targets, int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups){
  _imgActsCu(images, filters, targets, imgSize, paddingStart, moduleStride, numImgColors, numGroups, 0, 1, false);
}
extern "C" void localOutp(cudamat* images, cudamat* hidSums, cudamat* targets, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum){
  _weightActsCu(images, hidSums, targets, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, 1, 0, 1);
}

// Response Normalization.
extern "C" void ResponseNorm(cudamat* images, cudamat* denoms, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convResponseNormCu(images, denoms, targets, numFilters, sizeX, addScale,  powScale);
}

extern "C" void ResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convResponseNormUndoCu(outGrads, denoms, inputs, acts, targets, numFilters, sizeX, addScale, powScale, 0, 1);
}

// Contrast Normalization.
extern "C" void ContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convContrastNormCu(images, meanDiffs, denoms, targets, numFilters, sizeX, addScale,  powScale);
}

extern "C" void ContrastNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* meanDiffs, cudamat* acts, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convContrastNormUndoCu(outGrads, denoms, meanDiffs, acts, targets, numFilters, sizeX, addScale, powScale, 0, 1);
}

// Pooling.
extern "C" void MaxPool(cudamat* images, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  MaxPooler mpooler;
  convLocalPoolCu<MaxPooler>(images, targets, numFilters, subsX, startX, strideX, outputsX, mpooler);
}

extern "C" void MaxPoolUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* targets, int subsX, int startX, int strideX, int outputsX){
  convLocalMaxUndoCu(images, maxGrads, maxActs, targets, subsX, startX, strideX, outputsX, 0, 1);
}

