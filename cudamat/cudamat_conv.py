import ctypes as ct
import math
import pdb
_ConvNet = ct.cdll.LoadLibrary('libcudamat_conv.so')

def convUp(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups=1):
  """
  images - (n_images, img_w**2 * n_chans)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, n_locs**2 * n_filters)
  numModulesX - Number of filter locations along an axis. = n_locs
  paddingStart - Set to k for a k-pixel border of zeros. Usually set to 0.
  moduleStride - stride to move the filters by. 
  numImgColors - n_chans
  """
  numImages = images.shape[0]
  numFilters = filters.shape[0]

  assert targets.shape == (numImages, numFilters * numModulesX * numModulesX), '%s %d %d-%d-%d' % (targets.shape.__str__(), numImages, numFilters, numModulesX, numModulesX)

  _ConvNet.convUp(images.p_mat, filters.p_mat, targets.p_mat, numModulesX,
                  -paddingStart, moduleStride, numImgColors, numGroups)

def convDown(hidSums, filters, targets, numModulesX, paddingStart, moduleStride, filterSizeX, imSizeX, numImgColors):
  """
  hidSums - (n_images, n_locs**2 * n_filters)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, img_w**2 * n_chans)
  """
  numGroups = 1
  numFilters = filters.shape[0]
  numImages = hidSums.shape[0] 
  numModules = numModulesX**2 

  assert paddingStart >= 0
  assert targets.shape == (numImages, numImgColors * imSizeX * imSizeX)

  _ConvNet.convDown(hidSums.p_mat, filters.p_mat, targets.p_mat, imSizeX,
                    -paddingStart, moduleStride, numImgColors, numGroups)


def convOutp(images, hidSums, targets, numModulesX, paddingStart, filterSizeX, moduleStride, numImgColors):
  """
  images - (n_images, img_w**2 * n_chans)
  hidSums - (n_images, n_locs**2 * n_filters)
  targets - (n_filters, filter_w**2 * n_chans)
  """
  numGroups = 1
  partialSum = 0
  numImages = images.shape[0]
  numFilters = hidSums.shape[1] / (numModulesX**2)

  assert targets.shape == (numFilters, numImgColors * filterSizeX * filterSizeX), '%s %d %d-%d-%d' % (targets.shape.__str__(), numFilters, numImgColors, filterSizeX, filterSizeX)
  _ConvNet.convOutp(images.p_mat, hidSums.p_mat, targets.p_mat, numModulesX, filterSizeX, -paddingStart, moduleStride, numImgColors, 1, 0)

def localUp(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups=1):
  """
  images - (n_images, img_w**2 * n_chans)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, n_locs**2 * n_filters)
  numModulesX - Number of filter locations along an axis. = n_locs
  paddingStart - Set to k for a k-pixel border of zeros. Usually set to 0.
  moduleStride - stride to move the filters by. 
  numImgColors - n_chans
  """
  numImages = images.shape[0]
  numFilters = filters.shape[0]

  assert targets.shape == (numImages, numFilters * numModulesX * numModulesX), '%s %d %d-%d-%d' % (targets.shape.__str__(), numImages, numFilters, numModulesX, numModulesX)

  _ConvNet.localUp(images.p_mat, filters.p_mat, targets.p_mat,
           numModulesX, -paddingStart, moduleStride, numImgColors, numGroups)

def localDown(hidSums, filters, targets, numModulesX, paddingStart, moduleStride, filterSizeX, imSizeX, numImgColors):
  """
  hidSums - (n_images, n_locs**2 * n_filters)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, img_w**2 * n_chans)
  """
  numGroups = 1
  numFilters = filters.shape[0]
  numImages = hidSums.shape[0] 
  numModules = numModulesX**2 

  assert paddingStart >= 0
  assert targets.shape == (numImages, numImgColors * imSizeX * imSizeX)

  _ConvNet.localDown(hidSums.p_mat, filters.p_mat, targets.p_mat,
            imSizeX, -paddingStart, moduleStride, numImgColors, numGroups)


def localOutp(images, hidSums, targets, numModulesX, paddingStart, filterSizeX, moduleStride, numImgColors):
  """
  images - (n_images, img_w**2 * n_chans)
  hidSums - (n_images, n_locs**2 * n_filters)
  targets - (n_filters, filter_w**2 * n_chans)
  """
  numGroups = 1
  partialSum = 0
  numImages = images.shape[0]
  numFilters = hidSums.shape[1] / (numModulesX**2)

  assert targets.shape == (numFilters, numModulesX**2 * numImgColors * filterSizeX**2), '%s %d %d-%d-%d' % (targets.shape.__str__(), numFilters, numImgColors, filterSizeX, filterSizeX)
  _ConvNet.localOutp(images.p_mat, hidSums.p_mat, targets.p_mat,
            numModulesX, filterSizeX, -paddingStart, moduleStride, numImgColors, numGroups, partialSum)



def MaxPool(images, targets, numChannels, subsX, startX, strideX, outputsX):
  """
  images - (n_images, img_w**2 * n_chans)
  numChannels - number of filter/color channels
  subsX - width of pooling area
  startX - pixel where pooling starts
  strideX - stride
  outputsX - number of pooling sites
  """
  numImages = images.shape[0]

  assert targets.shape == (numImages, numChannels * outputsX * outputsX)
  
  _ConvNet.MaxPool(images.p_mat, targets.p_mat,
           numChannels, subsX, startX, strideX, outputsX)

def ProbMaxPool(images, rnd, targets, numChannels, subsX, startX, strideX, outputsX):
  """
  images - (n_images, img_w**2 * n_chans)
  rnd - (n_images, img_w**2 * n_chans)
  numChannels - number of filter/color channels
  subsX - width of pooling area
  startX - pixel where pooling starts
  strideX - stride
  outputsX - number of pooling sites
  """
  numImages = images.shape[0]

  assert targets.shape == (numImages, numChannels * outputsX * outputsX)
  assert rnd.shape == images.shape

  _ConvNet.ProbMaxPool(images.p_mat, rnd.p_mat, targets.p_mat,
           numChannels, subsX, startX, strideX, outputsX)


def MaxPoolUndo(images, targets, grad, maxes,
        subsX, startX, strideX, outputsX):
  """
  images - (n_images, img_w**2 * n_chans)
  grad - (n_images, outputsX**2 * n_chans) cudamat of deltas/gradients of loss wrt layer outputs.
  maxes - (n_images, outputsX**2 * n_chans) cudamat of layer outputs.
  subsX - width of pooling area
  startX - pixel where pooling starts
  strideX - stride
  outputsX - number of pooling sites
  """
  assert targets.shape == images.shape

  _ConvNet.MaxPoolUndo(images.p_mat, grad.p_mat, maxes.p_mat, targets.p_mat,
             subsX, startX, strideX, outputsX)

def ResponseNorm(images, denoms, targets, numChannels, sizeX, addScale, powScale):
  assert targets.shape == images.shape
  assert targets.shape == denoms.shape
  num_images = images.shape[0]
  numpixels = images.shape[1] / numChannels
  imgsize = int(math.sqrt(numpixels))
  #assert images.shape[1] == numChannels * numpixels
  #assert imgsize * imgsize == numpixels
  #pdb.setrace()
  _ConvNet.ResponseNorm(images.p_mat, denoms.p_mat, targets.p_mat,
             numChannels, sizeX, ct.c_float(addScale),
             ct.c_float(powScale))

def ResponseNormUndo(outGrad, denoms, inGrad, acts, targets, numChannels, sizeX,
           addScale, powScale):
  assert targets.shape == outGrad.shape
  assert targets.shape == denoms.shape
  assert targets.shape == inGrad.shape
  assert targets.shape == acts.shape
  _ConvNet.ResponseNormUndo(outGrad.p_mat, denoms.p_mat, inGrad.p_mat,
               acts.p_mat, targets.p_mat, numChannels, sizeX,
               ct.c_float(addScale), ct.c_float(powScale))
