### goal: write cudamat and gnumpy functions that 


import ctypes as ct
_ConvNet = ct.cdll.LoadLibrary('_ConvNet.so')
import gnumpy as g

def convUp(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups=1):
    """
    images - (n_images, img_w**2 * n_chans)
    filters - (n_filters, filter_w**2 * n_chans)
    targets - (n_images, n_locs**2 * n_filters)
    numModulesX - Number of filter locations along an axis. = n_locs
    paddingStart - Set to -k for a k-pixel border of zeros. Usually set to 0.
    moduleStride - stride to move the filters by. 
    numImgColors - n_chans
    
    """
    numImages = images.shape[0]
    numFilters = filters.shape[0]

    assert targets.shape == (numImages, numFilters * numModulesX * numModulesX), '%s %d %d-%d-%d' % (targets.shape.__str__(), numImages, numFilters, numModulesX, numModulesX)

    _ConvNet.convUp(images.p_mat, filters.p_mat, targets.p_mat,
                    numModulesX, paddingStart, moduleStride, numImgColors, numGroups)

    return targets

def convDown(hidActs, filters, targets, numModulesX, moduleStride, filterSizeX, imSizeX, numImgColors):
    """
    hidActs - (n_images, n_locs**2 * n_filters)
    filters - (n_filters, filter_w**2 * n_chans)
    targets - (n_images, img_w**2 * n_chans)
    """
    paddingStart = 0 
    numGroups = 1
    numFilters = filters.shape[0]
    numImages = hidActs.shape[0] 
    numModules = numModulesX**2 

    assert paddingStart <= 0
    assert targets.shape == (numImages, numImgColors * imSizeX * imSizeX)

    _ConvNet.convDown(hidActs.p_mat, filters.p_mat, targets.p_mat,
                      imSizeX, paddingStart, moduleStride, numImgColors, numGroups)

    return targets

def convOutp(images, hidActs, targets, numModulesX, filterSizeX, moduleStride, numImgColors):
    """
    images - (n_images, img_w**2 * n_chans)
    hidActs - (n_images, n_locs**2 * n_filters)
    targets - (n_filters, filter_w**2 * n_chans)
    """
    numGroups = 1
    partialSum = 0
    paddingStart = 0
    numImages = images.shape[0]
    numFilters = hidActs.shape[1] / (numModulesX**2)

    assert targets.shape == (numFilters, numImgColors * filterSizeX * filterSizeX), '%s %d %d-%d-%d' % (targets.shape.__str__(), numFilters, numImgColors, filterSizeX, filterSizeX)

    _ConvNet.convOutp(images.p_mat, hidActs.p_mat, targets.p_mat,
                      numModulesX, filterSizeX, paddingStart, moduleStride, numImgColors, numGroups, partialSum)

    return targets

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
    return targets

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

    return targets
