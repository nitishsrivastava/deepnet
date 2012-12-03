### goal: write cudamat and gnumpy functions that 


import ctypes as ct
_ConvNet = ct.cdll.LoadLibrary('_ConvNet.so')
import gnumpy as g

def convUp(images, filters, numModulesX, paddingStart, moduleStride, numImgColors, numGroups = 1):
    numImages = images.shape[1]
    numFilters = filters.shape[1]
    targets = g.zeros((numFilters * numModulesX * numModulesX, numImages))

    _ConvNet.convUp(images._base_as_2d().p_mat, filters._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                    numModulesX, paddingStart, moduleStride, numImgColors, numGroups)

    #void _filterActs(cudamat* images, cudamat* filters, cudamat* targets, int numModulesX, int paddingStart, int moduleStride,
    #                 int numImgColors, int numGroups, float scaleTargets, float scaleOutput, bool conv);

    return targets

def convDown(hidActs, filters, numModulesX, moduleStride, filterSizeX, imSizeX, numImgColors):
    paddingStart = 0 
    numGroups = 1
    numFilters = filters.shape[1]
    numImages = hidActs.shape[1] 

    assert paddingStart <= 0

    numModules = numModulesX**2 

    targets = g.zeros((numImgColors * imSizeX * imSizeX, numImages))

# * hidActs:     (numFilters, numModules, numImages)
# * filters:     (numFilterColors, filterPixels, numFilters)               if conv
# *              (numModules, numFilterColors, filterPixels, numFilters)   otherwise
# * targets:     (numImageColors, imgPixels, numImages)
    
    _ConvNet.convDown(hidActs._base_as_2d().p_mat, filters._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                      imSizeX, paddingStart, moduleStride, numImgColors, numGroups)

    return targets

    #void _imgActs(cudamat* hidActs, cudamat* filters, cudamat* targets, int imgSize, int paddingStart, int moduleStride,
    #              int numImgColors, int numGroups, float scaleTargets, float scaleOutput, bool conv);


def convOutp(images, hidActs, numModulesX, filterSizeX, moduleStride, numImgColors):
    numGroups = 1
    partialSum = 0
    paddingStart = 0
    numImages = images.shape[1]
    numFilters = hidActs.shape[0] / (numModulesX**2)

    targets = g.zeros((numImgColors * filterSizeX * filterSizeX, numFilters))

    _ConvNet.convOutp(images._base_as_2d().p_mat, hidActs._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                         numModulesX, filterSizeX, paddingStart, moduleStride, numImgColors, numGroups, partialSum)

    #void _weightActs(cudamat* images, cudamat* hidActs, cudamat* targets, int numModulesX, int filterSize, int paddingStart, int moduleStride,
    #                 int numImgColors, int numGroups, int partialSum, float scaleTargets, float scaleOutput);

    return targets

def MaxPool(images, 
            numChannels,
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numImages = images.shape[1]

    targets = g.zeros((numChannels * outputsX * outputsX, numImages))
    
    _ConvNet.MaxPool(images._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                  numChannels,
                  subsX,
                  startX,
                  strideX,
                  outputsX
                  )

    return targets

def MaxPoolUndo(images, 
                grad,
                maxes,
                
                subsX,
                startX,
                strideX,
                outputsX
       ):
    
    targets = g.zeros(images.shape)

    _ConvNet.MaxPoolUndo(images._base_as_2d().p_mat, grad._base_as_2d().p_mat, maxes._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                  subsX,
                  startX,
                  strideX,
                  outputsX
                  )

    return targets

def AvgPool(images, 
            numChannels,
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numImages = images.shape[1]
    targets = g.zeros((numChannels * outputsX * outputsX, numImages))
    
    _ConvNet.AvgPool(images._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                     numChannels,
                     subsX,
                     startX,
                     strideX,
                     outputsX,
                     subsX**2,
                     )

    return targets

def AvgPoolUndo(grads, 
            img_w, 
            numChannels,
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numImages = grads.shape[1]
    targets = g.zeros((numChannels * img_w * img_w, numImages))
    
    _ConvNet.AvgPoolUndo(grads._base_as_2d().p_mat, targets._base_as_2d().p_mat,
                     subsX,
                     startX,
                     strideX,
                     outputsX,
                     img_w)

    return targets

##################################################################### not implemented yet ######################################################################
################################################################


def localUp(images, filters):

    numChannels, imSizeX, imSizeX, numImages = images.shape

    ## this is a hell of a filter-matrix. 
    numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape


    assert numModulesX <= imSizeX


    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)

    paddingStart = -(numModulesX - imSizeX + filterSizeX - 1)
    assert paddingStart <= 0

    numModules = numModulesX**2 



    numGroups = 1
    moduleStride = 1  

    targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))

    numImgColors = numChannels






    imagesCu = images._base.p_mat
    filtersCu = filters._base.p_mat
    targetsCu = targets._base.p_mat


    imagesCu_orig, filtersCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (imagesCu, filtersCu, targetsCu)]

    from numpy import prod
    filtersTotSize = filters.size
    filtersCu.contents.size[0] = numFilterChannels * filterSizeX**2 * numModulesX**2
    filtersCu.contents.size[1] = numFilters
    assert filtersTotSize == prod(filtersCu.contents.size)

    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImgColors * imSizeX**2
    imagesCu.contents.size[1] = numImages
    assert imagesTotSize == prod(imagesCu.contents.size)


    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numFilters * numModulesX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size) 

    _ConvNet.localUp(imagesCu,
                   filtersCu,
                   targetsCu,

                   numModulesX,
                   paddingStart,
                   moduleStride,
                   numImgColors,  

                   numGroups,       
                   )

    for i in range(2):
        filtersCu.contents.size[i] = filtersCu_orig[i]
        imagesCu.contents.size[i] = imagesCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets





def localDown(hidActs, filters, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numModulesX_, numModulesX_, numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    ##### I DONT SUPPORT THE FUCKING STRIDE. SHIT.
    assert numModulesX_ == numModulesX
    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1

    numChannels = numFilterChannels * numGroups

    #paddingStart = -(numModulesX - imSizeX + filterSizeX + 1)

    numModules = numModulesX**2 

    targets = g.zeros((numChannels, imSizeX, imSizeX, numImages))

    numImgColors = numChannels




    hidActsCu = hidActs._base.p_mat
    filtersCu = filters._base.p_mat
    targetsCu = targets._base.p_mat


# * hidActs:     (numFilters, numModules, numImages)
# * filters:     (numFilterColors, filterPixels, numFilters)               if conv
# *              (numModules, numFilterColors, filterPixels, numFilters)   otherwise
# * targets:     (numImageColors, imgPixels, numImages)

    
    hidActsCu_orig, filtersCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (hidActsCu, filtersCu, targetsCu)]

    # filters are as before    
    from numpy import prod
    filtersTotSize = filters.size
    filtersCu.contents.size[0] = numFilterChannels * filterSizeX**2 * numModulesX**2
    filtersCu.contents.size[1] = numFilters
    assert filtersTotSize == prod(filtersCu.contents.size)
    
    # hidActs are like the targets of the past:
    hidActsTotSize = hidActs.size
    hidActsCu.contents.size[0] = numFilters * numModulesX**2
    hidActsCu.contents.size[1] = numImages
    assert hidActsTotSize == prod(hidActsCu.contents.size) 

    # targets are like images:
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImgColors * imSizeX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size)


    _ConvNet.localDown(
               hidActsCu,
               filtersCu,
               targetsCu,

               imSizeX,
               paddingStart,
               moduleStride,
               numImgColors,
               numGroups)

    for i in range(2):
        filtersCu.contents.size[i] = filtersCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets














def localOutp(images, hidActs, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  



    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numFilterChannels = numChannels / numGroups

    #imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1
    assert paddingStart <= 0
    filterSizeX = imSizeX - numModulesX + abs(paddingStart) + 1

    #assert partialSum is None
    #partialSum = numModulesX**2 

    targets = g.zeros((numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters))


    numImgColors = numChannels



    hidActsCu = hidActs._base.p_mat
    imagesCu = images._base.p_mat
    targetsCu = targets._base.p_mat

    imagesCu = images._base.p_mat
    imagesCu_orig, hidActsCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (imagesCu, hidActsCu, targetsCu)]

    from pylab import prod
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImgColors * imSizeX**2
    imagesCu.contents.size[1] = numImages
    assert imagesTotSize == prod(imagesCu.contents.size)

    hidActsTotSize = hidActs.size
    hidActsCu.contents.size[0] = numFilters * numModulesX**2
    hidActsCu.contents.size[1] = numImages
    assert hidActsTotSize == prod(hidActsCu.contents.size) 


    from numpy import prod
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numFilterChannels * filterSizeX**2 * numModulesX**2
    targetsCu.contents.size[1] = numFilters
    assert targetsTotSize == prod(targetsCu.contents.size)


    _ConvNet.localOutp(
        imagesCu,
        hidActsCu,
        targetsCu,

        numModulesX,
        filterSizeX,
        paddingStart,
        moduleStride,
        numImgColors,
        numGroups,
        )

    for i in range(2):
        imagesCu.contents.size[i] = imagesCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets

