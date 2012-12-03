### goal: write cudamat and gnumpy functions that 


import ctypes as ct
_ConvNet = ct.cdll.LoadLibrary('_ConvNet.so')

import gnumpy as g



def convUp(images, filters, moduleStride = 1, paddingStart = 0):
    assert paddingStart <= 0
    numChannels, imSizeX, imSizeX, numImages = images.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    assert (abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0

    numModulesX = (abs(paddingStart) + imSizeX - filterSizeX)/moduleStride + 1
    numModules = numModulesX**2 

    numGroups = 1
    #moduleStride = 1  

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
    filtersCu.contents.size[0] = numFilterChannels * filterSizeX**2
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

    _ConvNet.convUp(imagesCu,
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





def convDown(hidActs, filters, moduleStride = 1, paddingStart = 0):
    numGroups = 1

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape


    numModules = numModulesX**2 
    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    #imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1
    imSizeX = (numModulesX - 1) * moduleStride - abs(paddingStart) + filterSizeX

    #assert (abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
    numChannels = numFilterChannels * numGroups

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
    filtersCu.contents.size[0] = numFilterChannels * filterSizeX**2
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


    _ConvNet.convDown(
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














def convOutp(images, hidActs, moduleStride = 1, paddingStart = 0, partialSum = None):
    numGroups = 1

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numFilterChannels = numChannels / numGroups

    #imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1
    #filterSizeX = (imSizeX - numModulesX + abs(paddingStart))/moduleStride + 1
    filterSizeX = -(numModulesX - 1) * moduleStride + (abs(paddingStart) + imSizeX)


    assert partialSum is None
    partialSum = numModulesX**2 

    targets = g.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))


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
    targetsCu.contents.size[0] = numFilterChannels * filterSizeX**2
    targetsCu.contents.size[1] = numFilters
    assert targetsTotSize == prod(targetsCu.contents.size)



    _ConvNet.convOutp(
        imagesCu,
        hidActsCu,
        targetsCu,

        numModulesX,
        filterSizeX,
        paddingStart,
        moduleStride,
        numImgColors,
        numGroups,
        partialSum
        )

    for i in range(2):
        imagesCu.contents.size[i] = imagesCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets



def MaxPool(images, 
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numImgColors = numChannels

    targets = g.zeros((numChannels, outputsX, outputsX, numImages))

    
    imagesCu = images._base.p_mat
    targetsCu = targets._base.p_mat

    from pylab import prod
    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImgColors * imSizeX**2
    imagesCu.contents.size[1] = numImages
    assert imagesTotSize == prod(imagesCu.contents.size)

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImgColors * outputsX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size)

    numFilters = numImgColors

    _ConvNet.MaxPool(imagesCu,
                  targetsCu,
                  numFilters,
                  subsX,
                  startX,
                  strideX,
                  outputsX
                  )

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]

    return targets












def MaxPoolUndo(images, 
                grad,
                maxes,


                subsX,
                startX,
                strideX,
       ):
    
    numChannels, imSizeX_, imSizeX, numImages = images.shape    
    assert imSizeX_ == imSizeX
    numChannels = numChannels

    numChannels, outputsX_, outputsX, numImages = maxes.shape
    assert outputsX_ == outputsX
    
    assert maxes.shape == grad.shape 
    targets = g.zeros(images.shape)

    assert numChannels % 16 == 0
    

    imagesCu = images._base.p_mat
    maxesCu = maxes._base.p_mat
    gradCu = grad._base.p_mat
    targetsCu = targets._base.p_mat


    from pylab import prod
    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numChannels * imSizeX**2
    imagesCu.contents.size[1] = numImages
    assert imagesTotSize == prod(imagesCu.contents.size)

    from pylab import prod
    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numChannels * imSizeX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size)



    maxesCu_orig = tuple(maxesCu.contents.size)
    maxesTotSize = maxes.size
    maxesCu.contents.size[0] = numChannels * outputsX**2
    maxesCu.contents.size[1] = numImages
    assert maxesTotSize == prod(maxesCu.contents.size)

    gradCu_orig = tuple(gradCu.contents.size)
    gradTotSize = grad.size
    gradCu.contents.size[0] = numChannels * outputsX**2
    gradCu.contents.size[1] = numImages
    assert gradTotSize == prod(gradCu.contents.size)


    _ConvNet.MaxPoolUndo(imagesCu,
                     gradCu,
                     maxesCu,
                     targetsCu,

                  subsX,
                  startX,
                  strideX,
                  outputsX
                  )

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]
        gradCu.contents.size[i]=gradCu_orig[i]
        maxesCu.contents.size[i]=maxesCu_orig[i]


    return targets




## dosen't work for some reason.  Investigate the reason.
def AvgPool(images, 
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numImgColors = numChannels

    targets = g.zeros((numChannels, outputsX, outputsX, numImages))

    
    imagesCu = images._base.p_mat
    targetsCu = targets._base.p_mat

    from pylab import prod
    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImgColors * imSizeX**2
    imagesCu.contents.size[1] = numImages
    assert imagesTotSize == prod(imagesCu.contents.size)

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImgColors * outputsX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size)

    numFilters = numImgColors

    _ConvNet.AvgPool(imagesCu,
                     targetsCu,
                     numFilters,
                     subsX,
                     startX,
                     strideX,
                     outputsX,
                     subsX**2,
                     )

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]

    return targets















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

