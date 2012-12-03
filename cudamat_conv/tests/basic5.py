import cudamat_conv.cudamat_conv
reload (cudamat_conv.cudamat_conv)
import gnumpy as g

imSizeX = 10
numImages = 2
filterSizeX = 3
numChannels = 3
numGroups = 1
assert numChannels % numGroups == 0
numFilterColors = numChannels / numGroups
numFilters = 16 * numGroups

moduleStride = 1  

numModulesX = (imSizeX - filterSizeX + 1)
numModules = numModulesX**2 

### TODO: ask Alex about moduleStride and numGroups.
### But ignoring these I'm good to go.

paddingStart = 0 ## try it without padding for now.

numImgColors = numChannels



# create the images
images = g.randn((numChannels, imSizeX, imSizeX, numImages))+1
images[:, 3, 3, :] = 2

from cudamat_conv import MaxPool, AvgPool
from cudamat_conv.cudamat_conv_py import MaxPool as MaxPool_py, AvgPool as AvgPool_py

t1 = MaxPool(images,
             subsX = 3,
             startX = 0,
             strideX = 2,
             outputsX = imSizeX/2,
             )

t2 = MaxPool_py(images,
             subsX = 3,
             startX = 0,
             strideX = 2,
             outputsX = imSizeX/2
                )

print 'max pooling:'
print abs(t1).mean()
print abs(t2).mean()
print abs(t1-t2).mean()



t1 = AvgPool(images,
             subsX = 3,
             startX = 0,
             strideX = 2,
             outputsX = imSizeX/1,
             )

t2 = AvgPool_py(images,
                subsX = 3,
                startX = 0,
                strideX = 2,
                outputsX = imSizeX/1
                )

print 'avg pooling:'
print abs(t1).mean()
print abs(t2).mean()
print abs(t1-t2).mean()

