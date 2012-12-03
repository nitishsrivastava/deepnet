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
images = g.randn((numFilters, imSizeX, imSizeX, numImages))+1
images[:, 3, 3, :] = 2

from cudamat_conv import MaxPool, MaxPoolUndo
from cudamat_conv.cudamat_conv_py import MaxPoolUndo as MaxPoolUndo_py

M = MaxPool(images,
             subsX = 3,
             startX = 0,
             strideX = 2,
             outputsX = imSizeX/2,
             )

grad = g.randn(*M.shape)



print '.'
t2 = MaxPoolUndo_py(images, 
                    grad,
                    M, 
                    subsX = 2,
                    startX = 0,
                    strideX = 2)
print '.'
t1 = MaxPoolUndo(images, 
                 grad,
                 M, 
                subsX = 2,
                startX = 0,
                strideX = 2)
print '.'


print 'max unpooling:'
print abs(t1).mean()
print abs(t2).mean()
print abs(t1-t2).mean()

