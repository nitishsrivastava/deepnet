from cudamat_conv.cudamat_conv import _ConvNet as ConvNet
import gnumpy as g

imSizeX = 5
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
images = g.randn((numChannels, imSizeX, imSizeX, numImages))
filters = g.randn((numFilterColors, filterSizeX, filterSizeX, numFilters))
targets = g.randn((numFilters, numModulesX, numModulesX, numImages))

# extract the cudamat objects from the gnumpy objects.  We need to reshape
# it appropriately. 
imagesCu = images._base.p_mat
filtersCu = filters._base.p_mat
targetsCu = targets._base.p_mat




# Ok. This shit is in fortran order. What does it mean? 
# Probably nothing at all good.
zero, one = 0, 1 

abs(filters).mean()

from numpy import prod
filtersTotSize = filters.size
filtersCu.contents.size[zero] = numFilterColors * filterSizeX**2 #* numGroups
filtersCu.contents.size[one] = numFilters
assert filtersTotSize == prod(filtersCu.contents.size)

abs(images).mean()

imagesTotSize = images.size
imagesCu.contents.size[zero] = numImgColors * imSizeX**2
imagesCu.contents.size[one] = numImages
assert imagesTotSize == prod(imagesCu.contents.size)

#import pdb
#pdb.set_trace()

print 'pre-pre-call:', abs(targets).mean()

## resize the cudamat target thingie: 
targetsTotSize = targets.size
targetsCu.contents.size[zero] = numFilters * numModulesX**2
targetsCu.contents.size[one] = numImages
assert targetsTotSize == prod(targetsCu.contents.size) # nothing's changed.
### then call the thing.  Why won't I get a single call?  And why
### would these things change?   Why does the first time differ
### from the second?  It doesn't make any fucking sense.

ConvNet.convUp(imagesCu,
               filtersCu,
               targetsCu,

               numModulesX,
               paddingStart,
               moduleStride,
               numImgColors,  # I thought I got everything till this point.

               numGroups,       # I get the group business only very slightly.
               )
               #0, 1)


#void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
#                    int numModulesX, int paddingStart, int moduleStride,
#                    int numImgColors, int numGroups);

print 'post-call:', abs(targets).mean()

targets2 = targets*0

# next, we want to compute the targets manually.  If I am successful,
# then it's really cool and awesmoe. 
#images = g.randn((numChannels, imSizeX, imSizeX, numImages))
#filters = g.randn((numFilterColors, filterSizeX, filterSizeX, numFilters))
#targets = g.randn((numFilters, numModulesX, numModulesX, numImages))

filters = filters.as_numpy_array()
images = images.as_numpy_array()

for i in range(numImages):
    for f in range(numFilters):
        for c in range(numChannels):
            for y1 in range(numModulesX):
                for y2 in range(numModulesX):
                    for u1 in range(filterSizeX):
                        for u2 in range(filterSizeX):
                            x1 = y1 + u1
                            x2 = y2 + u2
                            targets2[f, y1, y2, i] += \
                                filters[c ,u1,u2,f] * \
                                images[c,x1,x2,i]


print abs(targets2).mean()
print abs(targets-targets2).mean()
### Hallelujah!!!! :D Holy-shit!!!  That was so much easier!!!
