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
filters = g.randn((numModulesX, numModulesX, numFilterColors, filterSizeX, filterSizeX, numFilters))


from cudamat_conv import localUp, localDown
from cudamat_conv.cudamat_conv_py import localUp as localUp_py, localDown as localDown_py

T1 = localUp(images, filters) #, paddingStart=-1)
t1 = localDown(T1, filters) #, paddingStart=-1)
t1_py = localDown_py(T1, filters)#, paddingStart=-1)
assert t1.shape==t1_py.shape
print 't1 = ',abs(t1).mean()
print 't1_py = ',abs(t1_py).mean()
print 't1_diff = ',abs(t1-t1_py).mean()
print 't1.shape = ', t1.shape
print

# T2 = localUp(images, filters, paddingStart=0)
# t2 = localDown(T2, filters, paddingStart=0)
# t2_py = localDown_py(T2, filters, paddingStart=0)
# assert t2.shape==t2_py.shape

# print 't2 = ',abs(t2).mean()
# print 't2_py = ',abs(t2_py).mean()
# print 't2_diff = ',abs(t2-t2_py).mean()
# print 't2.shape = ', t2.shape
# print

filters = g.randn((numModulesX+2, numModulesX+2, numFilterColors, filterSizeX, filterSizeX, numFilters))
T3 = localUp(images, filters)#, paddingStart=-2)
t3 = localDown(T3, filters, paddingStart=-2)
t3_py = localDown_py(T3, filters, paddingStart=-2)
assert t3.shape==t3_py.shape

print 't3 = ',abs(t3).mean()
print 't3_py = ',abs(t3_py).mean()
print 't3_diff = ',abs(t3-t3_py).mean()
print 't3.shape = ', t3.shape
print 


