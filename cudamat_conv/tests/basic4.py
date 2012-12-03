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


from cudamat_conv import convUp, convOutp
from cudamat_conv.cudamat_conv_py import convOutp as convOutp_py

T1 = convUp(images, filters, paddingStart=-1)
t1 = convOutp(images, T1, paddingStart=-1)
t1_py = convOutp_py(images, T1, paddingStart=-1)
assert t1.shape==t1_py.shape
print 't1 = ',abs(t1).mean()
print 't1_py = ',abs(t1_py).mean()
print 't1_diff = ',abs(t1-t1_py).mean()
print 't1.shape = ', t1.shape
print

T2 = convUp(images, filters, paddingStart=0)
t2 = convOutp(images, T2, paddingStart=0)
t2_py = convOutp_py(images, T2, paddingStart=0)
assert t2.shape==t2_py.shape

print 't2 = ',abs(t2).mean()
print 't2_py = ',abs(t2_py).mean()
print 't2_diff = ',abs(t2-t2_py).mean()
print 't2.shape = ', t2.shape
print

T3 = convUp(images, filters, paddingStart=-2)
t3 = convOutp(images, T3, paddingStart=-2)
t3_py = convOutp_py(images, T3, paddingStart=-2)
assert t3.shape==t3_py.shape

print 't3 = ',abs(t3).mean()
print 't3_py = ',abs(t3_py).mean()
print 't3_diff = ',abs(t3-t3_py).mean()
print 't3.shape = ', t3.shape
print 


