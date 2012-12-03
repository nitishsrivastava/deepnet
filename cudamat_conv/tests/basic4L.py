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



from cudamat_conv import localUp, localOutp
from cudamat_conv.cudamat_conv_py import localOutp as localOutp_py

filters = g.randn((numModulesX+1, numModulesX+1, numFilterColors, filterSizeX, filterSizeX, numFilters))
T1 = localUp(images, filters)#, paddingStart=-1)
print 'a'
t1 = localOutp(images, T1, paddingStart=-1)
print 'b'
t1_py = localOutp_py(images, T1, paddingStart=-1)
print 'c'
assert t1.shape==t1_py.shape
print 't1 = ',abs(t1).mean()
print 't1_py = ',abs(t1_py).mean()
print 't1_diff = ',abs(t1-t1_py).mean()
print 't1.shape = ', t1.shape
print

filters = g.randn((numModulesX, numModulesX, numFilterColors, filterSizeX, filterSizeX, numFilters))
T2 = localUp(images, filters)#, paddingStart=0)
print 'a'
t2 = localOutp(images, T2, paddingStart=0)
print 'b'
t2_py = localOutp_py(images, T2, paddingStart=0)
print 'c'
assert t2.shape==t2_py.shape

print 't2 = ',abs(t2).mean()
print 't2_py = ',abs(t2_py).mean()
print 't2_diff = ',abs(t2-t2_py).mean()
print 't2.shape = ', t2.shape
print

filters = g.randn((numModulesX+2, numModulesX+2, numFilterColors, filterSizeX, filterSizeX, numFilters))

T3 = localUp(images, filters)#, paddingStart=-2)
print 'a' 
t3 = localOutp(images, T3, paddingStart=-2)
print 'b' 
t3_py = localOutp_py(images, T3, paddingStart=-2)
print 'c'
assert t3.shape==t3_py.shape

print 't3 = ',abs(t3).mean()
print 't3_py = ',abs(t3_py).mean()
print 't3_diff = ',abs(t3-t3_py).mean()
print 't3.shape = ', t3.shape
print 


