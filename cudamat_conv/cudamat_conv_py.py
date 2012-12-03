import gnumpy as g
def convUp(images, filters, paddingStart = 0):
    assert paddingStart <= 0

    

    numChannels, imSizeX, imSizeX, numImages = images.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    moduleStride = 1  

    numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    numModules = numModulesX**2 
    numGroups = 1

    targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))

    images2 = g.zeros((numChannels, 
                       imSizeX+2*abs(paddingStart), 
                       imSizeX+2*abs(paddingStart), 
                       numImages))
    if paddingStart != 0:
        images2[:, 
            abs(paddingStart):-abs(paddingStart),
            abs(paddingStart):-abs(paddingStart),
            :] = images
    else:
        images2 = images


    for i in range(numImages):
        for f in range(numFilters):
            for c in range(numChannels):
                for y1 in range(numModulesX):
                    for y2 in range(numModulesX):
                        for u1 in range(filterSizeX):
                            for u2 in range(filterSizeX):
                                x1 = y1 + u1 
                                x2 = y2 + u2
                                targets[f, y1, y2, i] += \
                                    filters[c ,u1,u2,f] * \
                                    images2[c,x1,x2,i]
    return targets


def localUp(images, filters, count_unused=False):
    #assert paddingStart <= 0

    

    numChannels, imSizeX, imSizeX, numImages = images.shape
    numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    assert numModulesX <= imSizeX

    moduleStride = 1  

    paddingStart = -(numModulesX - imSizeX + filterSizeX - 1)

    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    numModules = numModulesX**2 
    numGroups = 1




    targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))

    images2 = g.zeros((numChannels, 
                       imSizeX+2*abs(paddingStart), 
                       imSizeX+2*abs(paddingStart), 
                       numImages))
    if paddingStart != 0:
        images2[:, 
            abs(paddingStart):-abs(paddingStart),
            abs(paddingStart):-abs(paddingStart),
            :] = images
    else:
        images2 = images


    used=0

    for i in range(numImages):
        for f in range(numFilters):
            for c in range(numChannels):
                for y1 in range(numModulesX):
                    for y2 in range(numModulesX):
                        for u1 in range(filterSizeX):
                            for u2 in range(filterSizeX):
                                x1 = y1 + u1 
                                x2 = y2 + u2
                                targets[f, y1, y2, i] += \
                                    filters[y1, y2, c ,u1,u2,f] * \
                                    images2[c,x1,x2,i]
                                # if images2 is exactly zero, it means we're the victims of padding.
                                used += (images2[c,x1,x2,i]!=0)

    if count_unused:
        unused = numImages*filters.size - used
        assert unused % numImages == 0
        print 'localUp: num unused filters: %s' % (unused / numImages)

    return targets



### Do the convDown. The end really is near.  Boy Jolly I am happy.
def convDown(hidActs, filters, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1

    numChannels = numFilterChannels * numGroups


    numModules = numModulesX**2 

    targets = g.zeros((numChannels, imSizeX, imSizeX, numImages))
    targets2 = g.zeros((numChannels, 
                        imSizeX + 2*abs(paddingStart), 
                        imSizeX + 2*abs(paddingStart), 
                        numImages))

    numImgColors = numChannels




    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape


    moduleStride = 1  

    numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    numModules = numModulesX**2 
    numGroups = 1

    #targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))



    for i in range(numImages):
        for f in range(numFilters):
            for c in range(numChannels):
                for y1 in range(numModulesX):
                    for y2 in range(numModulesX):
                        for u1 in range(filterSizeX):
                            for u2 in range(filterSizeX):
                                x1 = y1 + u1 
                                x2 = y2 + u2
                                # targets[f, y1, y2, i] += \
                                #     filters[c ,u1,u2,f] * \
                                #     images2[c,x1,x2,i]
                                targets2[c,x1,x2,i] += \
                                    filters[c ,u1,u2,f] * \
                                    hidActs[f, y1, y2, i]


    if paddingStart != 0:
        targets[:] = targets2[:, 
            abs(paddingStart):-abs(paddingStart),
            abs(paddingStart):-abs(paddingStart),
            :] 
    else:
        targets = targets2

    return targets



def localDown(hidActs, filters, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    # what about the stride?  I don't support stride.  I don't like it. 
    #paddingStart = -(numModulesX - imSizeX + filterSizeX + 1)

    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1

    numChannels = numFilterChannels * numGroups

    numModules = numModulesX**2 

    targets = g.zeros((numChannels, imSizeX, imSizeX, numImages))
    targets2 = g.zeros((numChannels, 
                        imSizeX + 2*abs(paddingStart), 
                        imSizeX + 2*abs(paddingStart), 
                        numImages))

    numImgColors = numChannels




    #numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    #numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape


    moduleStride = 1  

    numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    numModules = numModulesX**2 
    numGroups = 1

    #targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))



    for i in range(numImages):
        for f in range(numFilters):
            for c in range(numChannels):
                for y1 in range(numModulesX):
                    for y2 in range(numModulesX):
                        for u1 in range(filterSizeX):
                            for u2 in range(filterSizeX):
                                x1 = y1 + u1 
                                x2 = y2 + u2
                                # targets[f, y1, y2, i] += \
                                #     filters[c ,u1,u2,f] * \
                                #     images2[c,x1,x2,i]
                                targets2[c,x1,x2,i] += \
                                    filters[y1, y2, c ,u1,u2,f] * \
                                    hidActs[f, y1, y2, i]


    if paddingStart != 0:
        targets[:] = targets2[:, 
            abs(paddingStart):-abs(paddingStart),
            abs(paddingStart):-abs(paddingStart),
            :] 
    else:
        targets = targets2

    return targets














def convOutp(images, hidActs, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numFilterChannels = numChannels / numGroups
    filterSizeX = imSizeX - numModulesX + abs(paddingStart) + 1

    targets = g.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))




    numImgColors = numChannels

    images2 = g.zeros((numChannels, 
                       imSizeX+2*abs(paddingStart), 
                       imSizeX+2*abs(paddingStart), 
                       numImages))

    if paddingStart != 0:
        images2[:, 
            abs(paddingStart):-abs(paddingStart),
            abs(paddingStart):-abs(paddingStart),
            :] = images
    else:
        images2 = images


    for i in range(numImages):
        for f in range(numFilters):
            for c in range(numChannels):
                for y1 in range(numModulesX):
                    for y2 in range(numModulesX):
                        for u1 in range(filterSizeX):
                            for u2 in range(filterSizeX):
                                x1 = y1 + u1 
                                x2 = y2 + u2
                                # targets[f, y1, y2, i] += \
                                #     filters[c ,u1,u2,f] * \
                                #     images2[c,x1,x2,i]

                                targets[c ,u1,u2,f] += \
                                    hidActs[f, y1, y2, i] * \
                                    images2[c,x1,x2,i]


    return targets



def localOutp(images, hidActs, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numFilterChannels = numChannels / numGroups
    filterSizeX = imSizeX - numModulesX + abs(paddingStart) + 1

    targets = g.zeros((numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters))


    numImgColors = numChannels

    images2 = g.zeros((numChannels, 
                       imSizeX+2*abs(paddingStart), 
                       imSizeX+2*abs(paddingStart), 
                       numImages))

    if paddingStart != 0:
        images2[:, 
            abs(paddingStart):-abs(paddingStart),
            abs(paddingStart):-abs(paddingStart),
            :] = images
    else:
        images2 = images


    for i in range(numImages):
        for f in range(numFilters):
            for c in range(numChannels):
                for y1 in range(numModulesX):
                    for y2 in range(numModulesX):
                        for u1 in range(filterSizeX):
                            for u2 in range(filterSizeX):
                                x1 = y1 + u1 
                                x2 = y2 + u2
                                # targets[f, y1, y2, i] += \
                                #     filters[c ,u1,u2,f] * \
                                #     images2[c,x1,x2,i]

                                targets[y1,y2, c ,u1,u2,f] += \
                                    hidActs[f, y1, y2, i] * \
                                    images2[c,x1,x2,i]


    return targets




def MaxPool(images, 
               subsX,
               startX,
               strideX,
               outputsX
               ):
    

    numChannels, imSizeX, imSizeX, numImages = images.shape    




    numImgColors = numChannels


    targets = g.zeros((numChannels, outputsX, outputsX, numImages)) - 1e100

    def max(a,b):
        return a if a>b else b

    for i in range(numImages):
        for c in range(numChannels):
            o1 = 0
            for s1 in range(startX, imSizeX, strideX):
                if s1<0:
                    continue
                o2 = 0
                for s2 in range(startX, imSizeX, strideX):
                    if s2<0:
                        continue
                    for u1 in range(subsX):
                        for u2 in range(subsX):
                            try:
                                targets[c,o1,o2,i] = max(images[c,s1+u1,s2+u2,i],
                                                         targets[c,o1,o2,i])
                            except IndexError:
                                pass #?
                           
                    o2 += 1
                o1 += 1

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
    numImgColors = numChannels

    numChannels, outputsX_, outputsX, numImages = maxes.shape
    assert outputsX_ == outputsX
    
    assert maxes.shape == grad.shape
    targets = g.zeros(images.shape)


    for i in range(numImages):
        for c in range(numChannels):
            o1 = 0
            for s1 in range(startX, imSizeX, strideX):
                if s1<0:
                    continue
                o2 = 0
                for s2 in range(startX, imSizeX, strideX):
                    if s2<0:
                        continue
                    for u1 in range(subsX):
                        for u2 in range(subsX):
                            try:
                                if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
                                    targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]

                            except IndexError:
                                pass #??  I don't fucking get it.
                           
                    o2 += 1
                o1 += 1

    return targets






def AvgPool(images, 
               subsX,
               startX,
               strideX,
               outputsX
               ):
    

    numChannels, imSizeX, imSizeX, numImages = images.shape    




    numImgColors = numChannels


    targets = g.zeros((numChannels, outputsX, outputsX, numImages)) 


    for i in range(numImages):
        for c in range(numChannels):
            o1 = 0
            for s1 in range(startX, imSizeX, strideX):
                o2 = 0
                for s2 in range(startX, imSizeX, strideX):
                    for u1 in range(subsX):
                        for u2 in range(subsX):
                            try:
                                targets[c,o1,o2,i] += images[c,s1+u1,s2+u2,i]
                                                      
                            except IndexError:
                                pass #?
                           
                    o2 += 1
                o1 += 1

    return targets/subsX**2
