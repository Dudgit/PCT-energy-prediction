import numpy as np

def layer_filter(x,y,targetlayer):
    argmxs = np.argmax(x, axis=1)
    c1 = argmxs == float(targetlayer)
    c2 = argmxs == targetlayer-1
    c3 = argmxs == targetlayer+1
    x = x[c1 | c2 | c3]
    y = y[c1 | c2 | c3]
    return x,y

def getPeakData(x,targetlayer):
    x1 = x[:,targetlayer]
    x2 = x[:,targetlayer-1]
    x3 = x[:,targetlayer+1]
    x = np.column_stack((x1,x2,x3))
    return x

def getFitData(x):
    x1 = np.argmax(x,axis=1)
    idxhelp = np.arange(x.shape[0]) 
    tmp = x[idxhelp,x1]
    x2 = x[idxhelp,x1-1]
    x3 = x[idxhelp,x1+1]
    return np.column_stack((x1,x2/tmp,x3/tmp))

def getFitData_moreData(x,targetlayer):
    x1 = np.argmax(x,axis=1)
    prevData = x[:,targetlayer-10:targetlayer+1]
    return np.column_stack((x1/x.shape[1],prevData))