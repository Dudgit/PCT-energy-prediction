import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.filter import layer_filter, getPeakData,getFitData
yModifier = 200 

def getData(pth:str):
    x = np.load(pth+'_x.npy')
    y = np.load(pth+'_y.npy')
    argmaxes = np.argmax(x,axis=1)
    targetLayer = np.bincount(argmaxes).argmax()
    return x,y,targetLayer

def getxy(pth:str,particlelimint:int):
    x,y,targetLayer = getData(pth)
    x,y = layer_filter(x,y,targetLayer)
    x = x[:particlelimint]
    y = y[:particlelimint]
    x = getFitData(x)
    #x = getPeakData(x,targetLayer)
    #realLimit = np.random.randint(170,particlelimint)
    #x,y = x[:realLimit],y[:realLimit]
    #x = np.pad(x,((0,particlelimint-x.shape[0]),(0,0)),'constant',constant_values=(0,0))
    #y = np.pad(y,(0,particlelimint-y.shape[0]),'constant',constant_values=(0,0))
    return x,y/yModifier


def getSingleBatch(bidxs,paths:str = 'None',particleLimit:int = 200):
    x = []
    y = []
    for i in bidxs:
        inputX,inputY = getxy(paths[i],particleLimit)
        x.append(inputX)
        y.append(inputY)
    return torch.from_numpy(np.array(x)),torch.from_numpy(np.array(y))

@torch.no_grad()
def plotEvaluation(preds,targets,writer,epoch):
    fig = plt.figure()
    plt.hist2d(preds.flatten(),targets.flatten(),bins=100)
    plt.plot([np.min(preds.flatten()),200],[np.min(preds.flatten()),200],color='red',linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.title('Predictions vs Targets')
    writer.add_figure('Validation/Hist2D',fig,epoch)

@torch.no_grad()
def plotCroppedEvaluation(preds,targets,writer,epoch):
    fig = plt.figure()
    plt.hist2d(preds.flatten(),targets.flatten(),bins=100)
    plt.plot([100,200],[100,200],color='red',linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.title('Predictions vs Targets')
    plt.xlim(left = 90)
    plt.ylim(bottom = 90)

    writer.add_figure('Validation/Hist2D_cropped',fig,epoch)