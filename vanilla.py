import numpy as np
from utils.filter import layer_filter, getPeakData,getFitData
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def getData(pth:str):
    x = np.load(pth+'_x.npy')
    y = np.load(pth+'_y.npy')
    argmaxes = np.argmax(x,axis=1)
    targetLayer = np.bincount(argmaxes).argmax()
    return x,y,targetLayer

def fitfunc(x,a,b,c):
    return a*x[:,0]+b*x[:,1]+c*x[:,2]

def getxy(pth:str,particlelimint:int):
    x,y,targetLayer = getData(pth)
    x,y = layer_filter(x,y,targetLayer)
    x = x[:particlelimint]
    y = y[:particlelimint]
    x = getFitData(x)#x = getPeakData(x,targetLayer)
    return x,y


def fitStatistic(wptList):
    particleLimit = 200
    inputX,inputY = getxy(f'data/wpt_{wptList[0]}/1',particleLimit)
    for wpt in wptList:
        print('Fitting for WPT:',wpt)
        for i in range(1,850):
            x,y = getxy(f'data/wpt_{wpt}/{i}',particleLimit)
            inputX = np.concatenate((inputX,x))
            inputY = np.concatenate((inputY,y))
    inputY = np.array(inputY)
    popt,pcov = curve_fit(fitfunc,inputX,inputY)
    del inputX,inputY
    np.save('popts/vanilla.npy',popt)

def plotEvaluation(preds,targets):
    plt.hist2d(preds.flatten(),targets.flatten(),bins=100)
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.title('Predictions vs Targets')
    plt.show()
    plt.savefig('figs/vanilla.png')

def evaluate_step(x,popt):
    return fitfunc(x,*popt)

def evaluate(wptList):
    popt = np.load('popts/vanilla.npy')
    targets = []
    preds= []
    for wpt in wptList:
        print('Evaluating for WPT:',wpt)
        for i in range(1,850):
            x,y = getxy(f'data/wpt_{wpt}/{i}',200)
            print(np.mean(np.abs(y-evaluate_step(x,popt))),end='\r')
            preds.append(evaluate_step(x,popt))
            targets.append(y)
    preds = np.array(preds)
    targets = np.array(targets)
    plotEvaluation(preds,targets)



if __name__ == '__main__':
    wptList = [100,150,175,200]
    fitStatistic(wptList)
    evaluate(wptList)
    