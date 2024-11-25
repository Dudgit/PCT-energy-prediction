import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.filter import layer_filter, getPeakData,getFitData
from torch.utils.tensorboard import SummaryWriter

yModifier = 190

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
    x = getFitData(x)#x = getPeakData(x,targetLayer)
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
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.title('Predictions vs Targets')
    writer.add_figure('Validation/Predictions vs Targets',fig,epoch)

class Net(nn.Module):
    def __init__(self,batch_size:int = 128):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 1)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        return x
    def compile(self,optimizer,loss):
        self.optimizer = optimizer(self.parameters())
        self.loss = loss
    
    def fit(self,paths,validationPaths):
        particleLimit = 200
        writer = SummaryWriter()
        np.random.shuffle(paths)
        trainsteps = len(paths)//self.batch_size
        for epoch in range(100):
            trainLoss = 0.0
            availablePaths = paths.copy()
            for step in range(trainsteps):
                progression = step*100/trainsteps
                print(f'Progression: {progression:.3f} %',end='\r')
                bidxs = np.random.choice(len(availablePaths),self.batch_size)
                inputX,inputY = getSingleBatch(bidxs,availablePaths,particleLimit)
                availablePaths = np.delete(availablePaths,bidxs)
                inputX = inputX.float()
                inputY = inputY.float()
                self.optimizer.zero_grad()
                outputs = self(inputX)
                loss = self.loss(outputs.view(-1),inputY.view(-1))
                trainLoss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch: {epoch} Loss: {trainLoss/trainsteps:.4f}')
            writer.add_scalar('Train/Loss', trainLoss/trainsteps, epoch)
            self.validation(validationPaths,writer,epoch)
    
    def validation(self,valPaths,writer,epoch):
        with torch.no_grad():
            particleLimit = 200
            valsteps = len(valPaths)//self.batch_size
            targets = []
            preds= []
            valLoss = 0.0
            for step in range(valsteps):
                progression = step*100/valsteps
                print(f'Progression: {progression:.3f} %',end='\r')
                bidxs = np.random.choice(len(valPaths),self.batch_size)
                inputX,inputY = getSingleBatch(bidxs,valPaths,particleLimit)
                inputX = inputX.float()
                inputY = inputY.float()
                outputs = self(inputX)
                preds.append(outputs.detach().numpy())
                targets.append(inputY.detach().numpy())
                loss = self.loss(outputs.view(-1),inputY.view(-1))
                valLoss+= loss.item()
            writer.add_scalar('Validation/Loss', valLoss/valsteps, epoch)
            print(f'Validation Loss: {valLoss/valsteps:.4f}\n') 
            plotEvaluation(np.array(preds),np.array(targets),writer,epoch)
            writer.close()

if __name__ == '__main__':
    net = Net()
    net.compile(optimizer = torch.optim.Adam,loss = nn.MSELoss())
    #TODO: Create more WPT-> 110,120,130,140,160,170,180,190
    wptList = [100,150,175,200]
    allPossiblePath = np.array([f'data/wpt_{wpt}/{i}' for wpt in wptList for i in range(1,850)])
    validationPaths = np.array([f'data/wpt_{wpt}/{i}' for wpt in wptList for i in range(850,1000)])
    net.fit(allPossiblePath,validationPaths)