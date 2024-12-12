import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.filter import layer_filter, getPeakData,getFitData
from torch.utils.tensorboard import SummaryWriter
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

class Net(nn.Module):
    def __init__(self,batch_size:int = 128,input_size:int = 4):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.batch_size = batch_size

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

    def compile(self,optimizer,loss,optmizer_params = {}):
        self.optimizer = optimizer(self.parameters(*optmizer_params))
        self.loss = loss
    
    def fit(self,paths,validationPaths,writer,device,epochs = 100,initial_epoch = 0):
        particleLimit = 200
        np.random.shuffle(paths)
        trainsteps = len(paths)//self.batch_size
        for epoch in range(initial_epoch,initial_epoch+epochs):
            trainLoss = 0.0
            availablePaths = paths.copy()
            maxDiff = 0.0
            for step in range(trainsteps):
                progression = step*100/trainsteps
                #print(f'Progression: {progression:.3f} %',end='\r')
                bidxs = np.random.choice(len(availablePaths),self.batch_size)
                inputX,inputY = getSingleBatch(bidxs,availablePaths,particleLimit)
                availablePaths = np.delete(availablePaths,bidxs)
                inputX = inputX.float().to(device)
                inputY = inputY.float().to(device)
                self.optimizer.zero_grad()
                outputs = self(inputX)
                loss = self.loss(outputs.view(-1),inputY.view(-1))
                trainLoss += loss.item()
                loss.backward()
                self.optimizer.step()
                outputs = outputs.cpu().view(-1)
                inputY = inputY.cpu().view(-1)
                mdf = torch.max(torch.abs(outputs-inputY))
                if mdf > maxDiff:
                    maxDiff = mdf
            print(f'Epoch: {epoch} Loss: {trainLoss/trainsteps:.4f}')
            writer.add_scalar('Train/Loss', trainLoss/trainsteps, epoch)
            writer.add_scalar('Train/MaxDiff', maxDiff.item()*yModifier, epoch)
            self.validation(validationPaths,writer,epoch)
    
    def validation(self,valPaths,writer,epoch):
        maeLoss = nn.L1Loss()
        with torch.no_grad():
            particleLimit = 200
            valsteps = len(valPaths)//self.batch_size
            targets = []
            preds= []
            valLoss = 0.0
            maxdiff = 0.0
            for step in range(valsteps):
                progression = step*100/valsteps
                #print(f'Progression: {progression:.3f} %',end='\r')
                bidxs = np.random.choice(len(valPaths),self.batch_size)
                inputX,inputY = getSingleBatch(bidxs,valPaths,particleLimit)
                inputX = inputX.float().to(device)
                inputY = inputY.float().to(device)
                outputs = self(inputX)
                preds.append(outputs.detach().cpu().numpy())
                targets.append(inputY.detach().cpu().numpy())
                outputs = outputs.view(-1).cpu()
                inputY = inputY.view(-1).cpu()
                loss = self.loss(outputs,inputY)
                mdf = torch.max(torch.abs(outputs-inputY))
                if mdf > maxdiff:
                    maxdiff = mdf
                valLoss+= loss.item()
            targets = np.array(targets)*yModifier
            preds = np.array(preds)*yModifier
            writer.add_scalar('Validation/Loss', valLoss/valsteps, epoch)
            print(f'Validation Loss: {valLoss/valsteps:.4f}\n') 
            writer.add_scalar('Validation/MaxDiff', yModifier*maxdiff.item(), epoch)
            writer.add_scalar('Validation/MAE',maeLoss(torch.from_numpy(preds).view(-1),torch.from_numpy(targets).view(-1)).item(),epoch)
            writer.add_histogram('Validation/Predictions',preds,epoch)
            writer.add_histogram('Validation/Targets',targets,epoch)
            plotEvaluation(preds,targets,writer,epoch)
            plotCroppedEvaluation(preds,targets,writer,epoch)
            writer.close()

if __name__ == '__main__':
    bSize = 512
    epochs = 500
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    sampleData = 'data/wpt_100/1'
    x,y = getxy(sampleData,200)
    plim,inpSize = x.shape
    net = Net(batch_size=bSize,input_size=inpSize)
    net.compile(optimizer = torch.optim.Adam,loss = nn.MSELoss())
    writer = SummaryWriter(comment='Best learning with more steps')
    wptList = [100,110,120,130,140,150,160,170,180,190,200]
    allPossiblePath = np.array([f'data/wpt_{wpt}/{i}' for wpt in wptList for i in range(1,850)])
    validationPaths = np.array([f'data/wpt_{wpt}/{i}' for wpt in wptList for i in range(850,1000)])
    writer.add_graph(net,torch.rand(bSize,plim,inpSize))
    net = net.to(device)
    net.fit(allPossiblePath,validationPaths,writer,device,epochs)

    # Fine tune:
    #net.compile(optimizer = torch.optim.Adam,loss = nn.L1Loss(),optmizer_params = {'lr':1e-4})
    #net.fit(allPossiblePath,validationPaths,writer,device,epochs,epochs)