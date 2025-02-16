import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils.modelfuncs import getSingleBatch,plotEvaluation
yModifier = 200

class Net(nn.Module):
    def __init__(self,batch_size:int = 128,input_size:int = 4,hidden_size:int = 16):
        super(Net, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.batch_size = batch_size

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

    def compile(self,optimizer,loss,particleLimit:int = 200,optmizer_params = {},device:str = 'cpu'):
        self.optimizer = optimizer(self.parameters(*optmizer_params),**optmizer_params)
        self.loss = loss
        self.particleLimit = particleLimit
        self.device = device
        self.to(device)
    

    def add_logger(self,writer):
        self.writer = writer
        self.writer.add_graph(self,torch.rand(self.batch_size,self.particleLimit,self.input_size).to(self.device))
    #####################################################################################################################
    # Training
    #####################################################################################################################

    def trainStep(self,inputs,targets,metric = lambda x: x):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss(outputs.view(-1,self.particleLimit),targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(),metric(outputs,targets)
    
    def trainlog(self,trainLoss,trainsteps,maxdiff,epoch):
        print(f'Epoch: {epoch} Loss: {trainLoss/trainsteps:.4f}')
        self.writer.add_scalar('Train/Loss', trainLoss/trainsteps, epoch)
        self.writer.add_scalar('Train/MaxDiff', maxdiff.item()*yModifier, epoch)


    def fit(self,paths,validationPaths,epochs = 100,initial_epoch = 0):
        
        np.random.shuffle(paths)
        trainsteps = len(paths)//self.batch_size

        for epoch in range(initial_epoch,initial_epoch+epochs):

            trainLoss,maxDiff = 0.0, 0.0
            availablePaths = paths.copy()

            for step in range(trainsteps):
        
                bidxs = np.random.choice(len(availablePaths),self.batch_size)
                inputX,inputY = getSingleBatch(bidxs,availablePaths)
                availablePaths = np.delete(availablePaths,bidxs)
                
                inputX = inputX.float().to(self.device)
                inputY = inputY.float().to(self.device)

                mdf_metric = lambda output,inputY:torch.max(torch.abs(output.cpu().view(-1)-inputY.cpu().view(-1)))
                loss,mdf = self.trainStep(inputX,inputY,metric=mdf_metric)
                
                trainLoss+= loss
                if mdf > maxDiff:
                    maxDiff = mdf
            
            self.trainlog(loss,trainsteps,maxDiff,epoch)
            self.validation(validationPaths,epoch)

    #####################################################################################################################
    # Validation
    #####################################################################################################################
    def validationStep(self,bidxs,valPaths):
        inputX,inputY = getSingleBatch(bidxs,valPaths,self.particleLimit)
        inputX = inputX.float().to(self.device)
        inputY = inputY.float().to(self.device)
        
        outputs = self(inputX)
        loss = self.loss(outputs.view(-1,self.particleLimit),inputY)
        
        return loss.item(),outputs.view(-1).detach().cpu(),inputY.view(-1).detach().cpu()

    def validationlog(self,valLoss,preds,targets,maxdiff,valsteps,epoch,maeLoss):
        self.writer.add_scalar('Validation/Loss', valLoss/valsteps, epoch)
        print(f'Validation Loss: {valLoss/valsteps:.4f}\n') 
        
        self.writer.add_scalar('Validation/MaxDiff', yModifier*maxdiff.item(), epoch)
        self.writer.add_scalar('Validation/MAE',maeLoss(torch.from_numpy(preds).view(-1),torch.from_numpy(targets).view(-1)).item(),epoch)
        self.writer.add_histogram('Validation/Predictions',preds,epoch)
        self.writer.add_histogram('Validation/Targets',targets,epoch)
        
        plotEvaluation(preds,targets,self.writer,epoch)
        #plotCroppedEvaluation(preds,targets,writer,epoch)
        self.writer.close()

    def validation(self,valPaths,epoch):
        maeLoss = nn.L1Loss()
        with torch.no_grad():
            
            valsteps = len(valPaths)//self.batch_size
            targets,preds = np.array([]), np.array([]) 
            valLoss,maxdiff = 0.0, 0.0 

            for step in range(valsteps):
                bidxs = np.random.choice(len(valPaths),self.batch_size)
                loss, outputs,inputY = self.validationStep(bidxs,valPaths)
                
                preds = np.concatenate((preds,outputs),axis=None)
                targets = np.concatenate((targets,inputY),axis=None)
                
                mdf = torch.max(torch.abs(outputs-inputY))
                if mdf > maxdiff:
                    maxdiff = mdf
                valLoss+= loss
            
            targets = np.array(targets,dtype=np.float32)*yModifier
            preds = np.array(preds,dtype=np.float32)*yModifier
            # Writing out metrics
            self.validationlog(loss,preds,targets,maxdiff,valsteps,epoch,maeLoss)

#if __name__ == '__main__':
#    bSize = 512
#    epochs = 1
#    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    sampleData = 'data/wpt_100/1'
#    x,y = getxy(sampleData,200)
#    plim,inpSize = x.shape
#    net = Net(batch_size=bSize,input_size=inpSize)
#    net.compile(optimizer = torch.optim.Adam,loss = nn.MSELoss())
#    writer = SummaryWriter(comment='Testing')
#    wptList = [100,110,120,130,140,150,160,170,180,190,200]
#    allPossiblePath = np.array([f'data/wpt_{wpt}/{i}' for wpt in wptList for i in range(1,850)])
#    validationPaths = np.array([f'data/wpt_{wpt}/{i}' for wpt in wptList for i in range(850,1000)])
    
    # Fine tune:
    #net.compile(optimizer = torch.optim.Adam,loss = nn.L1Loss(),optmizer_params = {'lr':1e-4})
    #net.fit(allPossiblePath,validationPaths,writer,device,epochs,epochs)