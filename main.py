import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
from utils.model import Net
from omegaconf import OmegaConf
import datetime

def logMetadata(net,config):
    currDat = datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
    logdir = f'runs/{currDat}'
    conflog = logdir + '/config.yaml'
    writer = SummaryWriter(log_dir=logdir)
    net.add_logger(writer)
    with open(conflog,'w') as f:
        OmegaConf.save(config,f.name)

def main(config,device:str ='cpu'):
    net = Net(**config.modelParams)
    net.compile(optimizer = torch.optim.Adam,loss = nn.MSELoss(),device=device)
    logMetadata(net,config)
    
    #TODO: Data generator instead of storing paths
    allPossiblePath = np.array([f'data/wpt_{wpt}/{i}' for wpt in config.wptList for i in range(1,850)])
    validationPaths = np.array([f'data/wpt_{wpt}/{i}' for wpt in config.wptList for i in range(850,1000)])
    
    net.fit(paths = allPossiblePath,validationPaths=validationPaths,epochs=config.trainingParams.num_epochs)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load('config.yaml')
    for hsize in config.modelParams.hidden_size:
        cfg = config.copy()
        cfg.modelParams.hidden_size = hsize
        main(config=cfg,device=device)

