import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
from utils.model import Net
from omegaconf import OmegaConf
import datetime
from itertools import product


def logMetadata(net,config):
    currDat = datetime.datetime.now().strftime('%Y-%b-%d %H:%M:%S')
    logdir = f'runs/{currDat}'
    conflog = logdir + '/config.yaml'
    writer = SummaryWriter(log_dir=logdir)
    net.add_logger(writer)
    with open(conflog,'w') as f:
        OmegaConf.save(config,f.name)

def get_combinations(nested_dict):
    keys, values = zip(*nested_dict.items())
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations

def get_all_combinations(nested_dict):
    flat_dict = {f"{outer_key}.{inner_key}": inner_value for outer_key, inner_dict in nested_dict.items() for inner_key, inner_value in inner_dict.items()}
    combinations = get_combinations(flat_dict)
    return combinations

def main(config,device:str ='cpu'):
    net = Net(**config.modelParams)
    net.compile(optimizer = torch.optim.Adam,loss = nn.MSELoss(),device=device,particleLimit=config.trainingParams.numParticles)
    logMetadata(net,config)
    
    #TODO: Data generator instead of storing paths
    allPossiblePath = np.array([f'data/wpt_{wpt}/{i}' for wpt in config.wptList for i in range(1,850)])
    validationPaths = np.array([f'data/wpt_{wpt}/{i}' for wpt in config.wptList for i in range(850,1000)])
    
    net.fit(paths = allPossiblePath,validationPaths=validationPaths,epochs=config.trainingParams.num_epochs)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load('configs/const.yaml')
    changevars = OmegaConf.load('configs/multivar.yaml')
    combinations = get_all_combinations(changevars)
    for c in combinations:
        cflist = [ f'{key}={value}' for key, value in c.items()]
        dotlist = OmegaConf.from_dotlist(cflist)
        cfg = OmegaConf.merge(config, dotlist)
        main(config=cfg,device=device)

