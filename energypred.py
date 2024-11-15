import argparse, os, glob

# Specify GPU-s to use.
parser = argparse.ArgumentParser(description='Modell training, you can specify the configurations you would like to use')
parser.add_argument('--gpuID', help='GPU ID-s you can use for training',default=0)
parser.add_argument('-g', help='GPU ID-s you can use for training',default=0)
parser.add_argument("--comment",type=str,default="",help="Comment for the run")



args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
GPUID = args.gpuID if args.g is None else args.g
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUID)
# Kill logging, because it is annoying.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
batch_size = 32
dims_to_use = 3

dataRoot = '/home/bdudas/PCT/data'


def sigmaFilter(xdata,ydata,SigmaMask:float = 1.):
    argmaxes = np.argmax(xdata,axis=1)
    means = tf.reduce_mean(xdata,axis=0)
    stds = tf.math.reduce_std(xdata,axis=0)
    auni = np.unique(argmaxes)
    x_out,y_out = [],[]
    for au in auni:
        xin = xdata[argmaxes == au]
        yin = ydata[argmaxes == au]
        condition = (tf.abs(tf.gather(xin,au,axis = 1)-tf.gather(means,au))/tf.gather(stds,au) < SigmaMask)
        x = xin[condition]
        y = yin[condition]
        x_out.append(x)
        y_out.append(y)
    xarr = x_out[0]
    yarr = y_out[0]
    for x,y in zip(x_out[1:],y_out[1:]):
        xarr = tf.concat([xarr,x],axis=0)
        yarr = tf.concat([yarr,y],axis=0)
    return xarr,yarr

def peakFilter(xdata,ydata,wpt,peakMask:int = 1):
    agmxs = np.load(f'{dataRoot}/argmxs/wpt_{wpt}.npy')
    condition = ( np.argmax(xdata,axis=1) == agmxs) | (np.argmax(xdata,axis=1) == agmxs+1) | (np.argmax(xdata,axis=1) == agmxs-1)
    x = xdata[condition]
    y = ydata[condition]
    return x,y


def newX(x):
    apos = tf.argmax(x,axis=1)
    apos = tf.reshape(apos,(-1,1))
    tmp =  tf.gather(x,apos,axis=1,batch_dims=1)
    tmpprev = tf.gather(x,apos-1,axis=1,batch_dims=1)
    tmpnext = tf.gather(x,apos+1,axis=1,batch_dims=1)
    #res = tf.concat([tf.cast(apos/42,dtype= tf.float32),tmp],axis = 1)
    res = tf.concat([tf.cast(apos,dtype= tf.float32),tmpprev/tmp,tmpnext/tmp],axis = 1)
    return res #tf.cast(apos,dtype=tf.float32)/42


def get_xy2(p:str,sigma:float = 1.):
    tmp = tf.io.read_file(f'{dataRoot}/enertf/'+p+'.tfrecord')
    wpt = int(p.split('_')[1])
    x = tf.io.parse_tensor(tmp,tf.float32)
    x = tf.gather(x,2,axis=2)
    y = np.load(f'{dataRoot}/enerpred/'+p+'_y.npy')[0:200]
    x,y = peakFilter(x,y,wpt)
    x,y = sigmaFilter(x,y,SigmaMask=sigma)
    x = newX(x)
    return x.numpy(),y.numpy()

def fitfunc(x,a,b,c):
    return a*x[:,0]+b*x[:,1]+c*x[:,2]

def fitData():
    plt.style.use('ggplot')
    wptlist = [100,120,150,160,175,200]
    x,y = get_xy2('wpt_100_1')
    print('starting')
    for wpt in wptlist:
        for i in range(1,800):
            x_tm,y_tmp = get_xy2(f'wpt_{wpt}_{i}')
            x = np.concatenate([x,x_tm],axis=0)
            y = np.concatenate([y,y_tmp],axis=0)
    y = y.reshape(-1)
    popt, _ = curve_fit(fitfunc,x,y)
    x_test,y_test = get_xy2('wpt_100_1')
    print('Train finished')
    for wpt in wptlist:
        for i in range(850,1000):
            x_tm,y_tmp = get_xy2(f'wpt_{wpt}_{i}')
            x_test = np.concatenate([x_test,x_tm],axis=0)
            y_test = np.concatenate([y_test,y_tmp],axis=0)
    y_test = y_test.reshape(-1)
    y_pred = fitfunc(x_test,*popt)
    np.save('popts/popt2.npy',popt)
    return y_test,y_pred

def plotRes(y_test,y_pred,extra=''):
    plt.scatter(y_test,y_pred)
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('Energy Prediction')
    plt.show()
    plt.savefig('figs/energyPred'+extra+'.png')

# 
if __name__ == '__main__':
    y_test,y_pred = fitData()
    plotRes(y_test,y_pred)

