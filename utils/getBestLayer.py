import numpy as np

def getBestLayer(x):
    argmxs = np.argmax(x,axis=1)
    mostPopularLayer = np.bincount(argmxs).argmax()
    return mostPopularLayer


if __name__ == '__main__':
    wptList = [100,150,175,200]
    for wpt in wptList:
        print('wpt:',wpt)
        mostPopLayers = []
        for sampleID in range(1,1000):
            x = np.load(f'data/wpt_{wpt}/{sampleID}_x.npy')
            mostPopularLayer = getBestLayer(x)
            mostPopLayers.append(mostPopularLayer)
        print(f'wpt_{wpt}: {np.bincount(mostPopLayers)}')
        np.save(f'utils/wpt_{wpt}_armgxs.npy',mostPopLayers)