import numpy as np
import pandas as pd
import glob

rootF = '/home/bdudas/PCT_DATA/output/'
wpts = [100,120,150,160,175,200]
dataFolders = [f'{rootF}/wpt_{wpt}_psa' for wpt in wpts]

def padarray(A,size = 41):
    t = size - len(A)
    if t > 0:
        return np.pad(A, (0,t), 'constant', constant_values=(0,0))
    else:
        return A
    
def main(sampleID,wpt):
    hit = pd.DataFrame(np.load(glob.glob(f"{rootF}/wpt_{wpt}_psa/*_{sampleID}.hits.npy")[0]))
    psa = pd.DataFrame(np.load(glob.glob(f"{rootF}/wpt_{wpt}_psa/*_{sampleID}_AllPSA.npy")[0], allow_pickle=True))
    hit = hit[hit.parentID == 0]
    psa = psa[psa.ParentID == 0]
    hit['Layer'] =  2*(hit['volumeID[2]'])+hit['volumeID[3]']
    hit = hit.sort_values(['eventID','Layer']).loc[:,['eventID','Layer','edep']]
    
    df_psa = psa.groupby('EventID').max('Ekine').loc[:,['Ekine']]
    df_psa.reset_index(inplace=True)
    df_psa = df_psa[df_psa.EventID.isin(hit.eventID)]

    savePath = f'data/wpt_{wpt}/{sampleID}'
    edep_arrays = [padarray(group['edep'].values) for _, group in hit.groupby('eventID')]
    np.save(f'{savePath}_x.npy', edep_arrays)
    np.save(f'{savePath}_y.npy', df_psa.Ekine.values)

if __name__ == '__main__':
    for wpt in wpts:
        for sampleID in range(1,1001):
            main(sampleID,wpt)
            print(f'{sampleID/10}% done', end='\r')
        print(f'{wpt=} done')