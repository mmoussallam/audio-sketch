'''
speech.cortico_toy  -  Created on Nov 13, 2013

Listen to reconstruction of same phrase uttered by different speakers
at different scales 

@author: M. Moussallam
'''

import sys, os
#from classes.sketches.cochleo import SKETCH_ROOT
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
audio_outputpath = op.join(SKETCH_ROOT,'src/speech/sounds')

index = 20
spk1path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_ksp_arctic')
spk1filename = get_filepaths(spk1path, 0,  ext='wav')[index]
spk2path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
spk2filename = get_filepaths(spk2path, 0,  ext='wav')[index]

spk3filename = get_filepaths(spk2path, 0,  ext='wav')[index+50]

files  = [spk1filename, spk2filename,  spk3filename]
L = len(files)
#sidx,ridx = 1,7 
for sidx,ridx in [(0,0),]:
    plt.figure(figsize=(12,4*L))
    recs = []
    for fidx, filename in enumerate(files):
        sig = Signal(filename, mono=True)
    #    sig.downsample(16000)
        sk = CorticoSubPeaksSketch(**{'shift':0,'fac':-2,'BP':1,'n_inv_iter':10})
        sk.recompute(sig)    
        sk.sp_rep = np.zeros_like(sk.rep)
        sk.sp_rep[sidx,ridx, :,:] = sk.rep[sidx,ridx, :,:]
        sk.sp_rep[sidx,ridx+6, :,:] = sk.rep[sidx,ridx+6, :,:]
        plt.subplot(L,3,fidx*3 + 1)
        plt.imshow(np.abs(sk.sp_rep[sidx,ridx, :,:].T) + np.abs(sk.sp_rep[sidx,ridx+6, :,:].T) ,origin='lower')
        rec_aud = sk.cort.invert(sk.sp_rep, order=1)
        plt.subplot(L,3,fidx*3 + 2)
        plt.imshow(np.abs(rec_aud.T),origin='lower')
        recs.append(sk.synthesize(sparse=True))
        recs[fidx].normalize()
        plt.subplot(L,3,fidx*3 + 3)
        recs[fidx].spectrogram(log=True,cbar=False)
        recs[fidx].write(op.join(audio_outputpath,
                                 'CorticoSub_%d_%d_resynth_%d_vxf%d.wav'%(sidx,ridx,fidx,index)))
    
    plt.subplots_adjust(left=0.04,right=0.98,wspace=0.14,top=0.98)
plt.show()