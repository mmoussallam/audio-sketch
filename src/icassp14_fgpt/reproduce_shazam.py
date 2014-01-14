'''
manu_sandbox.reproduce_shazam  -  Created on Oct 22, 2013
@author: M. Moussallam

find the parameters that recreate the Shazam fingerprint

'''

import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
single_test_file1 = op.join(SND_DB_PATH,'genres/blues/blues.00001.au')
#single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000
tempdir = '.'
sparsity = 20
figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms

orig_sig = Signal(single_test_file1, normalize=True, mono=True)
orig_sig.downsample(fs)
orig_sig.crop(0,5*8192)
orig_sig.pad(4096)
orig_sig.write(op.join(tempdir, 'orig.wav'))

def _Process(fgpthandle, skhandle,nb_points):
    orig_sig.spectrogram(2048,256,ax=plt.gca(),order=0.5,log=False,cbar=False,
                         cmap=cm.bone_r, extent=[0,orig_sig.get_duration(),0, fs/2])
    if skhandle.params.has_key('n_atoms'):
        skhandle.params['n_atoms'] = nb_points
    skhandle.recompute(op.join(tempdir, 'orig.wav'))
    skhandle.sparsify(nb_points)
    fgpt = skhandle.fgpt(sparse=True)
    fgpthandle.populate(fgpt, skhandle.params, 0, display=True, ax=plt.gca())
    plt.xlim([0,orig_sig.get_duration()])
    plt.ylim([0, fs/2])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('%d Landmarks'%np.count_nonzero([d for d in plt.gca().get_children() if isinstance(d, matplotlib.patches.FancyArrow)]))
    return fgpt

# WANG 2003
lambdas = [0,1,5,10]

W03_fgpthandle = STFTPeaksBDB('STFTPPPairs.db',load=False,**{'wall':False,
                                                             'TZ_delta_f':0,
                                                             'TZ_delta_t':0,
                                                             'nb_neighbors_max':10,
                                                             'delta_t_max':60.0})
W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})

plt.figure(figsize=(12,4))
plt.subplot(1,len(lambdas)+1,1)
_Process(W03_fgpthandle, W03_skhandle,sparsity)


# Mine
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch



scales = [2048]
biaises = []
Ws = []
Wt = [2*W03_skhandle.params['t_width']]
Wsw = [W03_skhandle.params['f_width']/2]

for sidx,s in enumerate(scales):    
    # ultra penalize low frequencies
#    biais = np.linspace(1.0,0.0,s/2)
    biais = np.zeros((s/2,))
#    biais = np.maximum(0.00001, biais)    
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
    for k in range(-Wsw[sidx],Wsw[sidx]):
#    for k in range(-5,5):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)  
#    Wt.append(5*(scales[-1]/s))  
#    lambdas.append(10.0)
for lidx, l in enumerate(lambdas):
    M12_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs_Lamb%d.db'%l,load=False,**{'wall':False,
                                                                      'nb_neighbors_max':3,
                                                                      'delta_f_max':1000,
                                                                      'delta_f_min':10,
                                                                      'delta_t_max':3.0})
    M12_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':sparsity,
                                     'lambdas':[l],
                                     'biaises':biaises,
                                     'Wts':Wt,
                                     'Wfs':Ws,'pad':False,'debug':1})
    plt.subplot(1,len(lambdas)+1,lidx+2)
    _Process(M12_fgpthandle, M12_skhandle,sparsity)
    
    
plt.show()