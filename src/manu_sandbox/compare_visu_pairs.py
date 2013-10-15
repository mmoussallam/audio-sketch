'''
manu_sandbox.compare_visu_pairs  -  Created on Oct 15, 2013
@author: M. Moussallam

On a simple audio example: compare the constructed Landmarks with different methods:
 - Shazam : Peak Picking with monoscale Gabor
 - Cotton : MP with multiscale Gabor
 - Ours   : MP with Information Theoretic penalty  
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000
tempdir = '.'
sparsity = 20
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms

orig_sig = Signal(single_test_file1, normalize=True, mono=True)
orig_sig.downsample(fs)
orig_sig.pad(4096)
orig_sig.write(op.join(tempdir, 'orig.wav'))

def _Process(fgpthandle, skhandle):
    orig_sig.spectrogram(2048,256,ax=plt.gca(),order=1,log=False,cbar=False,
                         cmap=cm.bone_r, extent=[0,orig_sig.get_duration(),0, fs/2])
    skhandle.recompute(op.join(tempdir, 'orig.wav'))
    skhandle.sparsify(sparsity)
    fgpt = skhandle.fgpt(sparse=True)
    fgpthandle.populate(fgpt, skhandle.params, 0, display=True, ax=plt.gca())
    plt.xlim([0,orig_sig.get_duration()])
    plt.ylim([0, fs/2])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    return fgpt

# WANG 2003
W03_fgpthandle = STFTPeaksBDB('STFTPPPairs.db',load=False,**{'wall':False,
                                                             'delta_t_max':3.0})
W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})

plt.figure()
plt.subplot(131)
_Process(W03_fgpthandle, W03_skhandle)

# Cotton 2010
C10_fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':3,
                                                                      'delta_t_max':3.0})
C10_skhandle = XMDCTSparsePairsSketch(**{'scales':[64,512, 4096],'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})

plt.subplot(132)
_Process(C10_fgpthandle, C10_skhandle)

# Proposed
from sketch_objects import XMDCTPenalizedPairsSketch
M12_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':3,
                                                                      'delta_t_max':3.0})
M12_skhandle = XMDCTPenalizedPairsSketch(**{'scales':[64,512, 4096],'n_atoms':1,
                                 'lambdas':[10,10,10],'pad':False,'debug':1})
plt.subplot(133)
_Process(M12_fgpthandle, M12_skhandle)
plt.show()


