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
sparsity = 30
figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms

orig_sig = Signal(single_test_file1, normalize=True, mono=True)
orig_sig.downsample(fs)
orig_sig.pad(4096)
orig_sig.write(op.join(tempdir, 'orig.wav'))

def _Process(fgpthandle, skhandle):
    orig_sig.spectrogram(2048,256,ax=plt.gca(),order=0.5,log=False,cbar=False,
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

plt.figure(figsize=(12,4))
plt.subplot(131)
_Process(W03_fgpthandle, W03_skhandle)

# Cotton 2010
scales = [2048,]
C10_fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':3,
                                                                      'delta_t_max':3.0})
C10_skhandle = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})

plt.subplot(132)
_Process(C10_fgpthandle, C10_skhandle)
plt.ylabel('')
plt.yticks([])
# Proposed
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
M12_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':3,
                                                                      'delta_t_max':3.0})

biaises = []
Ws = []
lambdas = []
for s in scales:    
    # ultra penalize low frequencies
#    biais = np.linspace(1.0,0.0,s/2)
    biais = np.zeros((s/2,))
#    biais = np.maximum(0.00001, biais)    
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
#    for k in range(-int(2*np.log2(s)),int(2*np.log2(s))):
    for k in range(-s/32,s/32):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)
    lambdas.append(3.0)

M12_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'Ws':Ws,'pad':False,'debug':1})
plt.subplot(133)
_Process(M12_fgpthandle, M12_skhandle)
plt.ylabel('')
plt.yticks([])
plt.subplots_adjust(left=0.08,bottom=0.13,right=0.98,top=0.95, wspace=0.09)
plt.savefig(op.join(figure_path, 'KeyPoints_and_pairs.pdf'))
plt.show()


