'''
manu_sandbox.test_sketches_coherence  -  Created on Oct 22, 2013
@author: M. Moussallam
'''
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
import matplotlib
single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000
tempdir = '.'
sparsity = 35
figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch

orig_sig = Signal(single_test_file1, normalize=True, mono=True)
orig_sig.downsample(fs)
orig_sig.crop(0,5*4096)
orig_sig.pad(4096)
orig_sig.write(op.join(tempdir, 'orig.wav'))

#def _Process(fgpthandle, skhandle,nb_points):
#    orig_sig.spectrogram(2048,256,ax=plt.gca(),order=0.5,log=False,cbar=False,
#                         cmap=cm.bone_r, extent=[0,orig_sig.get_duration(),0, fs/2])
#    if skhandle.params.has_key('n_atoms'):
#        skhandle.params['n_atoms'] = nb_points
#    skhandle.recompute(op.join(tempdir, 'orig.wav'))
#    skhandle.sparsify(nb_points)
#    fgpt = skhandle.fgpt(sparse=True)
#    fgpthandle.populate(fgpt, skhandle.params, 0, display=True, ax=plt.gca())
#    plt.xlim([0,orig_sig.get_duration()])
#    plt.ylim([0, fs/2])
#    plt.xlabel('Time (s)')
#    plt.ylabel('Frequency (Hz)')
#    plt.title('%d Landmarks'%np.count_nonzero([d for d in plt.gca().get_children() if isinstance(d, matplotlib.patches.FancyArrow)]))
#    return fgpt


# Cotton 2010
scales = [64,512,2048]
C10_fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':5,
                                                                      'delta_t_max':3.0})
C10_skhandle = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})



# Proposed

M12_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':5,
#                                                                      'delta_f_min':10,
#                                                                      'delta_f_max':667,
                                                                      'delta_t_max':3.0})

biaises = []
Ws = []
Wt = [600, 90, 20]
lambdas = [0,0,0]
for s in scales:    
    # ultra penalize low frequencies
    biais = np.zeros((s/2,)) 
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
    for k in range(-int(np.log2(s)),int(np.log2(s))):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)  


M12_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                            'nature':'LOMDCT',
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'Wts':Wt,
                                 'Wfs':Ws,'pad':False,'debug':1})

C10_skhandle.recompute(orig_sig)
C10_skhandle.sparsify(sparsity)

M12_skhandle.recompute(orig_sig)
M12_skhandle.sparsify(sparsity)

for atomIdx in range(C10_skhandle.rep.atom_number):
    assert(C10_skhandle.rep[atomIdx] == M12_skhandle.rep[atomIdx])

