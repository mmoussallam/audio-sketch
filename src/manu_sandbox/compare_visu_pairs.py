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
rep = 'sqam'
name= 'voicefemale'
single_test_file1 = op.join(SND_DB_PATH,'%s/%s.wav'%(rep,name))
fs = 8000
tempdir = '.'
sparsity = 30
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
W03_fgpthandle = STFTPeaksBDB('STFTPPPairs.db',load=False,**{'wall':False,
                                                             'TZ_delta_f':-100,
                                                             'TZ_delta_t':5,
                                                             'delta_t_max':3.0})
W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})

plt.figure(figsize=(12,4))
plt.subplot(144)
_Process(W03_fgpthandle, W03_skhandle,sparsity)
plt.ylabel('')
plt.yticks([])
plt.title('(d)')
# Cotton 2010
scales = [64,128,256,512,1024,2048]
C10_fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':2,
                                                  'delta_f_min':250,                                                                              
                                                  'delta_f_max':2000,
                                                  'delta_t_min':0.5,                                                                              
                                                  'delta_t_max':2.0})
C10_skhandle = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})

plt.subplot(141)
_Process(C10_fgpthandle, C10_skhandle,2.5*sparsity)
plt.title('(a)')
# Proposed - 1
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
M12_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':2,
                                                  'delta_f_min':250,                                                                              
                                                  'delta_f_max':2000,
                                                  'delta_t_min':0.5,                                                                              
                                                  'delta_t_max':2.0})

biaises = []
Ws = []
Wt = [600,300,150,90,45,30,10]
lambdas = [10]*len(scales)
for s in scales:    
    # ultra penalize low frequencies
#    biais = np.linspace(1.0,0.0,s/2)
    biais = np.zeros((s/2,))
#    biais = np.maximum(0.00001, biais)    
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
    for k in range(-int(np.log2(s)),int(np.log2(s))):
#    for k in range(-5,5):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)  
#    Wt.append(5*(scales[-1]/s))  
#    lambdas.append(10.0)

M12_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                            'nature':'LOMDCT',
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'Wts':Wt,
                                 'Wfs':Ws,'pad':False,'debug':1})
plt.subplot(142)
_Process(M12_fgpthandle, M12_skhandle,2*sparsity)
[f.freq_bin for f in M12_skhandle.rep.atoms]
plt.ylabel('')
plt.yticks([])
plt.title('(b)')
# Proposed - 2
M12_fgpthandle_2 = SparseFramePairsBDB('SparseMP_PenPairs_2.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':2,
                                                  'delta_f_min':250,                                                                              
                                                  'delta_f_max':2000,
                                                  'delta_t_min':0.5,                                                                              
                                                  'delta_t_max':2.0})

biaises = []
Ws = []
Wt = []
lambdas = [10]*len(scales)
for s in scales:    
    W03_ref = STFTPeaksSketch(**{'scale':s,'step':s/4})
    W03_ref.recompute(Signal(np.random.randn(orig_sig.length), fs))
    W03_ref.sparsify(sparsity)
    K = W03_ref.params['f_width']
    T = W03_ref.params['t_width']
    print "K = %d, T=%d"%(K,T)
    # ultra penalize low frequencies
    biais = np.linspace(1,1/s,s/2)**4    
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
    for k in range(-int(K/2),int(K/2)):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)    
    Wt.append(T) 
#    Wt.append(5*(scales[-1]/s))  
#    lambdas.append(10.0)

M12_skhandle_2 = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                            'nature':'LOMDCT',
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'Wts':Wt,
                                 'Wfs':Ws,'pad':False,'debug':1})
plt.subplot(143)
_Process(M12_fgpthandle_2, M12_skhandle_2,1.5*sparsity)
[f.freq_bin for f in M12_skhandle.rep.atoms]

plt.ylabel('')
plt.yticks([])
plt.title('(c)')


plt.subplots_adjust(left=0.08,bottom=0.13,right=0.98,top=0.91, wspace=0.09)
plt.savefig(op.join(figure_path, 'KeyPoints_and_pairs_%s_%dk.pdf'%(name,sparsity)))

plt.show()

#for bI, block in enumerate(M12_skhandle.rep.dico.blocks):
#    plt.subplot(1, len(scales), bI+1)    
#    block.draw_mask()
#    plt.colorbar()
#    
#plt.show()
#
####### cprofile
#prof_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
#                                 'lambdas':lambdas,
#                                 'biaises':biaises,
#                                 'Wts':Wt,
#                                 'Wfs':Ws,'pad':False,'debug':1})
#import cProfile
#cProfile.runctx('prof_skhandle.sparsify(100)',globals(), locals())