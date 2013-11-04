'''
manu_sandbox.visu_pairs_gtzan  -  Created on Oct 23, 2013
@author: M. Moussallam
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
sparsity = 64
figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
max_file_num = 20
seg_dur = 4
audio_path,ext = bases[set_id]
filenames = get_filepaths(audio_path, 0,  ext=ext)[:max_file_num]
fs=8000
n_segments = 5


def _Process(fgpthandle, skhandle,nb_points,subsig):

    subsig.spectrogram(2048,256,ax=plt.gca(),order=0.5,log=False,cbar=False,
                         cmap=cm.bone_r, extent=[0,subsig.get_duration(),0, fs/2])

    skhandle.recompute(subsig)
    skhandle.sparsify(nb_points)
    fgpt = skhandle.fgpt(sparse=True)
    fgpthandle.populate(fgpt, skhandle.params, 0, display=True, ax=plt.gca())
    plt.xlim([0,subsig.get_duration()])
    plt.ylim([0, fs/2])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('%d Landmarks'%np.count_nonzero([d for d in plt.gca().get_children() if isinstance(d, matplotlib.patches.FancyArrow)]))
    return fgpt

# compare the landmarks on real data
W03_fgpthandle = STFTPeaksBDB('STFTPPPairs.db',load=False,**{'wall':False,
                                                             'delta_t_max':3.0})
W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})


scales = [64,128,256,512,1024,2048]
#scales = [1024]
C10_fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':2,
                                                                      'delta_f_min':250,
                                                                      'delta_f_max':2000,
                                                                      'delta_t_min':0.5, 
                                                                      'delta_t_max':2.0})
C10_skhandle = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})
l=5
M13_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs_1.db',load=False,**{'wall':False,
                                                                              'nb_neighbors_max':2,
                                                                              'delta_f_min':250,                                                                              
                                                                              'delta_f_max':2000,
                                                                              'delta_t_min':0.5,                                                                              
                                                                              'delta_t_max':2.0})    
biaises = []
Ws = []
Wt = []
lambdas = [l]*len(scales)

for s in scales:    
    # ultra penalize low frequencies
    W03_ref = STFTPeaksSketch(**{'scale':s,'step':s/4})
    W03_ref.recompute(Signal(np.random.randn(fs*seg_dur), fs))
    W03_ref.sparsify(sparsity)
    K = W03_ref.params['f_width']
    T = W03_ref.params['t_width']
    print "K = %d, T=%d"%(K,T)
#    biais = np.zeros((s/2,))    
    biais = np.linspace(1,1/s,s/2)**2
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
    for k in range(-int(K/2),int(K/2)):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)    
    Wt.append(T)
    

M13_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'nature':'LOMDCT',
                                 'Wts':Wt,
                                 'Wfs':Ws,'pad':False,'debug':0,'entropic':True})
t=time.time()
n_processed_segments=0
for fileidx, filename in enumerate(filenames):
    l_sig = LongSignal(filename, frame_duration = seg_dur, Noverlap=0)
    tnow = time.time()
    esttime = ((tnow-t)/(n_processed_segments+1))*(n_segments-n_processed_segments)
    print "Elapsed %2.2f seconds. Estimated: %d minutes %d secs"%(tnow-t, esttime/60, esttime % 60)
    for segIdx in range(l_sig.n_seg-1):
        if n_processed_segments>=n_segments:
            break
        sub_sig = l_sig.get_sub_signal(segIdx,1, mono=True)
        sub_sig.resample(fs)
        sub_sig.pad(4096)
        plt.figure()
        plt.subplot(121)
        _Process(C10_fgpthandle, C10_skhandle, sparsity, sub_sig)
        plt.subplot(122)
        _Process(M13_fgpthandle, M13_skhandle, sparsity, sub_sig)
        n_processed_segments+=1
        
plt.show()