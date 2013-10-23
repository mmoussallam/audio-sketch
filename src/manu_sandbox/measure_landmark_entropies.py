'''
manu_sandbox.measure_landmark_entropies  -  Created on Oct 23, 2013
@author: M. Moussallam


Ok so we want to measure the entropy of the empirical distribution for the keypoints 
and the landmarks
'''

import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000

sparsity = 60
figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
max_file_num = 20
seg_dur = 4
audio_path,ext = bases[set_id]
filenames = get_filepaths(audio_path, 0,  ext=ext)[:max_file_num]
fs=8000
n_segments = 100
nb_bins = 100
def _compute_landmarks(skhandle, fgpthandle,n_segments,format=0):
    """ iterate on the file: build the landmarks, get the keys
    and build an histogram"""
    keypoints_histo = []
    landmark_histo = []
    n_processed_segments = 0
    t = time.time()
    for fileidx, filename in enumerate(filenames):
        l_sig = LongSignal(filename, frame_duration = seg_dur, Noverlap=0.5)
        tnow = time.time()
        esttime = ((tnow-t)/(n_processed_segments+1))*(n_segments-n_processed_segments)
        print "Elapsed %2.2f seconds. Estimated: %d minutes %d secs"%(tnow-t, esttime/60, esttime % 60)
        for segIdx in range(l_sig.n_seg-1):
            if n_processed_segments>=n_segments:
                break
            sub_sig = l_sig.get_sub_signal(segIdx,1, mono=True)
            sub_sig.resample(fs)
            sub_sig.pad(4096)
            skhandle.recompute(sub_sig)
            skhandle.sparsify(sparsity)
            # now get the keys
            keys, values = fgpthandle._build_pairs(skhandle.fgpt(), skhandle.params, 0)
            for key in keys:
                if format==0:
                    (f1, f2, deltat) = key
                    keypoints_histo.append(f1)
                    keypoints_histo.append(f2)
                    landmark_histo.append(f1*(fs/2) + f2)
                else:
                    (f1, deltaf, deltat) = key
                    keypoints_histo.append(f1)
                    keypoints_histo.append(f1+deltaf)
                    landmark_histo.append(f1*(fs/2) + f1+deltaf)
            n_processed_segments += 1
    return  keypoints_histo, landmark_histo

# WANG 2003
W03_fgpthandle = STFTPeaksBDB('STFTPPPairs.db',load=False,**{'wall':False,
                                                             'delta_t_max':3.0})
W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})


W03Histo_kp,W03Histo_lm = _compute_landmarks(W03_skhandle, W03_fgpthandle,n_segments)

plt.figure()
plt.subplot(211)
plt.hist(W03Histo_kp, nb_bins)
plt.subplot(212)
plt.hist(W03Histo_lm, nb_bins)



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

C10Histo_kp,C10Histo_lm = _compute_landmarks(C10_skhandle, C10_fgpthandle,n_segments,1)

plt.figure()
plt.subplot(211)
plt.hist(C10Histo_kp, nb_bins)
plt.subplot(212)
plt.hist(C10Histo_lm, nb_bins)
plt.suptitle("C10")


# Mixed
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch

for l in [5,]:
    M13_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs_%d.db'%l,load=False,**{'wall':False,
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
        W03_ref = STFTPeaksSketch(**{'scale':s,'step':s/4})
        W03_ref.recompute(Signal(np.random.randn(fs*seg_dur), fs))
        W03_ref.sparsify(sparsity)
        K = W03_ref.params['f_width']
        T = W03_ref.params['t_width']
        print "K = %d, T=%d"%(K,T)
#        biais = np.zeros((s/2,))
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
                                     'Wfs':Ws,'pad':False,'debug':0})
    M13Histo_kp,M13Histo_lm = _compute_landmarks(M13_skhandle, M13_fgpthandle,n_segments,format=1)
    plt.figure()
    plt.subplot(211)
    plt.hist(M13Histo_kp, nb_bins)
    plt.subplot(212)
    plt.hist(M13Histo_lm, nb_bins)
    plt.suptitle("M13 %d"%l)
    
plt.show()

########
nb_bins = 25
w03_kphisto, bins = np.histogram(W03Histo_kp, nb_bins, normed=True) 
C10_kphisto, bins = np.histogram(C10Histo_kp, nb_bins, normed=True) 
M13_kphisto, bins = np.histogram(M13Histo_kp, nb_bins, normed=True) 

w03_lmhisto, bins = np.histogram(W03Histo_lm, nb_bins, normed=True) 
C10_lmhisto, bins = np.histogram(C10Histo_lm, nb_bins, normed=True) 
M13_lmhisto, bins = np.histogram(M13Histo_lm, nb_bins, normed=True) 

figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')

plt.figure(figsize=(8,6))
plt.subplot(211)
plt.plot(np.log(w03_kphisto),'o-')
plt.plot(np.log(C10_kphisto),'x-')
plt.plot(np.log(M13_kphisto),'s-')
plt.ylabel('Log-probability',fontsize=16)
plt.xlabel('Keypoint index',fontsize=16)
##plt.yticks([])
plt.xticks(range(0,nb_bins,nb_bins/10),[])
plt.grid()
plt.subplot(212)
plt.plot(np.log(w03_lmhisto),'o-')
plt.plot(np.log(C10_lmhisto),'x-')
plt.plot(np.log(M13_lmhisto),'s-')
plt.legend(('W03','C10 $\lambda_H=0$','M13 $\lambda_H=5$'), loc='lower left')
plt.ylabel('Log-probability',fontsize=16)
plt.xlabel('Landmark index',fontsize=16)
#plt.yticks([])
plt.xticks(range(0,nb_bins,nb_bins/20),[])
plt.grid()
plt.subplots_adjust(left=0.10,bottom=0.07,right=0.97,top=0.96)
plt.savefig(op.join(figure_path,'EmpiricalKPLMdistribs_%s_%d_%dk_%dxLOMDCT_%dbins.pdf'%(set_id,n_segments,sparsity,len(scales),nb_bins)))
plt.show()

