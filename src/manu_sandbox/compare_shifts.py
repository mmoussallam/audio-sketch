'''
manu_sandbox.compare_shifts  -  Created on Oct 22, 2013
@author: M. Moussallam
'''
'''
manu_sandbox.compare_robustness  -  Created on Oct 17, 2013
@author: M. Moussallam

Now we want to compare the robustness curves of the different methods
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
fs = 8000

figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/robustness')
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
seg_dur = 4;

def addshift( orig_data, shift):
    shifted = np.concatenate((np.zeros((shift,)), orig_data[:-shift]))   
    return shifted

def _process_sig(fgpthandle, skhandle, subsig, shifts, nb_points):
    
    subsig.downsample(fs)
    subsig.crop(0,seg_dur*8192)
    subsig.pad(8192)
    # Original decomposition
#    subsig.write(op.join(tempdir, 'orig.wav'))
    skhandle.recompute(subsig)
    skhandle.sparsify(nb_points)
    fgpt = skhandle.fgpt(sparse=True)    
    fgpthandle.populate(fgpt, skhandle.params, 0, 0)#, max_pairs=sparsity)
    anchor = np.sum(fgpthandle.retrieve(fgpt, skhandle.params, nbCandidates=1))
    # now make others
    orig_data = np.copy(subsig.data)
#    snrs = np.zeros((len(sigmas), ntest))
    shiftsscores = np.zeros((len(shifts), ))
    for ishift, shift in enumerate(shifts):
        shifted =  Signal(addshift(orig_data, shift), subsig.fs, normalize=False)          

        skhandle.recompute(shifted)
        skhandle.sparsify(nb_points)

        noisy_fgpt = skhandle.fgpt(sparse=True)    
        hist = fgpthandle.retrieve(noisy_fgpt, skhandle.params, nbCandidates=1)
        shiftsscores[ishift] = float(np.sum(hist))/float(anchor)
    return shiftsscores

def _Process(fgpthandle, skhandle,nb_points, shifts,  n_segments):
    n_processed_segments = 0
    noisescores = np.zeros((n_segments,len(shifts)))
    t = time.time()
    for fIdx, filename in enumerate(file_names):
        l_sig = LongSignal(filename, frame_duration = seg_dur+1)
        tnow = time.time()
        esttime = ((tnow-t)/(n_processed_segments+1))*(n_segments-n_processed_segments)
        print "Elapsed %2.2f seconds. Estimated: %d minutes %d secs"%(tnow-t, esttime/60, esttime % 60)
        for segIdx in range(l_sig.n_seg-1):
            if n_processed_segments>=n_segments:
                break
            sub_sig = l_sig.get_sub_signal(segIdx,1, mono=True)
            
            noisescores[n_processed_segments,:] = _process_sig(fgpthandle, skhandle, sub_sig, shifts, nb_points)
            
            n_processed_segments += 1
    return noisescores
###################################################################
sparsity = 30
shifts = [1,10,100]#0,5,10,20]
nsegs = 6
suffix = '%dshifts_%dsegs_%dsparsity'%(len(shifts),nsegs,sparsity)
duration = seg_dur*nsegs
plt.figure()
#################### WANG 2003
W03_fgpthandle = STFTPeaksBDB('STFTPPPairs.db',load=False,**{'wall':False,
                                                             'delta_t_max':3.0})
W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})


W03scores = _Process(W03_fgpthandle, W03_skhandle, sparsity, shifts, nsegs)
plt.plot(shifts, np.mean(W03scores, axis=0))


W03kvperseconds = float(W03_fgpthandle.get_kv_size())/float(duration)
savemat(op.join(output_path,'W03_%s.mat'%suffix),{'scores':W03scores,
                                                  'kvpersecs':int(W03kvperseconds),
                                                 'shifts':shifts})
#################### Cotton 2010
scales = [64,512,2048]
C10_fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                      'nb_neighbors_max':3,
                                                                      'delta_t_max':3.0})
C10_skhandle = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})

C10_scores = _Process(C10_fgpthandle, C10_skhandle,sparsity, shifts, nsegs)
plt.plot(shifts, np.mean(C10_scores, axis=0))

C10kvperseconds = float(C10_fgpthandle.get_kv_size())/float(duration)
savemat(op.join(output_path,'C10_%s.mat'%suffix),{'scores':C10_scores,
                                                  'kvpersecs':int(C10kvperseconds),
                                                 'shifts':shifts})
##################### Proposed #######################
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
Lambdas = [0,]
for l in Lambdas:
    M13_fgpthandle = SparseFramePairsBDB('SparseMP_PenPairs_%d.db'%l,load=False,**{'wall':False,
                                                                          'nb_neighbors_max':3,
                                                                          'delta_t_max':3.0})    
    biaises = []
    Ws = []
    Wt = [512,96,24]
    Kmax = 5
    lambdas = [l]*len(scales)
    for s in scales:    
        # ultra penalize low frequencies
        biais = np.zeros((s/2,))    
        biaises.append(biais)
        W = np.zeros((s/2,s/2))
        for k in range(-Kmax,Kmax):
            W += np.eye(s/2,s/2,k)
        Ws.append(W)    
    M13_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                                'nature':'LOMDCT',
                                     'lambdas':lambdas,
                                     'biaises':biaises,
                                     'Wts':Wt,
                                     'Wfs':Ws,'pad':False,'debug':0})
    M13_scores = _Process(M13_fgpthandle, M13_skhandle, sparsity, shifts, nsegs)
    plt.plot(shifts, np.mean(M13_scores, axis=0))    
    M13kvperseconds = float(M13_fgpthandle.get_kv_size())/float(duration)    
    savemat(op.join(output_path,'M13_%s_LO_lambda_%d_K_%d.mat'%(suffix,np.sum(lambdas),Kmax)),
            {'scores':M13_scores,
             'kvpersecs':int(M13kvperseconds),
             'lambdas':lambdas,
             'Ws':Ws,
             'Wt':Wt,
             'shifts':shifts})


plt.show()