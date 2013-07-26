'''
sketch_scripts.prepare_cortico_fgpt  -  Created on Jul 25, 2013
@author: M. Moussallam

# It's better to pre-compute all cochleograms and corticograms for the datasets

'''
import os
import os.path as op
import time
from scipy.io import savemat
from classes.sketches import cortico
from PyMP import signals
#from tools.fgpt_tools import db_creation, db_test
from joblib import Parallel, delayed
import numpy as np
from scipy.io import savemat
audio_path = '/sons/rwc/Learn'
save_path = '/media/manu/TOURO/corticos'

def process_seg(f,  l_sig, segIdx, resample=8000, pad=0, type='npy'):
    sk = cortico.CorticoSketch()
    
    out_name = "%s_seg_%d.%s"%(f, segIdx, type)
    if os.path.exists(op.join(save_path, out_name)):
        print "%s already computed"%out_name
        return
    
    print " starting work on %s segment %d"%(f, segIdx)
    sig_local = l_sig.get_sub_signal(segIdx, 1, mono=True, 
        normalize=True, 
        pad=pad)
    
    print sig_local.fs
    if resample>0:
        sig_local.resample(resample)
    # run the representation
    sk.recompute(sig_local)
    
#     save the output    
    print "saving to %s"%op.join(save_path, out_name)
    if type =="npy":
        np.save(op.join(save_path, out_name), sk.cort.cor)
    elif type=="mat":
        savemat(op.join(save_path, out_name),{'cor':sk.cort.cor})
    
file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
n_jobs = 4
seg_duration = 5.0
step = 5.0

for f in file_names:
        
    l_sig = signals.LongSignal(op.join(audio_path, f), frame_duration=seg_duration, 
        mono=True, 
        Noverlap=(1.0 - float(step) / float(seg_duration)))
    
    print "Loaded file %s - with %d segments of %1.1f seconds" % (f, l_sig.n_seg, 
            seg_duration)
    
    if n_jobs >1:
        # Loop on segments :  Sparsifying all of them
        Parallel(n_jobs=n_jobs)(delayed(process_seg)(f, l_sig, segIdx)                                                        
                                    for segIdx in range(l_sig.n_seg -1))
    

    