'''
fgpt_scripts.pluri_cortico_faster  -  Created on Jul 30, 2013
@author: M. Moussallam
'''


'''
fgpt_scripts.pluti_cortico_rwc  -  Created on Jul 29, 2013
@author: M. Moussallam


Let us compare the cortico scale/rate representations on the RWC recognition task
'''

import os
import os.path as op
import time
from scipy.io import savemat
from classes.pydb import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes.sketches.bench import *
from tools.fgpt_tools import db_creation, db_test

import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(0,256*1024*1024,0)
print env.get_cachesize()
# The RWC subset path
audio_path = '/sons/rwc/Learn'
db_path = '/home/manu/workspace/audio-sketch/fgpt_db'
score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'

file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
nb_files = len(file_names)
# define experimental conditions
set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered
sparsity = 300
seg_dur = 5.0
fs = 8000
downfs = fs

# Multiple comparison
sk_db_handles = [
#                 (CochleoPeaksSketch(**{'fs':fs,'step':128,'downsample':fs}), CochleoPeaksBDB),
#                 (CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs,'sub_slice':(4,11)}), CochleoPeaksBDB),
#                 (CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs,'sub_slice':(0,11)}), CochleoPeaksBDB),
#                 (CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs,'sub_slice':(4,6)}), CochleoPeaksBDB),
#                 (CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs,'sub_slice':(0,6)}), CochleoPeaksBDB),
                 (CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs,
                                           'sub_slice':(2,9)}),CochleoPeaksBDB)
#                 (STFTPeaksSketch(**{'scale':2048, 'step':512,'fs':fs}), STFTPeaksBDB),
                 ]



# Initialize the sketchifier
for sk, dbhandle in  sk_db_handles:
    sk_id = sk.__class__.__name__[:-6]
    # construct a nice name for the DB object to be saved on disk
    db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                        int(seg_dur), int(fs))

    # check for existing db 
#    load = os.path.exists(op.join(db_path, db_name))       
    load = False
    # initialize the fingerprint Handler object
    fgpthandle = dbhandle(op.join(db_path, db_name),
                                   load=load,
                                   persistent=True, **{'wall':False})

    ################# This is a complete experimental run given the setup ############## 
    # create the base:
    if not load:
        print " Computing The database"
        db_creation(fgpthandle, sk, sparsity,
                file_names, 
                force_recompute = True,
                seg_duration = seg_dur, resample = fs,
                files_path = audio_path, debug=False, n_jobs=3)


    # run a fingerprinting experiment
    test_proportion = 0.1 # proportion of segments in each file that will be tested
    
    tstart = time.time()
    scores, failures = db_test(fgpthandle, sk, sparsity,
                     file_names, 
                     files_path = audio_path,
                     test_seg_prop = test_proportion,
                     seg_duration = seg_dur, resample =fs,
                     step = 5.0, tolerance = 7.5, shuffle=True, debug=False, n_jobs=3)
    ttest = time.time() - tstart
    ################### End of the complete run #####################################
    # saving the results
    score_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs), int(100.0*test_proportion))
    
    stats =  os.stat(op.join(db_path, db_name))
    savemat(op.join(score_path,score_name), {'score':scores, 'time':ttest,
                                             'size':stats.st_size,
                                             'failures': failures})
