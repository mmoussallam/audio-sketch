'''
fgpt_scripts.primary_rwc  -  Created on Jun 28, 2013
@author: M. Moussallam

A first script to assess performance of Shazam's like method on a real database 
The RWC one
'''
import os
import os.path as op
import time
from scipy.io import savemat
from classes import pydb
from classes.sketches.cortico import *
from tools.fgpt_tools import db_creation, db_test

# define a pair FgptHandle/Sketch 
#fgpt_sketches = [
#                 (pydb.STFTPeaksBDB('STFTPeaks.db', **{'wall':False}),
#                  sketch.STFTPeaksSketch(**{'scale':2048, 'step':512})), 
#                 (pydb.XMDCTBDB('xMdct.db', **{'wall':False}),
#                  sketch.XMDCTSparseSketch(**{'scales':[64,512,2048],'n_atoms':100})),                                     
#                    ]


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


# Initialize the sketchifier
#sk = sketch.STFTPeaksSketch(**{'scale':2048, 'step':512})
#sk = sketch.CochleoPeaksSketch(**{'fs':fs,'step':512})
sk = CorticoIndepSubPeaksSketch(**{'fs':fs,'downsample':fs,'frmlen':8,'shift':0,'fac':-2,'BP':1})

sk_id = sk.__class__.__name__[:-6]

# construct a nice name for the DB object to be saved on disk
db_name = "%s_%s_k%d_%s_%dsec_%dfs"%(set_id, sk_id, sparsity, sk.get_sig(),
                                        int(seg_dur), int(fs))
#db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
#                                        int(seg_dur), int(fs))


# initialize the fingerprint Handler object
#fgpthandle = pydb.STFTPeaksBDB(op.join(db_path, db_name),
#                               load=True,
#                               persistent=True, **{'wall':False})
#fgpthandle = pydb.CochleoPeaksBDB(op.join(db_path, db_name),
#                               load=True,
#                               persistent=True, **{'wall':False})
fgpthandle = pydb.CorticoIndepSubPeaksBDB(op.join(db_path, db_name),
                                           load=True, persistent=True ,**{'wall':True})
################# This is a complete experimental run given the setup ############## 
# create the base:
db_creation(fgpthandle, sk, sparsity,
            file_names, 
            force_recompute = True,
            seg_duration = seg_dur, resample = fs,
            files_path = audio_path, debug=True, n_jobs=4)


# run a fingerprinting experiment
test_proportion = 0.1 # proportion of segments in each file that will be tested

tstart = time.time()
scores = db_test(fgpthandle, sk, sparsity,
                 file_names, 
                 files_path = audio_path,
                 test_seg_prop = test_proportion,
                 seg_duration = seg_dur, resample =fs,
                 step = 5.0, tolerance = 7.5, shuffle=True, debug=False)
ttest = time.time() - tstart
################### End of the complete run #####################################
# saving the results
score_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                        int(seg_dur), int(fs), int(100.0*test_proportion))

stats =  os.stat(op.join(db_path, db_name))
savemat(op.join(score_path,score_name), {'score':scores, 'time':ttest, 'size':stats.st_size})

