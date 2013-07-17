'''
fgpt_scripts.primary_rwc  -  Created on Jun 28, 2013
@author: M. Moussallam

A first script to assess performance of Shazam's like method on a real database 
The RWC one
'''
import os
import os.path as op
from classes import pydb, sketch
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

file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
nb_files = len(file_names)
# define experimental conditions
set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered
sparsity = 300
seg_dur = 5.0
fs = 16000


# Initialize the sketchifier
#sk = sketch.STFTPeaksSketch(**{'scale':2048, 'step':512})
sk = sketch.CochleoPeaksSketch(**{'fs':fs,'step':512})

sk_id = sk.__class__.__name__[:-6]

# construct a nice name for the DB object to be saved on disk
db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                        int(seg_dur), int(fs))

# initialize the fingerprint Handler object
#fgpthandle = pydb.STFTPeaksBDB(op.join(db_path, db_name),
#                               load=True,
#                               persistent=True, **{'wall':False})
fgpthandle = pydb.CochleoPeaksBDB(op.join(db_path, db_name),
                               load=True,
                               persistent=True, **{'wall':False})
################# This is a complete experimental run given the setup ############## 
# create the base:
db_creation(fgpthandle, sk, sparsity,
            file_names, 
            force_recompute = False,
            seg_duration = seg_dur, resample = fs,
            files_path = audio_path, debug=True, n_jobs=4)



# run a fingerprinting experiment
test_proportion = 0.1 # proportion of segments in each file that will be tested


scores = db_test(fgpthandle, sk, sparsity,
                 file_names, 
                 files_path = audio_path,
                 test_seg_prop = test_proportion,
                 seg_duration = seg_dur, resample =fs,
                 step = 5.0, tolerance = 7.5, shuffle=True, debug=True)

################### End of the complete run #####################################