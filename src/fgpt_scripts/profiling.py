'''
fgpt_scripts.profiling  -  Created on Jul 17, 2013
@author: M. Moussallam
'''
#import os
#import os.path as op
#import bsddb.db as db
#import cProfile
#from classes import pydb, sketch
#from classes.sketches import cochleo
#from tools.fgpt_tools import db_creation, db_test

#audio_path = '/sons/rwc/Learn'
#db_path = '/home/manu/workspace/audio-sketch/fgpt_db'
#
#file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
#nb_files = len(file_names)
## define experimental conditions
#set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered
#sparsity = 200
#seg_dur = 5.0
#fs = 16000
#
#sk = cochleo.CochleoPeaksSketch(**{'fs':fs,'step':512,'frmlen':8})
#sk_id = sk.__class__.__name__[:-6]
#
## construct a nice name for the DB object to be saved on disk
#db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
#                                        int(seg_dur), int(fs))
#
## initialize the fingerprint Handler object
##fgpthandle = pydb.STFTPeaksBDB(op.join(db_path, db_name),
##                               load=True,
##                               persistent=True, **{'wall':False})
#
## before anything happens, define the DB environment 
#env = db.DBEnv()
##env.remove("")
#env.set_cachesize(0,2*1024*1024,0)
#env.open(None, db.DB_CREATE | db.DB_INIT_MPOOL)
#
#print env.get_cachesize()
#
#fgpthandle = pydb.CochleoPeaksBDB(None,
#                               load=True,
#                               persistent=False, dbenv=env, **{'wall':False})
#
#commandStr =  'db_creation(fgpthandle, sk, sparsity, file_names[:2], force_recompute = True, seg_duration = seg_dur, resample = fs,files_path = audio_path, debug=True, n_jobs=1)'
#cProfile.runctx(commandStr, globals(), locals())
#env.remove("")
#env.close()


###############"" Profiling the db_testing
import os
import os.path as op
import time
import cProfile
from scipy.io import savemat
from classes.sketches.bench import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes import pydb
from tools.fgpt_tools import db_creation, db_test, db_test_cortico

db_path = '/home/manu/workspace/audio-sketch/fgpt_db/'
import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(10,512*1024*1024,0)
#env.remove(db_path)
env.open(db_path, db.DB_INIT_MPOOL|db.DB_CREATE )
# The RWC subset path
audio_path = '/sons/rwc/Learn'

score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'

file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
nb_files = len(file_names)
# define experimental conditions
set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered
sparsity = 10
seg_dur = 5.0
fs = 8000

## Initialize the sketchifier
#sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
sk = CorticoIndepSubPeaksSketch(**{'fs':fs,'downsample':fs,'frmlen':8,
                                   'shift':0,'fac':-2,'BP':1})
#sk = CochleoPeaksSketch(**{'fs':fs,'step':512,'downsample':fs})
sk_id = sk.__class__.__name__[:-6]

db_name = "%s_%s_k%d_%s_%dsec_%dfs/"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs))        
    # initialize the fingerprint Handler object
fgpthandle = pydb.CorticoIndepSubPeaksBDB(op.join(db_path, db_name),
                                              load=True, persistent=True, dbenv=env,
                                              rd_only=False,
                                               **{'wall':False,'max_pairs':500})


commandStr = "db_test_cortico(fgpthandle, sk, sparsity,\
                         file_names[1:2], \
                         files_path = audio_path,\
                         test_seg_prop = 0.1,\
                         seg_duration = seg_dur, resample =fs,\
                         step = 5.0, tolerance = 7.5, shuffle=True, debug=False,n_jobs=1,\
                         n_files=84)"
                         
cProfile.runctx(commandStr, globals(), locals())                         