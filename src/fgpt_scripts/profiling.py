'''
fgpt_scripts.profiling  -  Created on Jul 17, 2013
@author: M. Moussallam
'''
import os
import os.path as op
import bsddb.db as db
import cProfile
from classes import pydb, sketch
from classes.sketches import cochleo
from tools.fgpt_tools import db_creation, db_test

audio_path = '/sons/rwc/Learn'
db_path = '/home/manu/workspace/audio-sketch/fgpt_db'

file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
nb_files = len(file_names)
# define experimental conditions
set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered
sparsity = 200
seg_dur = 5.0
fs = 16000

sk = cochleo.CochleoPeaksSketch(**{'fs':fs,'step':512})
sk_id = sk.__class__.__name__[:-6]

# construct a nice name for the DB object to be saved on disk
db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                        int(seg_dur), int(fs))

# initialize the fingerprint Handler object
#fgpthandle = pydb.STFTPeaksBDB(op.join(db_path, db_name),
#                               load=True,
#                               persistent=True, **{'wall':False})

# before anything happens, define the DB environment 
env = db.DBEnv()
#env.remove("")
env.set_cachesize(0,2*1024*1024,0)
env.open(None, db.DB_CREATE | db.DB_INIT_MPOOL)

print env.get_cachesize()

fgpthandle = pydb.CochleoPeaksBDB(None,
                               load=True,
                               persistent=False, dbenv=env, **{'wall':False})

commandStr =  'db_creation(fgpthandle, sk, sparsity, file_names[:2], force_recompute = True, seg_duration = seg_dur, resample = fs,files_path = audio_path, debug=True, n_jobs=3)'
cProfile.runctx(commandStr, globals(), locals())
env.remove("")
env.close()