'''
fgpt_scripts.many_sparsities  -  Created on Jul 30, 2013
@author: M. Moussallam

Now let us see how performances evolve with the sparsity
'''
import os
import os.path as op
import time
from scipy.io import savemat
from classes.sketches.bench import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes import pydb
from tools.fgpt_tools import db_creation, db_test
from tools.fgpt_tools import get_filepaths
db_path = '/home/manu/workspace/audio-sketch/fgpt_db/'
import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(0,512*1024*1024,0)
env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
env.log_set_config(db.DB_LOG_IN_MEMORY, 1)
env.open(None, env_flags)

bases = {'RWCLearn':('/home/manu/workspace/databases/rwc/Learn/','.wav'),
         'voxforge':('/sons/voxforge/main/Learn/','.wav'),
         'GTZAN':('/home/manu/workspace/databases/genres/','.au')}

# bases = {'RWCLearn':('/sons/rwc/Learn/','.wav'),
#          'voxforge':('/sons/voxforge/main/Learn/','.wav'),
#          'GTZAN':('/home/manu/workspace/databases/genres/','.au')}

# The RWC subset path
#audio_path = '/sons/rwc/Learn'
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered

audio_path,ext = bases[set_id]
score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'

file_names = get_filepaths(audio_path, 0,  ext=ext)

nb_files = len(file_names)
# define experimental conditions

sparsities = [100,]
seg_dur = 5
fs = 11025
step = 3.0
## Initialize the sketchifier

# sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
sk = XMDCTSparseSketch(**{'scales':[1024, 8192,],'n_atoms':5,'nature':'LOMDCT',
                          'penalty':0.3,'mask_size':1})
sk_id = sk.__class__.__name__[:-6]
 
learn = True
test = True
subset = 100
for sparsity in sparsities:    
    # construct a nice name for the DB object to be saved on disk
    db_name = "%s_%s_k%d_%s_%dsec_%dfs_subset%d.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs),subset)
        
    # initialize the fingerprint Handler object
    fgpthandle = pydb.XMDCTBDB(op.join(db_path, db_name),
                                   load=not learn,
                                   persistent=True, **{'wall':False})

    ################# This is a complete experimental run given the setup ############## 
    # create the base:
    if learn:
        db_creation(fgpthandle, sk, sparsity,
                file_names[:subset], 
                force_recompute = True,
                seg_duration = seg_dur, resample = fs,
                files_path = audio_path, debug=False, n_jobs=1)
    
    
    # run a fingerprinting experiment
    test_proportion = 0.25 # proportion of segments in each file that will be tested
#    print fgpthandle.dbObj.stat_print()
    if test:
        tstart = time.time()
        scores, failures = db_test(fgpthandle, sk, sparsity,
                         file_names[:subset], 
                         files_path = audio_path,
                         test_seg_prop = test_proportion,
                         seg_duration = seg_dur, resample =fs,
                         step = step, tolerance = 7.5, shuffle=True, debug=False,n_jobs=1)
        ttest = time.time() - tstart
        ################### End of the complete run #####################################
        # saving the results
        score_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d_step%d_subset%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                                int(seg_dur), int(fs), int(100.0*test_proportion),int(step),subset)
        
        del fgpthandle # useful to make sure the dump to disk has been done
        stats =  os.stat(op.join(db_path, db_name))
        savemat(op.join(score_path,score_name), {'score':scores, 'time':ttest,
                                                 'size':stats.st_size,'failures': failures})
        
