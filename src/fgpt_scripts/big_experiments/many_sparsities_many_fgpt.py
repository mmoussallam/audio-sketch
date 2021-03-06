'''
fgpt_scripts.big_experiments.many_sparsities_many_fgpt  -  Created on Sep 17, 2013
@author: M. Moussallam
'''

import os
import os.path as op
import time
from scipy.io import savemat
import sys
#sys.path.append('../../..')
import os
from os import chdir
chdir('/Users/loa-guest/Documents/Laure/audio-sketch')

from src.classes.sketches.base import *
from src.classes.sketches.bench import *
from src.classes.sketches.cortico import *
from src.classes.sketches.cochleo import *


from src.classes.fingerprints import *
from src.classes.fingerprints.bench import *
from src.classes.fingerprints.cortico import *
from src.classes.fingerprints.cochleo import *
from src.classes.fingerprints.CQT import *
from src.tools.fgpt_tools import db_creation, db_test
from src.tools.fgpt_tools import get_filepaths


SKETCH_ROOT = os.environ['SKETCH_ROOT']
db_path = op.join(SKETCH_ROOT,'fgpt_db')
score_path = op.join(SKETCH_ROOT,'fgpt_scores')

SND_DB_PATH = os.environ['SND_DB_PATH']

import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(0,512*1024*1024,0)
env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
env.log_set_config(db.DB_LOG_IN_MEMORY, 1)
env.open(None, env_flags)

bases = {'RWCLearn':(op.join(SND_DB_PATH,'rwc/Learn/'),'.wav'),
         'voxforge':(op.join(SND_DB_PATH,'voxforge/main/'),'wav'),
         #'voxforge':(op.join(SND_DB_PATH,'voxforge/main/Learn/'),'wav'),
         'GTZAN':(op.join(SND_DB_PATH,'genres/'),'.au')}

# The RWC subset path
#audio_path = '/sons/rwc/Learn'
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]


file_names = get_filepaths(audio_path, 0,  ext=ext)

nb_files = len(file_names)
# define experimental conditions

sparsities = [5,10,30,50]
seg_dur = 5
fs = 8000
step = 3.0
learn = True
test = True

## Initialize the sketchifier
setups = [
#          ((SparseFramePairsBDB,{'wall':False,'nb_neighbors_max':3,'delta_t_max':3.0}),1,
#      XMDCTSparsePairsSketch(**{'scales':[64,512,4096],'n_atoms':1,
#                                 'nature':'LOMDCT'})),
#          ((XMDCTBDB,{'wall':False}),
#           1,
#           XMDCTSparseSketch(**{'scales':[2048, 4096, 8192],'n_atoms':1,
#                                                  'nature':'LOMDCT'})),     
#                     (SWSBDB(None, **{'wall':False,'n_deltas':2}),                  
#                     SWSSketch(**{'n_formants_max':7,'time_step':0.01})), 
#                ((STFTPeaksBDB,{'wall':False,'delta_t_max':60.0}),1,
#                 STFTPeaksSketch(**{'scale':1024, 'step':512,'downsample':fs})), 
#                     ((CochleoPeaksBDB,{'wall':False}),1,
#                     CochleoPeaksSketch(**{'fs':fs,'step':128,'downsample':fs,'frmlen':8})),
 ((CQTPeaksBDB,{'wall':False}),1,
     CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':fs})) 
                 ]

for (fgpthandlename, fgptparams),n_jobs,sk in setups:
    
    sk_id = sk.__class__.__name__[:-6]
#    print fgpthandlename, sk
    for sparsity in sparsities:    
        # construct a nice name for the DB object to be saved on disk
        db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                                int(seg_dur), int(fs))
        print db_name
        fgpthandle = fgpthandlename(op.join(db_path, db_name),
                                       load=not learn,
                                       persistent=True, **fgptparams)

        ################# This is a complete experimental run given the setup ############## 
        # create the base:
        if learn:
            db_creation(fgpthandle, sk, sparsity,
                    file_names, 
                    force_recompute = True,
                    seg_duration = seg_dur, resample = fs,
                    files_path = audio_path, debug=False, n_jobs=n_jobs)
        
        
        # run a fingerprinting experiment
        test_proportion = 0.25 # proportion of segments in each file that will be tested
    #    print fgpthandle.dbObj.stat_print()
        if test:
            tstart = time.time()
            scores, failures = db_test(fgpthandle, sk, sparsity,
                             file_names, 
                             files_path = audio_path,
                             test_seg_prop = test_proportion,
                             seg_duration = seg_dur, resample =fs,
                             step = step, tolerance = 7.5, shuffle=True, debug=False,
                             n_jobs=n_jobs)
            ttest = time.time() - tstart
            ################### End of the complete run #####################################
            # saving the results
            score_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d_step%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                                    int(seg_dur), int(fs), int(100.0*test_proportion),int(step))
            
            stats =  os.stat(op.join(db_path, db_name))
            savemat(op.join(score_path,score_name), {'score':scores, 'time':ttest,
                                                     'size':stats.st_size,'failures': failures})
            
