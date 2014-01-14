'''
fgpt_scripts.many_sparsities_cortico  -  Created on Aug 5, 2013
@author: M. Moussallam
'''
'''
fgpt_scripts.many_sparsities  -  Created on Jul 30, 2013
@author: M. Moussallam

Now let us see how performances evolve with the sparsity
'''
import sys, os
from classes.sketches.cochleo import SKETCH_ROOT
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
from tools.fgpt_tools import get_filepaths
from tools.fgpt_tools import db_creation, db_test, db_test_cortico

db_path = '/home/manu/workspace/audio-sketch/fgpt_db/'


# The RWC subset path
set_id = 'voxforge' # Choose a unique identifier for the dataset considered
audio_path, ext = bases[set_id]

score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'

file_names = get_filepaths(audio_path, 0,  ext=ext)
nb_files = 10
file_names = file_names[:nb_files]
# define experimental conditions

sparsities =  [100,]
seg_dur = -1
fs = 8000

## Initialize the sketchifier
#sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
sk = CorticoIndepSubPeaksSketch(**{'fs':fs,'downsample':fs,'frmlen':8,
                                   'shift':0,'fac':-2,'BP':1})
#sk = CochleoPeaksSketch(**{'fs':fs,'step':512,'downsample':fs})
sk_id = sk.__class__.__name__[:-6]


test = True
learn= False
for sparsity in sparsities:    
    # construct a nice name for the DB object to be saved on disk
    db_name = "%s_%s_k%d_%s_%dsec_%dfs/"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs))
        
    # initialize the fingerprint Handler object
    fgpthandle = CorticoIndepSubPeaksBDB(op.join(db_path, db_name),
                                              load=True, persistent=True, dbenv=None,
                                              rd_only=not learn,
                                               **{'wall':False,'max_pairs':500})
#    fgpthandle = pydb.STFTPeaksBDB(op.join(db_path, db_name),
#                                   load=True,
#                                   persistent=True, **{'wall':False})
#    fgpthandle = pydb.CochleoPeaksBDB(op.join(db_path, db_name),
#                                   load=True,
#                                   persistent=True, **{'wall':True})
    ################# This is a complete experimental run given the setup ############## 
    # create the base:
    if learn:
        db_creation(fgpthandle, sk, sparsity,
                file_names, 
                force_recompute = True,
                seg_duration = seg_dur, resample = fs,
                files_path = audio_path, debug=False, n_jobs=1)
    
    
    # run a fingerprinting experiment
    test_proportion = 1.0 # proportion of segments in each file that will be tested
    step = -1
    if test:
        tstart = time.time()
        scores = db_test_cortico(fgpthandle, sk, sparsity,
                         file_names, 
                         files_path = audio_path,
                         test_seg_prop = test_proportion,
                         seg_duration = seg_dur, resample =fs,
                         step = step, tolerance = 7.5, shuffle=True, debug=False,n_jobs=1)
        ttest = time.time() - tstart
        ################### End of the complete run #####################################
        # saving the results
        score_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d_step%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                                int(seg_dur), int(fs), int(100.0*test_proportion),int(step))
        
        
        stats =  fgpthandle.get_db_sizes()
        savemat(op.join(score_path,score_name), {'score':scores, 'time':ttest,
                                                 'size':stats})
        
