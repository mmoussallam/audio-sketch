'''
manu_sandbox.compare_shazam_params  -  Created on Oct 22, 2013
@author: M. Moussallam
'''

import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
fs = 8000

figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/recognition')
db_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/db')
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms

import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(2,512*1024*1024,0)
env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
env.log_set_config(db.DB_LOG_IN_MEMORY, 1)
env.open(None, env_flags)

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)

nb_files = 100
file_names = file_names[:nb_files]

def _run_reco_expe(fgpthandle, skhandle, sparsity, test_proportion):
    ################# This is a complete experimental run given the setup ############## 
    # create the base:
    if learn:
        db_creation(fgpthandle, skhandle, sparsity,
                file_names, 
                force_recompute = True,
                step = float(seg_dur)/2,
                seg_duration = seg_dur, 
                files_path = audio_path, debug=False, n_jobs=1)
    
    
        # run a fingerprinting experiment
    if test:
        tstart = time.time()
        scores, failures = db_test(fgpthandle, skhandle, sparsity,
                         file_names, 
                         files_path = audio_path,
                         test_seg_prop = test_proportion,
                         seg_duration = seg_dur, 
                         step = step, tolerance = 7.5, shuffle=1001, debug=False,
                         n_jobs=1)
        ttest = time.time() - tstart
        ################### End of the complete run #####################################
    stats =  os.stat(op.join(db_path, db_name))
    return scores, stats

# Try with wang
sparsities = [100]
test_proportion = 0.25
seg_dur = 5
step = 3
learn = True
test = True
for sparsity in sparsities:
#    W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})
#    sk_id = "W03_no_TZ"
#    db_name = "%s_%s_k%d_%dsec_%dfs.db"%(set_id, sk_id, sparsity,
#                                                    int(seg_dur), int(fs))
#    
#    W03_fgpthandle = STFTPeaksBDB(op.join(db_path, db_name),load=False,persistent=True,
#                                  **{'wall':False,
#                                     'delta_t_max':600.0})
#    tstart = time.time()
#    
#    scores, stats =  _run_reco_expe(W03_fgpthandle, W03_skhandle, sparsity, test_proportion)
#    
#    ttest = time.time() - tstart
#    # saving the results
#    score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_.mat"%(set_id,nb_files, sk_id, sparsity, 
#                                            int(seg_dur), int(fs), int(100.0*test_proportion),
#                                            int(step))    
#    
#    savemat(op.join(output_path,score_name), {'score':scores, 'time':ttest,
#                                             'size':W03_fgpthandle.get_kv_size()})
    
    scales = [128,512,2048]
    C10_skhandle = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})
    sk_id = "C10_LO_%dxMDCT"%len(scales)
    db_name = "%s_%s_k%d_%dsec_%dfs.db"%(set_id, sk_id, sparsity,
                                                    int(seg_dur), int(fs))
    
    C10_fgpthandle = SparseFramePairsBDB(op.join(db_path, db_name),load=False,persistent=True,
                                         **{'wall':False,
                                          'nb_neighbors_max':3,
                                          'delta_t_max':3.0})
    tstart = time.time()
    
    scores, stats =  _run_reco_expe(C10_fgpthandle, C10_skhandle, sparsity, test_proportion)
    
    ttest = time.time() - tstart
    # saving the results
    score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_.mat"%(set_id,nb_files, sk_id, sparsity, 
                                            int(seg_dur), int(fs), int(100.0*test_proportion),
                                            int(step))    
    
    savemat(op.join(output_path,score_name), {'score':scores, 'time':ttest,
                                             'size':C10_fgpthandle.get_kv_size()})
    
    