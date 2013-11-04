'''
manu_sandbox.compare_recognition_monoscale  -  Created on Oct 22, 2013
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
                step = float(seg_dur)*0.5,
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


sparsities = [20]
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 0.25
learn = True
test = True
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
Lambdas = [0,1,10]
scales = [2048]
nature = 'LOMDCT'
#################### WANG 2003
#    print fgpthandlename, sk
for sparsity in sparsities:    
    
    # W parameters same as W03
    W03_skhandle = STFTPeaksSketch(**{'scale':scales[0],'step':scales[0]/4})
    noisy_test = Signal(np.random.randn(seg_dur*fs), mono=True)
    noisy_test.pad(2*8192)
    W03_skhandle.recompute(noisy_test)
    W03_skhandle.sparsify(sparsity)
    
    print "Parameters", W03_skhandle.params['f_width'],W03_skhandle.params['t_width']
    Kmax = W03_skhandle.params['f_width']/2
    # define the skhandle
    biaises = []
    Ws = []
    Wt = [W03_skhandle.params['t_width']]           
    
    for sidx, s in enumerate(scales):    
        # ultra penalize low frequencies                
        biaises.append(0.000001*np.zeros((s/2,))) # no biais for now
        W = np.zeros((s/2,s/2))
        for k in range(-Kmax,Kmax):
            W += np.eye(s/2,s/2,k)
        Ws.append(W)    
    # Run with various Lambdas and Kmax    
    for l in Lambdas:
        lambdas = [l]*len(scales)
        M13_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                                    'nature':nature,
                                         'lambdas':lambdas,
                                         'biaises':biaises,
                                         'Wts':Wt,'fs':fs,#'crop':(seg_dur-1)*8192,
                                         'Wfs':Ws,'pad':2*8192,'debug':1})
        sk_id = "M13_Kmax%d_lambH%d_%s"%(Kmax,l,nature)
        db_name = "%s_%d_%s_k%d_%dsec_%dfs.db"%(set_id,nb_files, sk_id, sparsity,
                                            int(seg_dur), int(fs))
        
        M13_fgpthandle = SparseFramePairsBDB(op.join(db_path, db_name),load=False,persistent=True,
                                             **{'wall':False,
                                                'nb_neighbors_max':3,
                                                'delta_t_max':3.0})    

        tstart = time.time()
        try:
            scores, stats =  _run_reco_expe(M13_fgpthandle, M13_skhandle, sparsity, test_proportion)
        except:
            print "FATAL ERROR on this one ", Kmax, l, sparsity
            continue
        ttest = time.time() - tstart
        # saving the results
        score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dxMCDT.mat"%(set_id,nb_files, sk_id, sparsity, 
                                                int(seg_dur), int(fs), int(100.0*test_proportion),
                                                int(step),len(scales))
        
        
        savemat(op.join(output_path,score_name), {'score':scores, 'time':ttest,
                                                 'size':M13_fgpthandle.get_kv_size()})
        del M13_skhandle, M13_fgpthandle
    
        # Run Wang 03
#    W03_skhandle = STFTPeaksSketch(**{'scale':2048,'step':512})
#    sk_id = "W03"
#    
#    db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
#                                                int(seg_dur), int(fs))
#    
#    W03_fgpthandle = STFTPeaksBDB(db_name,load=False,persistent=True,
#                                  **{'wall':False, 'delta_t_max':3.0})
#    
#    scores, stats = _run_reco_expe(W03_fgpthandle, W03_skhandle, sparsity)
#    # saving the results
#    score_name = "%s_%s_k%d_%dsec_%dfs_test%d_step%d.mat"%(set_id, sk_id, sparsity, 
#                                            int(seg_dur), int(fs), int(100.0*test_proportion),int(step))
#    
#    
#    savemat(op.join(output_path,score_name), {'score':scores, 'time':ttest,
#                                             'size':stats.st_size,'failures': failures})
#    