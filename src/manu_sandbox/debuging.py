'''
manu_sandbox.debuging  -  Created on Oct 18, 2013
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
file_names = get_filepaths(audio_path, 0,  ext=ext)[:10]

from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
sparsity = 10
#sparsities = [5,10,30,50]
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 1.0
scales = [64,512,4096]
l = 0
kmax = 10
biaises = []
Ws = []
Wt = [0,0,0]            
lambdas = [l]*len(scales)
for sidx, s in enumerate(scales):    
    # ultra penalize low frequencies                
    biaises.append(0.000001*np.zeros((s/2,))) # no biais for now
    W = np.zeros((s/2,s/2))
    for k in range(-(sidx+1)*kmax,(sidx+1)*kmax):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)    
M13_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':sparsity,
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'Wts':Wt,'fs':fs,#'crop':4*8192,
                                 'Wfs':Ws,'pad':2*8192,'debug':1})
sk_id = "M13_Kmax%d_lambH%d"%(kmax,l)
db_name = "%s_%s_k%d_%dsec_%dfs.db"%(set_id, sk_id, sparsity,
                                    int(seg_dur), int(fs))

M13_fgpthandle = SparseFramePairsBDB(op.join(db_path, db_name),load=False,persistent=True,
                                     **{'wall':False,
                                        'nb_neighbors_max':3,
                                        'delta_t_max':3.0})  

#pb_sig = LongSignal(file_names[1], mono=True, frame_duration=seg_dur)
#for segIdx in range(pb_sig.n_seg):
#    print segIdx
#    sub_sig = pb_sig.get_sub_signal(segIdx,1, normalize=True)
##    sub_sig.downsample(fs)    
##    sub_sig.crop(0,4*8192)
##    sub_sig.pad(2*8192)
#    M13_skhandle.recompute(sub_sig)
#    M13_skhandle.sparsify(sparsity)
    

#db_creation(M13_fgpthandle, M13_skhandle, sparsity,
#                file_names, 
#                force_recompute = True,
#                seg_duration = seg_dur, 
#                files_path = audio_path, debug=False, n_jobs=1)
#
#scores, failures = db_test(M13_fgpthandle, M13_skhandle, sparsity,
#                         file_names, 
#                         files_path = audio_path,
#                         test_seg_prop = test_proportion,
#                         seg_duration = seg_dur, 
#                         step = step, tolerance = 7.5, shuffle=True, debug=False,
#                         n_jobs=1)
#print scores





