'''
manu_sandbox.debugging2  -  Created on Oct 21, 2013
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
sparsity = 30
#sparsities = [5,10,30,50]
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 1.0
scales = [64,512,4096]
l = 0
kmax = 10
biaises = []
Ws0 = []
Ws1 = []
Wt = [512,96,20]            
lambdas = [l]*len(scales)
for sidx, s in enumerate(scales):    
    # ultra penalize low frequencies                
    biaises.append(0.000001*np.zeros((s/2,))) # no biais for now
    W = np.zeros((s/2,s/2))
    for k in range(-(sidx+1)*kmax,(sidx+1)*kmax):
        W += np.eye(s/2,s/2,k)
    Ws0.append(W)
    Ws1.append(np.zeros((s/2,s/2)))    

M13_skhandle_0 = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                 'lambdas':[0,0,0],
                                 'biaises':biaises,
                                 'Wts':Wt,'fs':fs,#'crop':4*8192,
                                 'Wfs':Ws1,'pad':2*8192,'debug':1})


M13_skhandle_1 = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                 'lambdas':[0,0,0],
                                 'biaises':biaises,
                                 'Wts':Wt,#'crop':4*8192,
                                 'Wfs':Ws0,'pad':False,'debug':1})

pb_sig = LongSignal(file_names[1], mono=True, frame_duration=seg_dur)
for segIdx in range(pb_sig.n_seg):
    print segIdx
    sub_sig = pb_sig.get_sub_signal(segIdx,1, normalize=True)
    M13_skhandle_0.recompute(sub_sig)
    M13_skhandle_0.sparsify(sparsity)    
    M13_skhandle_1.recompute(sub_sig)
    M13_skhandle_1.sparsify(sparsity)
    print M13_skhandle_0.rep
    print M13_skhandle_1.rep
#    for atomIdx in range(M13_skhandle_0.rep.atom_number):
#        assert(M13_skhandle_0.rep.atoms[atomIdx]==M13_skhandle_1.rep.atoms[atomIdx])
#
#M13_skhandle_0.rep.dico.blocks[1].draw_mask()


########################
M13_skhandle_2 = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':1,
                                 'lambdas':[0,0,0],
                                 'biaises':biaises,
                                 'Wts':Wt,#'crop':4*8192,
                                 'Wfs':Ws0,'pad':False,'debug':1})
import cProfile
M13_skhandle_2.recompute(sub_sig)
cProfile.runctx('M13_skhandle_2.sparsify(100)', globals(), locals())