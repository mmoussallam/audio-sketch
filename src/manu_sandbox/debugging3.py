'''
manu_sandbox.debugging3  -  Created on Oct 24, 2013
@author: M. Moussallam
'''
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
scales = [64,128,256,512,1024,2048]
l = 1

biaises = []
Ws = []
Wt = []            
lambdas = [l]*len(scales)
for sidx, s in enumerate(scales):    
    W03_ref = STFTPeaksSketch(**{'scale':s,'step':s/4})
    W03_ref.recompute(Signal(np.random.randn(fs*seg_dur), fs))
    W03_ref.sparsify(sparsity)
    K = W03_ref.params['f_width']
    T = W03_ref.params['t_width']
    print "K = %d, T=%d"%(K,T)
#        biais = np.zeros((s/2,))
    biais = np.linspace(1,1/s,s/2)**2    
    biaises.append(biais)
    W = np.zeros((s/2,s/2))
    for k in range(-int(K/2),int(K/2)):
        W += np.eye(s/2,s/2,k)
    Ws.append(W)    
    Wt.append(T)     


sparsity = 10
M13_skhandle = XMDCTPenalizedPairsSketch(**{'scales':scales,'n_atoms':sparsity,
                                 'nature':'LOMDCT',
                                 'lambdas':lambdas,
                                 'biaises':biaises,
                                 'Wts':Wt,#'crop':4*8192,
                                 'Wfs':Ws,'pad':False,'debug':1})

pb_sig = LongSignal(file_names[1], mono=True, frame_duration=seg_dur)


sub_sig = pb_sig.get_sub_signal(0,1, normalize=True)
M13_skhandle.recompute(sub_sig)
M13_skhandle.sparsify(sparsity)    

for block in M13_skhandle.rep.dico.blocks:
    plt.figure()
    block.draw_mask()
    plt.colorbar()

plt.show()