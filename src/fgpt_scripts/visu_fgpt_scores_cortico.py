'''
fgpt_scripts.visu_fgpt_scores_cortico  -  Created on Aug 20, 2013
@author: M. Moussallam
'''

import os
import os.path as op
import matplotlib.pyplot as plt
from classes.sketches.cortico import *
from scipy.io import loadmat
from PyMP import Signal
db_path = '/home/manu/workspace/audio-sketch/fgpt_db/'
score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'
set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered

seg_dur = 5.0
test_proportion = 0.25
step = 5
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
sparsities =  [30,20,10,9,8,7,6,5,4,3]
fs = 8000

## Initialize the sketchifier
#sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
sk = CorticoIndepSubPeaksSketch(**{'fs':fs,'downsample':fs,'frmlen':8,
                                   'shift':0,'fac':-2,'BP':1})
#sk = CochleoPeaksSketch(**{'fs':fs,'step':512,'downsample':fs})
sk_id = sk.__class__.__name__[:-6]
    
    
# initialize the sketch on noise
sk.recompute(Signal(np.random.randn(seg_dur*fs), fs, mono=True))

(N,M) = sk.cort.cor.shape[:2]
sizes = np.zeros((N,M/2, len(sparsities)))
scores = np.zeros((N,M/2, len(sparsities)))
cons_scores = np.zeros((N,M/2, len(sparsities)))
times = []

for sp_ind, sparsity in enumerate(sparsities):
    
    # we just need a short adaptation
    sk.sparsify(sparsity)
    
    sc_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d_step%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs), int(100.0*test_proportion), step)
    D = loadmat(op.join(score_path,sc_name))
    
    db_name = "%s_%s_k%d_%s_%dsec_%dfs/"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs))
    path = op.join(db_path, db_name)
    for n in range(N):
        for m in range(M/2):
            
            sizes[n,m, sp_ind] = os.stat(op.join(path,"_%d_%d.db"%(n,m))).st_size/(1024.0*1024.0)  
            

#    sizes[:,:,sp_ind] = D['size']/(1024.0*1024.0)  
    
    scores[:,:,sp_ind] = D['score'][0]
    cons_scores[:,:,sp_ind] = (1-D['score'][-1])
    times.append(D['time'][0][0])

plt.figure()
for n in range(N):
    for m in range(M/2):
        plt.subplot(N, M/2, n*(M/2) + m +1)
#         plt.semilogx(sizes[n,m,:], 100*np.array(scores[n,m,:]), 'b')
        plt.semilogx(sizes[n,m,:], 100*np.array(cons_scores[n,m,:]),'g')
        plt.ylim([90,98])
        plt.xlim([0.1,10])
        plt.grid()

plt.xlabel('DB size (Mbytes)')
plt.ylabel('Recognition rate (\%)')

#legends.append(("%s hard"%sk_id))
#legends.append(("%s soft"%sk_id))
#plt.subplot(212)
#plt.plot(times, 100*np.array(scores),'+')
#plt.grid()
#plt.xlabel('Comp. Time (s)')
#plt.ylabel('Recognition rate (\%)')

# plt.grid()    
#plt.legend(legends, loc='lower right')    
plt.show()
