
'''
fgpt_scripts.visu_fgpt_scores  -  Created on Jul 30, 2013
@author: M. Moussallam
'''

import os.path as op
from classes.sketches.bench import *
from classes.sketches.cochleo import *
from scipy.io import loadmat
from PyMP import Signal

score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered

seg_dur = 5.0


setups = [
           (STFTPeaksSketch(**{'scale':2048, 'step':512}),8000, [100,50,30,10,5], '-+', 1.0),
           (CochleoPeaksSketch(**{'fs':8000,'step':512}),8000, [30,10,5,4,3], '-o', 0.25)   
              ]
#sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
#sk = CochleoPeaksSketch(**{'fs':fs,'step':512})
legends = []
plt.figure()
for setup in setups:
    (sk, fs, sparsities, mark, test_proportion) = setup

    sk_id = sk.__class__.__name__[:-6]
    
    
    # initialize the sketch on noise
    sk.recompute(Signal(np.random.randn(seg_dur*fs), fs, mono=True))
    
    sizes = []
    scores = []
    cons_scores = []
    times = []
    for sparsity in sparsities:
        
        # we just need a short adaptation
        sk.sparsify(sparsity)
        
        sc_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
                                                int(seg_dur), int(fs), int(100.0*test_proportion))
        D = loadmat(op.join(score_path,sc_name))
        sizes.append(float(D['size'])/(1024.0*1024.0))
        scores.append(D['score'][0][0])
        cons_scores.append(1-D['score'][-1])
        times.append(D['time'][0][0])
    
    
    #plt.subplot(211)
    plt.semilogx(sizes, 100*np.array(scores), 'b'+mark)
    plt.semilogx(sizes, 100*np.array(cons_scores),'g'+mark)
    
    plt.xlabel('DB size (Mbytes)')
    plt.ylabel('Recognition rate (\%)')
    
    legends.append(("%s hard"%sk_id))
    legends.append(("%s soft"%sk_id))
    #plt.subplot(212)
    #plt.plot(times, 100*np.array(scores),'+')
    #plt.grid()
    #plt.xlabel('Comp. Time (s)')
    #plt.ylabel('Recognition rate (\%)')

plt.grid()    
plt.legend(legends, loc='lower right')    
plt.show()
