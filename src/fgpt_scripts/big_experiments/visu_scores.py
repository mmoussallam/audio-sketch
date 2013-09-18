'''
fgpt_scripts.big_experiments.visu_scores  -  Created on Sep 18, 2013
@author: M. Moussallam
'''

import os.path as op
from classes.sketches.bench import *
from classes.sketches.cochleo import *
from scipy.io import loadmat
from PyMP import Signal
from classes import pydb
score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'
db_path = '/home/manu/workspace/audio-sketch/fgpt_db'
set_id = 'RWCLearn' # Choose a unique identifier for the dataset considered
figure_path = '/home/manu/workspace/audio-sketch/src/reporting/figures'
seg_dur = 5
step = 3.0

sparsities = [200,150,100,50,30,20,10,5,3]
sparsities = [200,150]
setups = [
          (XMDCTSparseSketch(**{'scales':[2048, 4096, 8192],'n_atoms':150,
                                                  'nature':'LOMDCT'}),8000,sparsities,'k-s',.25),
           (STFTPeaksSketch(**{'scale':2048, 'step':512}),8000, sparsities, 'b-+', .25),
           (CochleoPeaksSketch(**{'fs':8000,'step':512}),8000, sparsities, 'r-o', 0.25)    
              ]
#sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
#sk = CochleoPeaksSketch(**{'fs':fs,'step':512})
legends = []
plt.figure(figsize=(12,6))
for setup in setups:
    (sk, fs, sparsities, mark, test_proportion) = setup

    sk_id = sk.__class__.__name__[:-6]
    
    
    # initialize the sketch on noise
    if seg_dur>0:
        sk.recompute(Signal(np.random.randn(seg_dur*fs), fs, mono=True))
    
    sizes = []
    scores = []
    nkeys= []
    cons_scores = []
    times = []
    for sparsity in sparsities:
        
        # we just need a short adaptation
#        if seg_dur>0:
#            sk.sparsify(sparsity)
#            sc_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d_step%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
#                                                int(seg_dur), int(fs), int(100.0*test_proportion), int(step))
#        else:
        sc_root = "%s_%s_k%d_"%(set_id, sk_id, sparsity)
        cands = [ f for f in os.listdir(op.join(score_path)) if sc_root in f]
        # filter again with fs and proportion
        subcands = [f for f in cands if '%dfs_test%d_step%d.mat'%(int(fs), int(100.0*test_proportion), int(step)) in f]
        sc_name = subcands[0]
            
        
        D = loadmat(op.join(score_path,sc_name))
        
#        sizes.append(float(D['size'])/(1024.0*1024.0))
        scores.append(D['score'][0][0])
        cons_scores.append(1-D['score'][-1])
        times.append(D['time'][0][0])
#        db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
#                                            int(seg_dur), int(fs))
        db_root = "%s_%s_k%d_"%(set_id, sk_id, sparsity)
        cands = [ f for f in os.listdir(op.join(db_path)) if db_root in f]
        # filter again with fs and proportion
        subcands = [f for f in cands if '%dfs.db'%(int(fs)) in f]
        db_name = subcands[0]
        
        fgpthandle = pydb.FgptHandle(op.join(db_path, db_name), load=True, persistent=True, rd_only=True)
        nkeys.append(float(fgpthandle.dbObj.stat()['nkeys']))
        sizes.append(float(os.stat(op.join(db_path, db_name)).st_size))
#        fgpthandle.dbObj.close()
    #plt.subplot(211)
#    plt.plot(nkeys, 100*np.array(scores), mark)
    plt.subplot(121)
    plt.semilogx(np.array(sizes)/(1024.0*1024.0), 100*np.array(cons_scores),mark)    
    plt.xlabel('DB size (Mbytes)')
    plt.ylabel('Recognition rate (\%)')
    plt.subplot(122)
    plt.semilogx(np.array(times), 100*np.array(cons_scores),mark)    
    plt.xlabel('Computation Times (s)')
    plt.ylabel('Recognition rate (\%)')
    
    legends.append(("%s "%sk_id))
#    legends.append(("%s soft"%sk_id))
    #plt.subplot(212)
    #plt.plot(times, 100*np.array(scores),'+')
    #plt.grid()
    #plt.xlabel('Comp. Time (s)')
    #plt.ylabel('Recognition rate (\%)')
plt.subplot(121)
plt.grid()
plt.subplot(122)
plt.grid()    
plt.legend(legends, loc='lower right')    
plt.subplots_adjust(left=0.06,right=0.96, top=0.96)
#plt.savefig(op.join(figure_path, '%s_Scores_%dfgpts_dur%d.pdf'%(set_id, len(setups), int(seg_dur))))
#plt.savefig(op.join(figure_path, '%s_Scores_%dfgpts_dur%d.png'%(set_id, len(setups), int(seg_dur))))
plt.show()
