'''
manu_sandbox.visu_recognition_monoscale  -  Created on Oct 22, 2013
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

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered

sparsities =  [5,10,30,50,]
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 0.25
learn = True
test = True
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
Lambdas = [0,1,10]
scales = [2048]
nb_files = 10

 
legends=[]
# Run with various Lambdas and Kmax    
for l in Lambdas:    
    scores = []
    cons_scores=[]
    times=[]
    sizes=[]
    for sparsity in sparsities:   
        
        W03_skhandle = STFTPeaksSketch(**{'scale':scales[0],'step':scales[0]/4})
        noisy_test = Signal(np.random.randn(seg_dur*fs), mono=True)
        noisy_test.pad(2*8192)
        W03_skhandle.recompute(noisy_test)
        W03_skhandle.sparsify(sparsity)
        
        print "Parameters", W03_skhandle.params['f_width'],W03_skhandle.params['t_width']
        Kmax = W03_skhandle.params['f_width']/2
        
        sk_id = "M13_Kmax%d_lambH%d"%(Kmax,l)
        # saving the results
        score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dxMCDT.mat"%(set_id,nb_files,sk_id, sparsity, 
                                                int(seg_dur), int(fs),
                                                int(100.0*test_proportion),
                                                int(step),len(scales))

        D = loadmat(op.join(output_path,score_name))
        
#        sizes.append(float(D['size'])/(1024.0*1024.0))
        scores.append(D['score'][0][0])
        cons_scores.append(1-D['score'][-1][0])
        times.append(D['time'][0][0])
        sizes.append(D['size'][0][0])
        print cons_scores
    plt.subplot(121)
    plt.semilogx(np.array(sizes)/(1024.0*1024.0), 100*np.array(cons_scores),'-', linewidth=2.0)
    plt.xlabel('DB size (Mbytes)')
    plt.ylabel('Recognition rate (\%)')
    plt.subplot(122)
    plt.semilogx(np.array(times), 100*np.array(cons_scores),'-', linewidth=2.0)    
    plt.xlabel('Computation Times (s)')
    plt.ylabel('Recognition rate (\%)')
        
    legends.append(("%s "%sk_id))

# state of the art comparison
sparsities = [5,10,30,50,100,200]
sizes = []
scores = []
nkeys= []
cons_scores = []
times = []
sim_sized = []
score_path =op.join(SKETCH_ROOT,'fgpt_scores')
for sparsity in sparsities:
    sc_root = "%s_STFTPeaks_k%d_"%(set_id,  sparsity)
    cands = [ f for f in os.listdir(op.join(score_path)) if sc_root in f]
    subcands = [f for f in cands if '%dfs_test%d_step%d.mat'%(int(fs), int(100.0*0.25), int(step)) in f]
    sc_name = subcands[0]
        
    D = loadmat(op.join(score_path,sc_name))
    scores.append(D['score'][0][0])
    cons_scores.append(1-D['score'][-1])
    times.append(D['time'][0][0])
    
    db_root = "%s_STFTPeaks_k%d_"%(set_id,  sparsity)
    cands = [ f for f in os.listdir(op.join(SKETCH_ROOT,'fgpt_db')) if db_root in f]
    # filter again with fs and proportion
    subcands = [f for f in cands if '%dfs.db'%(int(fs)) in f]
    db_name = subcands[0]

    fgpthandle = FgptHandle(op.join(op.join(SKETCH_ROOT,'fgpt_db'), db_name), load=True, persistent=True, rd_only=True)
    sim_sized.append((fgpthandle.dbObj.stat()['ndata'] + fgpthandle.dbObj.stat()['nkeys']))
#       
plt.subplot(121)
plt.semilogx(np.array(sim_sized)/(1024.0*1024.0), 100*np.array(cons_scores),'--', linewidth=2.0)
          
plt.subplot(122)
plt.semilogx(np.array(times), 100*np.array(cons_scores),'--', linewidth=2.0)
legends.append('STFT')    
plt.legend(legends,loc='lower right')
plt.show()
