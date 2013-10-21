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

sparsities = [5,10]
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 1.0
learn = True
test = True
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
Lambdas = [0,]
scales = [64,512,4096]
Kmaxes = [1,10]
nb_files = 10

 
legends=[]
# Run with various Lambdas and Kmax    
for l in Lambdas:
    for Kmax in Kmaxes:
        scores = []
        cons_scores=[]
        times=[]
        sizes=[]
        for sparsity in sparsities:   
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

plt.legend(legends,loc='lower right')
plt.show()