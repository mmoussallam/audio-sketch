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

sparsities = [5,10,30,50,100]
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 0.25
learn = True
test = True
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
Lambdas = [1,10]
scales = [64,128,256,512,1024,2048]
nb_files = 1000
nature = 'LOMDCT'
 
legends=[]

plt.figure(figsize=(8,3))

# W03
sizes = []
scores = []
nkeys= []
cons_scores = []
times = []
sim_sized = []
score_path = output_path

for sparsity in sparsities:
    sc_root = "%s_%d_W03_no_TZ_k%d_"%(set_id, nb_files,  sparsity)
    cands = [ f for f in os.listdir(op.join(score_path)) if sc_root in f]
#    subcands = [f for f in cands if '%dfs_test%d_step%d.mat'%(int(fs), int(100.0*0.25), int(step)) in f]
    sc_name = cands[0]
        
    D = loadmat(op.join(score_path,sc_name))
    scores.append(D['score'][0][0])
    cons_scores.append(1-D['score'][-1])
    times.append(D['time'][0][0])
 
plt.plot(np.array(sparsities), 100*np.array(cons_scores),'kx-', linewidth=3.0,markersize=10.0)
legends.append('W03')


# With Lambda = 0
scores = []
cons_scores=[]
times=[]
sizes=[]
for sparsity in sparsities:   
    sk_id = "M13_justbias_lambH%d_%dx%s"%(0,len(scales),nature)
    # saving the results
    score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dx%s.mat"%(set_id,nb_files,sk_id, sparsity, 
                                            int(seg_dur), int(fs),
                                            int(100.0*test_proportion),
                                            int(step),len(scales),nature)

    D = loadmat(op.join(output_path,score_name))
    
#        sizes.append(float(D['size'])/(1024.0*1024.0))
    scores.append(D['score'][0][0])
    cons_scores.append(1-D['score'][-1][0])
    times.append(D['time'][0][0])
    sizes.append(D['size'][0][0])
    print cons_scores
plt.plot(np.array(sparsities), 100*np.array(cons_scores),'o-', linewidth=2,markersize=8.0)
plt.xlabel('Sparsity $k$',fontsize=16)
plt.ylabel('Recognition rate (%)',fontsize=16)
    
legends.append(("C10"))

# Get the ones with just the bias
for il, l in enumerate(Lambdas):
    
    scores = []
    cons_scores=[]
    times=[]
    sizes=[]
    for sparsity in sparsities:   
        sk_id = "M13_justbias_lambH%d_%dx%s"%(l,len(scales),nature)
        # saving the results
        score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dx%s.mat"%(set_id,nb_files,sk_id, sparsity, 
                                                int(seg_dur), int(fs),
                                                int(100.0*test_proportion),
                                                int(step),len(scales),nature)

        D = loadmat(op.join(output_path,score_name))
        
#        sizes.append(float(D['size'])/(1024.0*1024.0))
        scores.append(D['score'][0][0])
        cons_scores.append(1-D['score'][-1][0])
        times.append(D['time'][0][0])
        sizes.append(D['size'][0][0])
        print cons_scores
    plt.plot(np.array(sparsities), 100*np.array(cons_scores),'s--', linewidth=(il+1.5),markersize=8.0)
#    plt.xlabel('Sparsity $k$',fontsize=16)
#    plt.ylabel('Recognition rate (%)',fontsize=16)
        
    legends.append(("$\lambda_H = $%d ($b$,0)"%l))


## Get the ones with bias + W
#for il, l in enumerate(Lambdas):
    
#    scores = []
#    cons_scores=[]
#    times=[]
#    sizes=[]
#    for sparsity in sparsities:   
#        sk_id = "M13_bias_lambH%d_%dx%s"%(l,len(scales),nature)
#        # saving the results
#        score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dx%s.mat"%(set_id,nb_files,sk_id, sparsity, 
#                                                int(seg_dur), int(fs),
#                                                int(100.0*test_proportion),
#                                                int(step),len(scales),nature)
#
#        D = loadmat(op.join(output_path,score_name))
#        
##        sizes.append(float(D['size'])/(1024.0*1024.0))
#        scores.append(D['score'][0][0])
#        cons_scores.append(1-D['score'][-1][0])
#        times.append(D['time'][0][0])
#        sizes.append(D['size'][0][0])
#        print cons_scores
#    plt.plot(np.array(sparsities), 100*np.array(cons_scores),'d-', linewidth=(il+1.5),markersize=8.0)
##    plt.xlabel('Sparsity $k$',fontsize=16)
##    plt.ylabel('Recognition rate (\%)',fontsize=16)
#        
#    legends.append(("$\lambda_H = $%d ($b$,$W$)"%l))
    
plt.legend(legends,loc='lower right',ncol=2)
plt.subplots_adjust(left=0.10,bottom=0.18,right=0.97,top=0.96)
plt.grid()
plt.ylim([20,101])
plt.xlim([3,max(sparsities)+5])
plt.savefig(op.join(figure_path,'reco_complete_%s_%d.pdf'%(set_id,nb_files)))
plt.savefig(op.join(figure_path,'reco_complete_%s_%d.png'%(set_id,nb_files)))
plt.show()
