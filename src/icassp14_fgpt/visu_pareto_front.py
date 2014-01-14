'''
manu_sandbox.visu_pareto_front  -  Created on Oct 24, 2013
Can we visualize a pareto front?
That would be a really nice figure!
@author: M. Moussallam
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
fs = 8000

figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
robust_output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/robustness')
reco_output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/recognition')
db_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/db')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered

sparsities = [10,30]
targsnr = 10
seg_dur = 5
fs = 8000
step = 3.0
test_proportion = 1.0
learn = True
test = True
from src.manu_sandbox.sketch_objects import XMDCTPenalizedPairsSketch
Lambdas = [1,10]
scales = [64,128,256,512,1024,2048]
nb_files = 100
nature = 'LOMDCT'
ntest = 2
snrs = [-10,-5,0,5,10,20,30]#0,5,10,20]
nsegs = 20



markers = ('o','s')
plt.figure(figsize=(4,3))
for isp, sparsity in enumerate(sparsities):
    ys = []
    xs = []
    suffix = '%dsnrs_%dsegs_%dtests_%dsparsity'%(len(snrs),nsegs,ntest,sparsity)

    # Ok let us start with the reference W03 the x position is the reco rate at given sparsity
    # the y position is the Proportion of Identical Landmarks
    
    ################# WANG 03 ##############
    W03D = loadmat(op.join(robust_output_path,'W03_%s.mat'%suffix))
    W03scores = W03D['scores']
    
    W03y = np.mean(W03scores[:,snrs.index(targsnr),:])
    
    ys.append(W03y)
    
    sc_root = "%s_%d_W03_no_TZ_k%d_"%(set_id, nb_files,  sparsity)
    cands = [ f for f in os.listdir(op.join(reco_output_path)) if sc_root in f]

    sc_name = cands[0]        
    D = loadmat(op.join(reco_output_path,sc_name))
    W03x = 1-D['score'][-1]
    
    xs.append(W03x)
    ################# Cotton 10 ##############
    C10D = loadmat(op.join(robust_output_path,'C10_%s.mat'%suffix))
    C10scores = C10D['scores']
    
    C10y = np.mean(C10scores[:,snrs.index(targsnr),:])
    ys.append(C10y)
    sk_id = "M13_bias_lambH%d_%dx%s"%(0,len(scales),nature)
    # saving the results
    score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dx%s.mat"%(set_id,nb_files,sk_id, sparsity, 
                                            int(seg_dur), int(fs),
                                            int(100.0*test_proportion),
                                            int(step),len(scales),nature)

    D = loadmat(op.join(reco_output_path,score_name))
    C10x = 1-D['score'][-1]
    xs.append(C10x)
    
    ################# Ours 13 ##############
    for l in Lambdas:
        M13D = loadmat(op.join(robust_output_path,'M13_%s_lambda_%d.mat'%(suffix,l)))
        M13scores = M13D['scores']
        
#        C10y = np.mean(C10scores[:,snrs.index(targsnr),:])
        ys.append(np.mean(M13scores[:,snrs.index(targsnr),:]))
        
        sk_id = "M13_bias_lambH%d_%dx%s"%(l,len(scales),nature)
        # saving the results
        score_name = "%s_%d_%s_k%d_%dsec_%dfs_test%d_step%d_%dx%s.mat"%(set_id,nb_files,sk_id, sparsity, 
                                                int(seg_dur), int(fs),
                                                int(100.0*test_proportion),
                                                int(step),len(scales),nature)
    
        D = loadmat(op.join(reco_output_path,score_name))        
        xs.append(1-D['score'][-1])
    
#    legends.append('')
    

    plt.plot(xs,ys,markers[isp])
    
plt.xlabel('Recognition Rate (%)',fontsize=14)
plt.ylabel('PIL (%)',fontsize=14)
plt.subplots_adjust(left=0.16,bottom=0.18,right=0.95,top=0.93)
plt.grid()
plt.show()