'''
manu_sandbox.visu_shifts  -  Created on Oct 22, 2013
@author: M. Moussallam
'''
'''
manu_sandbox.visu_robustness  -  Created on Oct 18, 2013
@author: M. Moussallam
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
fs = 8000

figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')
output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs/robustness')

##### parameters
sparsity = 30
shifts = np.array([1.0,10.0,100.0])/float(fs)#0,5,10,20]
nsegs = 60
suffix = '%dshifts_%dsegs_%dsparsity'%(len(shifts),nsegs,sparsity)

legends = []

plt.figure(figsize=(6,3))
#### WANG 03
W03D = loadmat(op.join(output_path,'W03_%s.mat'%suffix))
W03scores = W03D['scores']
plt.plot(shifts, np.mean(W03scores, axis=0),'x-')
legends.append('W03 - $\lambda_H=\infty$')
#### Cotton 10
C10D = loadmat(op.join(output_path,'C10_%s.mat'%suffix))
C10scores = C10D['scores']
plt.plot(shifts,np.mean(C10scores, axis=0),'o-')
legends.append('C10 - $\lambda_H=0$')
### M13
scales = [64,512,2048]
Kmax = 5
Lambdas = [0,1,10,100]
for i, l in enumerate(Lambdas):
    lambdas = [l]*len(scales)
    M13D = loadmat(op.join(output_path,'M13_%s_lambda_%d_K_%d.mat'%(suffix,np.sum(lambdas),Kmax)))
    M13_scores = M13D['scores']
    plt.plot(shifts, np.mean(M13_scores, axis=0),'d-',linewidth=i+1)  
    legends.append('$\lambda_H=%d$'%l)
    
plt.xlabel('Shift (s)')
plt.ylabel('Proportion of Identical Landmarks')
plt.subplots_adjust(left=0.17,bottom=0.17,right=0.96,top=0.87)
plt.legend(legends, loc='lower right')
plt.grid()
plt.savefig(op.join(figure_path, 'robustness_shifts_%s%s'%(suffix,'.pdf')))
plt.savefig(op.join(figure_path, 'robustness_shifts_%s%s'%(suffix,'.png')))
plt.show()