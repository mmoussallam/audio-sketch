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
ntest = 2
snrs = [-10,-5,0,5,10,20,30]#0,5,10,20]
nsegs = 20
suffix = '%dsnrs_%dsegs_%dtests_%dsparsity'%(len(snrs),nsegs,ntest,sparsity)

legends = []

plt.figure(figsize=(8,3))
#### WANG 03
W03D = loadmat(op.join(output_path,'W03_%s.mat'%suffix))
W03scores = W03D['scores']
plt.plot(snrs, np.mean(np.mean(W03scores, axis=2),axis=0),'kx-',linewidth=3.0,markersize=10.0)
legends.append('W03')
#### Cotton 10
C10D = loadmat(op.join(output_path,'C10_%s.mat'%suffix))
C10scores = C10D['scores']
plt.plot(snrs, np.mean(np.mean(C10scores, axis=2),axis=0),'o-',linewidth=2.0,markersize=8.0)
legends.append('C10')
### M13
scales = [64,128,256,512,1024,2048]

Lambdas = [1,10]
colors = ['r','m']
for i, l in enumerate(Lambdas):
    lambdas = [l]*len(scales)
    M13D = loadmat(op.join(output_path,'M13_%s_lambda_%d.mat'%(suffix,l)))
    M13_scores = M13D['scores']
    plt.plot(snrs, np.mean(np.mean(M13_scores, axis=2),axis=0),colors[i]+'d-',linewidth=i+1.5,markersize=8.0)  
    legends.append('$\lambda_H=%d$ ($b$,$W$)'%l)
    
plt.xlabel('SNR (dB)',fontsize=16.0)
plt.ylabel('PIL',fontsize=16.0)
plt.subplots_adjust(left=0.10,bottom=0.17,right=0.96,top=0.93)
plt.legend(legends, loc='lower right',ncol=2)
plt.grid()
plt.savefig(op.join(figure_path, 'robustness_%s%s'%(suffix,'.pdf')))
plt.savefig(op.join(figure_path, 'robustness_%s%s'%(suffix,'.png')))
plt.show()