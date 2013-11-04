'''
manu_sandbox.visu_keypoints_distributions  -  Created on Oct 25, 2013
@author: M. Moussallam
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
import matplotlib
single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000

sparsity = 60
figure_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/figures')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
max_file_num = 100
seg_dur = 4
audio_path,ext = bases[set_id]
filenames = get_filepaths(audio_path, 0,  ext=ext)[:max_file_num]
fs=8000
n_segments = 600
nb_bins = 50
l = 5
scales = [64,128,256,512,1024,2048]
output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs')

w03_kphisto = np.load(op.join(output_path,
                              'W03keypoints_distrib_%d_segs_k%d_%dbins.npy'%(n_segments,
                                                                             sparsity,
                                                                             nb_bins)))

C10_kphisto = np.load(op.join(output_path,
                              'C10keypoints_distrib_%d_segs_k%d_%dbins.npy'%(n_segments,
                                                                             sparsity,
                                                                             nb_bins)))
M13_kphisto = np.load(op.join(output_path,
                              'M13keypoints_distrib_%d_segs_k%d_%dbins.npy'%(n_segments,
                                                                             sparsity,
                                                                             nb_bins)))
w03_lmhisto = np.load(op.join(output_path,
                              'W03landmarks_distrib_%d_segs_k%d_%dbins.npy'%(n_segments,
                                                                             sparsity,
                                                                             nb_bins)))
C10_lmhisto = np.load(op.join(output_path,
                              'C10landmarks_distrib_%d_segs_k%d_%dbins.npy'%(n_segments,
                                                                             sparsity,
                                                                             nb_bins)))
M13_lmhisto = np.load(op.join(output_path,
                              'M13landmarks_distrib_%d_segs_k%d_%dbins.npy'%(n_segments,
                                                                             sparsity,
                                                                             nb_bins)))

    
plt.figure(figsize=(8,3))
#plt.subplot(211)
#plt.plot(np.log(w03_kphisto),'kx-',linewidth=2.0)
#plt.plot(np.log(C10_kphisto),'bo-',linewidth=2.0)
#plt.plot(np.log(M13_kphisto),'rs-',linewidth=2.0)
#plt.ylabel('Log-probability',fontsize=16)
#plt.xlabel('Keypoint index',fontsize=16)
##plt.yticks([])
#plt.gca().set_yticklabels([])
#plt.xticks(range(0,nb_bins,nb_bins/10),[])
#plt.grid()
#plt.subplot(212)
plt.plot(np.log(w03_lmhisto),'kx-',linewidth=2.0)
plt.plot(np.log(C10_lmhisto),'bo-',linewidth=2.0)
plt.plot(np.log(M13_lmhisto),'rs-',linewidth=2.0)
plt.legend(('W03','C10','$\lambda_H= %d$ ($b$,$W$)'%l), loc='lower left')
plt.ylabel('Log-probability',fontsize=16)
plt.xlabel('Landmark index',fontsize=16)
#plt.yticks([])
plt.gca().set_yticklabels([])
plt.xticks(range(0,nb_bins,nb_bins/20),[])
plt.grid()
plt.subplots_adjust(left=0.08,bottom=0.15,right=0.97,top=0.96)
plt.savefig(op.join(figure_path,'EmpiricalKPLMdistribs_%s_%d_%dk_%dxLOMDCT_%dbins.pdf'%(set_id,n_segments,sparsity,len(scales),nb_bins)))
plt.show()