'''
manu_sandbox.visu_cooc_formatted  -  Created on Oct 21, 2013
@author: M. Moussallam
'''

import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
import matplotlib.cm as cm
#mem = Memory('/tmp/audio-sketch/')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
max_file_num = 200

seg_dur = 4
sparsity = 100
fs = 8000.0
# the density is approximately of sparsity/seg_dur features per second
#scales = [2**j for j in range(5,12)]
#scales = [64,512,4096]
scales = [2048,]

figure_path = op.join(SKETCH_ROOT,'src/manu_sandbox/figures')
output_path = op.join(SKETCH_ROOT,'src/manu_sandbox/outputs')

WF = np.load(op.join(output_path, "W_F_%dfiles_%dxMDCT_%dk.npy"%(max_file_num, len(scales), sparsity)))
WT = np.load(op.join(output_path, "W_T_%dfiles_%dxMDCT_%dk.npy"%(max_file_num, len(scales), sparsity)))

plt.figure(figsize=(8,3))
plt.subplot(211)
A = np.sum(WF, axis=0).astype(float)
plt.plot(range(1024), np.log(A))
plt.xticks([64,128,256,512,1024],[0.25, 0.5,1,2,4])
plt.xlabel('Frequency (KHz)')
plt.ylabel('Log-probability')
plt.subplot(212)
B = np.sum(WT, axis=0).astype(float)
plt.plot(np.log(B))
plt.xticks([0,100,200,300],[0,1,2,3])
plt.xlabel('Time (sec)')
plt.ylabel('probability')
plt.show()


if False:
    map = cm.bone_r
    
    plt.figure(figsize=(8,4*len(scales)))
    
    plt.subplot(1,2,1)
    biais = np.mean(WF, axis=0)
    r_biais = np.maximum(biais,1).reshape(len(biais),1)
    r_biais_mat = np.dot(r_biais, r_biais.T)
    plt.imshow(np.log(WF/r_biais_mat), origin='lower', cmap=map) 
    plt.ylim([20,512])
    plt.xlim([20,512]) 
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Frequency Index ',size=16.0)
    plt.ylabel('Frequency Index ',size=16.0)
    #plt.colorbar()
    plt.subplot(1,2,2)
    biais = np.mean(WT, axis=0)
    t_biais = np.maximum(biais,1).reshape(len(biais),1)
    t_biais_mat = np.dot(t_biais, t_biais.T)
    plt.imshow(np.log(WT/t_biais_mat), origin='lower', cmap=map)
    plt.ylim([0,300])
    plt.xlim([0,300])
    #plt.colorbar() 
    plt.xticks([])
    plt.yticks([])   
    plt.xlabel('Time Index',size=16.0)
    plt.ylabel('Time Index',size=16.0)
    plt.subplots_adjust(left=0.07,top=0.97, right=0.97)                       
                                                      
    figure_path = op.join(SKETCH_ROOT,'src/manu_sandbox/figures')
    plt.savefig(op.join(figure_path, "WF_WT_%dfiles_%dxMDCT_%dk.pdf"%(max_file_num, len(scales), sparsity)))
    plt.savefig(op.join(figure_path, "WF_WT_%dfiles_%dxMDCT_%dk.png"%(max_file_num, len(scales), sparsity)))                                 
    plt.show()
