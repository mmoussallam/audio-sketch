'''
manu_sandbox.visu_cooc_TF_per_scale  -  Created on Oct 16, 2013
@author: M. Moussallam

See the Wt and Wf matrices for three scales on GTZAN or voxforge

'''

import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
#mem = Memory('/tmp/audio-sketch/')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
max_file_num = 100

seg_dur = 4
sparsity = 100
fs = 8000.0
# the density is approximately of sparsity/seg_dur features per second
#scales = [2**j for j in range(5,12)]
#scales = [64,512,4096]
scales = [64,512,4096]
skhandlename = XMDCTSparseSketch
nature = 'LOMDCT'
params = {'downsample':fs,'scales':scales,'nature':nature,'n_atoms':sparsity}

skhandle = skhandlename(**params)

sp_reps = []
freqs = []
times = []


from scipy.sparse import dok_matrix

# we first need to get the whole matrix dimension
# test on the first sub_seg
t_sig = LongSignal(file_names[0],frame_duration=seg_dur, mono=True)
t_seg = t_sig.get_sub_signal(0,1, mono=True, normalize=True)
t_seg.resample(fs)
skhandle.recompute(t_seg)
L = skhandle.rep.recomposed_signal.length
M = len(scales)*L
sp_mat = dok_matrix((M,M))
pad=4096
Wfs = []
Wts = []
for s in scales:
#    Wfs.append(dok_matrix((s/2,s/2)))
#    T = L/(s/2)
#    Wts.append(dok_matrix((T,T)))
    Wfs.append(np.zeros((s/2,s/2), dtype=np.int16))
    T = seg_dur*100#/(s/2)
    Wts.append(np.zeros((T,T), dtype=np.int16))

## Computing all the sparse rep
for fIdx, file_name in enumerate(file_names[:max_file_num]):
    l_sig =  LongSignal(file_name,frame_duration=seg_dur, mono=True,Noverlap=0.5)
    # loop on segments
    t = time.time()
    for segIdx in range(l_sig.n_seg):
        sub_sig = l_sig.get_sub_signal(segIdx,1, mono=True, normalize=True)
        sub_sig.resample(fs)
        sub_sig.window(pad)
#        t = time.time()
        skhandle.recompute(sub_sig)
        print skhandle.rep
        for atomAnch in skhandle.rep.atoms:
            anch_bin = atomAnch.freq_bin
#            anch_frame = atomAnch.frame
#            anch_time = float(atomAnch.time_position) / fs
            anch_scale = float(atomAnch.length) 
#            anch_idx = anch_frame*(anch_scale/2) + anch_bin + scales.index(anch_scale)*L
            for atomTarg in skhandle.rep.atoms:
                targ_bin = atomTarg.freq_bin
                targ_frame = atomTarg.frame
#                targ_time = float(atomTarg.time_position) / fs
                targ_scale = float(atomTarg.length) 
#                targ_idx = targ_frame*(targ_scale/2) + targ_bin + scales.index(targ_scale)*L
#                sp_mat[anch_idx,targ_idx] += 1
                
                Wf = Wfs[scales.index(targ_scale)]
                Wt = Wts[scales.index(targ_scale)]
                trans_anch_bin = int(anch_bin * targ_scale / anch_scale)
                trans_anch_frame = int(atomAnch.time_position / (targ_scale/2))
                Wf[trans_anch_bin,targ_bin]  += 1 
                Wt[float(atomAnch.time_position-pad)/fs*100, float(atomTarg.time_position-pad)/fs*100] += 1 
                
                
    print "elapsed ",time.time()-t

figure_path = op.join(SKETCH_ROOT,'src/manu_sandbox/figures')
output_path = op.join(SKETCH_ROOT,'src/manu_sandbox/outputs')

plt.figure(figsize=(8,4*len(scales)))
for sidx,s in enumerate(scales):
    np.save(op.join(output_path, "W_F_%dfiles_%d_outof_%dxMDCT_%dk"%(max_file_num,sidx, len(scales), sparsity)), np.array(Wfs[sidx]))
    np.save(op.join(output_path, "W_T_%dfiles_%d_outof_%dxMDCT_%dk"%(max_file_num,sidx, len(scales), sparsity)),np.array(Wts[sidx]))
    plt.subplot(len(scales),2,(sidx*2)+1)
#    plt.spy(Wfs[sidx],marker='o',markersize=0.5)
    # the biais is simply the sum over the columns
    biais = np.mean(Wfs[sidx], axis=0)
    r_biais = np.maximum(biais,1).reshape(len(biais),1)
    r_biais_mat = np.dot(r_biais, r_biais.T)
    plt.imshow(np.log(Wfs[sidx]/r_biais_mat))  
    plt.subplot(len(scales),2,(sidx*2)+2)
    plt.imshow(Wts[sidx])  
                                                                 
plt.show()
#plt.spy(sp_mat,marker='o',markersize=0.5)    
#plt.subplots_adjust(left=0.05,right=1.0, bottom=0.05,top=0.97)
#plt.yticks([])
#plt.xticks([])
#plt.xlabel('Atom Index')
#plt.ylabel('Atom Index',rotation='vertical')
#plt.savefig(op.join(figure_path,
#                    'empirical_cooc_mat_%s_%dfiles_%datoms_%dxMDCT.png'%(set_id,
#                                                                        max_file_num,
#                                                                        sparsity,
#       