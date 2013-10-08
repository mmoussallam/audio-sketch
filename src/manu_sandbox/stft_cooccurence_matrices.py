'''
manu_sandbox.stft_cooccurence_matrices  -  Created on Oct 4, 2013

Let's take a database of sounds, build the sketches (peaks) and look at their
empirical distributions and cooccurrence matrices

@author: M. Moussallam
'''
import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.manu_sandbox.settingup import *
mem = Memory('/tmp/audio-sketch/')

@mem.cache
def get_sparse_histogram(skhandlename, params,sparsity, file_names, max_file_num, seg_dur, overlap=0):
    # initialize sketch
    skhandle = skhandlename(**params)
    t_sig = LongSignal(file_names[0],frame_duration=seg_dur, mono=True, Noverlap=overlap)
    skhandle.recompute(t_sig.get_sub_signal(0,1))
    skhandle.sparsify(sparsity)
    
    histogram = np.zeros(skhandle.rep.shape, dtype=int)
    # Ok now do it for all signals and all segments
    for fIdx, file_name in enumerate(file_names[:max_file_num]):
        l_sig =  LongSignal(file_name,frame_duration=seg_dur, mono=True)
        # loop on segments
        for segIdx in range(l_sig.n_seg):
            skhandle.recompute(l_sig.get_sub_signal(segIdx,1, mono=True, normalize=True))
            skhandle.sparsify(sparsity)
            histogram[np.nonzero(skhandle.sp_rep)] += 1
    return histogram


# get the filenames
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
max_file_num = 50

seg_dur = 5
sparsity = 100
# the density is approximately of sparsity/seg_dur features per second


skhandlename = STFTPeaksSketch
params = {'downsample':8000,'scale':1024,'step':256}
histogram = get_sparse_histogram(skhandlename,params,sparsity, file_names,
                                 max_file_num, seg_dur, overlap=0)
t_vec = np.arange(histogram.shape[2])
f_vec = np.arange(histogram.shape[1])
# test on pure noise
#histogram = get_sparse_histogram(skhandlename,params, ['/sons/sqam/glocs.wav',], 1, seg_dur)
fig = plt.figure()
ax = fig.add_axes([0.3,0.3,0.6,0.6])
ax.imshow(histogram[0,:,:,]>1, aspect='auto', origin='lower',cmap=cm.bone_r)
ax1 = fig.add_axes([0.07,0.3,0.15,0.6], sharey=ax)
ax1.plot(np.sum(histogram[0,:,:], axis=1), f_vec)
ax2 = fig.add_axes([0.3,0.07,0.6,0.15], sharex=ax)
ax2.plot(t_vec, np.sum(histogram[0,:,:], axis=0))
plt.show()