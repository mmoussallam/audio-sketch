'''
expe_scripts.invert.visu_cross_synthesis  -  Created on May 15, 2013
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
from PyMP import Signal
import sys
import os
from feat_invert import regression, transforms, features
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
sys.path.append('/usr/local/lib')
sys.path.append('/home/manu/workspace/toolboxes/MSongsDB-master/PythonSrc')
sys.path.append('/usr/local/python_packages')
import hdf5_utils as HDF5
import hdf5_getters
import stft

from tools.learning_tools import resynth_sequence, get_ten_features, get_ten_features_from_file, get_track_info
from feat_invert.transforms import spec_morph

################# EXPE 2 ###########################
learntype = 'Piano'
ext = '.WAV'
ref_audio_dir = '/sons/rwc/%s/'%learntype
h5_dir =  '/sons/rwc/%s/hdf5/'%learntype
# load the learned features and segments
# for each segment, we must keep a trace of the original corresponding file
learn_feats, learn_segs, learn_confidence = get_ten_features(h5_dir)
(n_seg, n_feats) = learn_feats.shape
ref_seg_indices = np.zeros(n_seg,int)
l_seg_duration = np.zeros(n_seg)
l_seg_start = np.zeros(n_seg)
c_idx = 0
for segI in range(learn_segs.shape[0]):
    n_ub_seg = len(learn_segs[segI,0])
    ref_seg_indices[c_idx:c_idx+n_ub_seg] = segI
    l_seg_start[c_idx:c_idx+n_ub_seg] = learn_segs[segI,0]
    l_seg_duration[c_idx:c_idx+n_ub_seg-1] = learn_segs[segI,0][1:] - learn_segs[segI,0][0:-1]
    c_idx += n_ub_seg

# now let us take an example from the test dataset (or any file in the MSDataSet)
test_file = 'blues.00056'
dir_path = '/home/manu/workspace/databases/genres/blues'
extref = 'au'
#dir_path = '/sons/rwc/Learn/'
#ext = '.WAV'
#h5_file_path = '/sons/rwc/Learn/hdf5/rwc-g-m01_4.h5'
#audio_file_path = '/sons/rwc/Learn/rwc-g-m01_4.wav'
h5_file_path = '%s/hdf5/%s.h5'%(dir_path,test_file)
audio_file_path = '%s/%s.%s'%(dir_path,test_file,extref)
#test_file = '/sons/rwc/Learn/hdf5/rwc-g-m01_4.h5'

test_feats_list = []
test_segs_list = []
test_confidence_list = []
get_ten_features_from_file(test_feats_list, test_segs_list, test_confidence_list, h5_file_path)

test_feats = test_feats_list[0]
test_segs = test_segs_list[0][0]
test_confidence = np.concatenate(test_confidence_list) 

#title, artist = get_track_info(test_file)
#print title, artist 
t_seg_duration = np.diff(test_segs)

# Do the nearest neighbor search
n_neighbs = 1
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbs)

# fit on the learning data
n_feat = 12
if n_feat == 13:
    neigh.fit(learn_feats[:,0:12])
    neighb_segments_idx = neigh.kneighbors(test_feats[:,0:12], return_distance=False)
if n_feat == 24:
    neigh.fit(np.hstack([learn_feats[:,0:12],learn_feats[:,-12:]]))
    neighb_segments_idx = neigh.kneighbors(np.hstack([test_feats[:,0:12],test_feats[:,-12:]]), return_distance=False)
elif n_feat == 12:
    neigh.fit(learn_feats[:,-n_feat:])
    neighb_segments_idx = neigh.kneighbors(test_feats[:,-n_feat:], return_distance=False)
else:
    neigh.fit(learn_feats)
    neighb_segments_idx = neigh.kneighbors(test_feats, return_distance=False)    

# We need a routine that takes a reference signal, the original segment bounds and the targeted length

from feat_invert.transforms import get_audio, time_stretch

max_synth_idx = len(neighb_segments_idx)-5
max_synth_idx = 17
rescale = True
stretch = True

#vit_path = Viterbi(neigh, distance, t_penalty=0.01, c_value=20)
#vit_cands = [neigh[ind,neighbind] for ind, neighbind in enumerate(vit_path)]
#

sig_out =  resynth_sequence(neighb_segments_idx[:,0], test_segs, t_seg_duration, 
            learn_segs, learn_feats, ref_audio_dir, ext, 22050,
            dotime_stretch=False, max_synth_idx=max_synth_idx, normalize=False,
            marge=3, verbose=True)

sig_out_normalized = resynth_sequence(neighb_segments_idx[:,0], test_segs, t_seg_duration, 
            learn_segs, learn_feats, ref_audio_dir, ext, 22050,
            dotime_stretch=True, max_synth_idx=max_synth_idx, normalize=True,
            marge=3, verbose=True)



#sig_viterbi = Signal(sig_out_viterbi, 22050, normalize=True)

rec_sig = Signal(sig_out, 22050, normalize=True)
rec_sig.crop(0, test_segs[max_synth_idx]*rec_sig.fs)

rec_sig_normalized = Signal(sig_out_normalized, 22050, normalize=True)
rec_sig_normalized.crop(0, test_segs[max_synth_idx]*rec_sig.fs)
# load original audio
orig_data, fs = get_audio(audio_file_path, 0, rec_sig.get_duration(),
                          targetfs=None, verbose=True)



orig_sig = Signal(orig_data, fs, normalize=True)
Lmax = min(orig_sig.length,rec_sig.length)
t_vec = np.arange(float(Lmax))/float(fs)
plt.figure(figsize=(8,8))
ax1 = plt.subplot(311)
plt.plot(t_vec,orig_sig.data[:Lmax])
plt.stem(test_segs[:max_synth_idx], 0.8*np.ones((max_synth_idx,)), linefmt='k-', markerfmt='s')
plt.subplot(312, sharex=ax1, sharey=ax1)
#plt.xticks([])
plt.grid(axis='x')
plt.plot(t_vec,rec_sig.data[:Lmax])
plt.stem(test_segs[:max_synth_idx], 0.8*np.ones((max_synth_idx,)), linefmt='k-', markerfmt='s')
#plt.xticks([])
plt.grid(axis='x')
plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot(t_vec,rec_sig_normalized.data[:Lmax])
plt.stem(test_segs[:max_synth_idx], 0.8*np.ones((max_synth_idx,)), linefmt='k-', markerfmt='s')
plt.xlim((-0.1,rec_sig.get_duration()+0.1))
plt.ylim((-1,1))
plt.xlabel('Time (s)', fontsize=16.0)
plt.grid(axis='x')
plt.subplots_adjust(left=0.07, top=0.97,right=0.97,hspace=0.04)
plt.savefig('/home/manu/Documents/Articles/ISMIR2013/ListeningMSD/Figures/visu_concatenative_synth%s.pdf'%learntype)
plt.show()


####### Now using the Viterbi smoothing
n_neighbs = 20
neigh = NearestNeighbors(n_neighbs)
neigh.fit(learn_feats)
distance, neighb_segments_idx = neigh.kneighbors(test_feats, return_distance=True)    
from tools.learning_tools import Viterbi
vit_path = Viterbi(neighb_segments_idx, distance, t_penalty=0.01, c_value=20)
vit_cands = [neighb_segments_idx[ind,neighbind] for ind, neighbind in enumerate(vit_path)]

sig_out_viterbi = resynth_sequence(np.squeeze(vit_cands), test_segs, t_seg_duration,
                           learn_segs, learn_feats, ref_audio_dir, ext, 22050,
                           dotime_stretch=True,max_synth_idx=max_synth_idx,  normalize=True,
                           marge=3, verbose=True)
sig_viterbi = Signal(sig_out_viterbi, 22050, normalize=True)

