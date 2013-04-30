'''
expe_scripts.resynth_features_constrain_selection  -  Created on Apr 10, 2013
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

from tools.learning_tools import find_indexes, get_ten_features, get_ten_features_from_file, get_track_info
from feat_invert.transforms import spec_morph

outputpath = '/home/manu/workspace/audio-sketch/src/results/audio'

def save_audio(learntype, np, test_file, n_feat, rescale_str, sigout,  fs, norm_segments=False):
    """ saving output vector to an audio wav"""
    norm_str = ''
    if norm_segments:
        norm_str = 'normed'
        mean_energy = np.mean([np.sum(sig**2)/float(len(sig)) for sig in sigout])
        for sig in sigout:
            sig /= np.sum(sig**2)/float(len(sig))
            sig *= mean_energy        
    rec_sig = Signal(np.concatenate(sigout), fs, normalize=True)
    rec_sig.write('%s/%s_with%s_%dfeats_%s%s.wav' % (outputpath,
                                                   os.path.split(test_file)[-1],
                                                   learntype, n_feat,
                                                   rescale_str,
                                                   norm_str))
    
def save_audio_full_ref(learntype, test_file, n_feat, rescale_str, sigout, fs, norm_segments=False):
    """ do not cut the sounds """
    # first pass for total length
    max_idx = int(sigout[-1][1] + len(sigout[-1][0])) + 4*fs
    print "total length of ",max_idx
    sig_data = np.zeros((max_idx,))
#    seg_energy = np.sum(sigout[-1][0]**2)
    for (sig, startidx) in sigout:
#        print sig.shape, sig_data[int(startidx):int(startidx)+sig.shape[0]].shape
        sig_data[int(startidx):int(startidx)+sig.shape[0]] += sig#*seg_energy/np.sum(sig**2)
        
    rec_sig = Signal(sig_data, fs, normalize=True)
    rec_sig.write('%s/%s_with%s_%dfeats_%s%s.wav' % (outputpath,
                                                   os.path.split(test_file)[-1],
                                                   learntype, n_feat,
                                                   rescale_str,
                                                   'full_ref'))
    
    
################# EXPE 1 ###########################
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
#test_file = '/sons/rwc/Learn/hdf5/rwc-g-m01_1.h5'

# Beethoven n7 - Karayan
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/R/V/TRARVJE128F93127EF.h5'

# Rolling Stones Angie
test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/A/D/TRAADLN128F14832E9.h5'

#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/R/A/TRARAAG128F42437FB.h5'
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/Z/W/TRAZWGK128F93141E3.h5'
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/D/D/TRADDXS12903CEDB38.h5'
#test_file = '/home/manu/workspace/databases/MillionSongSubset/data/A/R/T/TRARTEH128F423DBC1.h5'

#test_file = '/home/manu/workspace/databases/MillionSongSubset/data/A/D/H/TRADHZN128F428DA0D.h5'

test_feats_list = []
test_segs_list = []
test_confidence_list = []
get_ten_features_from_file(test_feats_list, test_segs_list, test_confidence_list, test_file)

test_feats = test_feats_list[0]
test_segs = test_segs_list[0][0]
test_confidence = np.concatenate(test_confidence_list) 

title, artist = get_track_info(test_file)
print title, artist 
t_seg_duration = np.diff(test_segs)

# Do the nearest neighbor search : limit to 5 best?
n_neighbs = 1
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbs)

# fit on the learning data
n_feat = 24
if n_feat == 13:
    neigh.fit(learn_feats[:,0:12])
    neighb_segments_idx = neigh.kneighbors(test_feats[:,0:12], return_distance=False)
if n_feat == 24:
    neigh.fit(np.hstack([learn_feats[:,0:12],learn_feats[:,-12:]]))
    neighb_segments_idx = neigh.kneighbors(np.hstack([test_feats[:,0:12],test_feats[:,-12:]]),
                                           return_distance=False)
elif n_feat in [12,15]:
    neigh.fit(learn_feats[:,-n_feat:])
    distance, neighb_segments_idx = neigh.kneighbors(test_feats[:,-n_feat:], return_distance=True)
else:
    neigh.fit(learn_feats)
    neighb_segments_idx = neigh.kneighbors(test_feats, return_distance=False)    

# We need a routine that takes a reference signal, the original segment bounds and the targeted length
from feat_invert.transforms import get_audio, time_stretch

max_synth_idx = len(neighb_segments_idx)-5
max_synth_idx = 10
rescale = True
rescale_str = ''

# FIRST OF ALL: FORCE COHERENCE OF SEGMENT LENGTHS
sigout = []

# We set an arbitrary penalty cost as half the standard deviation of the distances
# between first and second candidates
# so if best candidate is highly more representative, penalty will not be sufficient

total_target_duration = 0
for test_seg_idx in range(max_synth_idx):
    print "----- %d/%d ----"%(test_seg_idx,max_synth_idx)
    target_audio_duration = t_seg_duration[test_seg_idx]  
    total_target_duration += target_audio_duration 
    
    
    length_ratios = np.zeros(n_neighbs)
    ref_seg_idx = []
    ref_audio_path = []    
    ref_audio_start = np.zeros(n_neighbs)
    ref_audio_duration = np.zeros(n_neighbs)
    for num_neigh in range(n_neighbs):
        ref_seg_idx.append(ref_seg_indices[neighb_segments_idx[test_seg_idx][num_neigh]])
        ref_audio_path.append(learn_segs[ref_seg_idx[-1],1])
        ref_audio_start[num_neigh] = l_seg_start[neighb_segments_idx[test_seg_idx][num_neigh]]
        ref_audio_duration[num_neigh] = l_seg_duration[neighb_segments_idx[test_seg_idx][num_neigh]]    

        length_ratios[num_neigh] = float(ref_audio_duration[num_neigh])/float(target_audio_duration)

    # Compare situation where the best candidate is used, and the one where the 
    # most timely related candidate is used
    filepath = ref_audio_dir + ref_audio_path[0] + ext
    signalin, fs = get_audio(filepath, ref_audio_start[0], ref_audio_duration[0])
    target_length = target_audio_duration*fs
    print "Loaded %s length of %d "%( filepath, len(signalin))
    print "Stretching to %2.2f"%length_ratios[0]
    if length_ratios[0]<1.0:
        sigout.append((signalin, test_segs[test_seg_idx]*fs))
    else:
        sigout.append((time_stretch(signalin, length_ratios[0], wsize=1024, tstep=128)[128:-1024], test_segs[test_seg_idx]*fs))
        
save_audio_full_ref(learntype,  test_file, n_feat, '_full_ref_', sigout, fs, norm_segments=False)
'''
expe_scripts.resynth_ten_full_ref_sample  -  Created on Apr 29, 2013
@author: M. Moussallam
'''

## segments and durations
#plt.figure()
#for segIdx in range(10):
#    print test_segs[segIdx], t_seg_duration[segIdx]
#    plt.axvspan(test_segs[segIdx], test_segs[segIdx]+ t_seg_duration[segIdx], 0, 1,facecolor='g', alpha=0.5)
#    
#plt.show()
