'''
expe_scripts.invert.expe_detec_onset  -  Created on Apr 30, 2013
Remove segment starts that appears not to be real onsets
based on the Loudness Delta 
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

import stft

from tools.learning_tools import find_indexes, get_ten_features, get_ten_features_from_file, get_track_info, resynth
from feat_invert.transforms import spec_morph



def get_candidate(chroma_neigh, feats, n_feats, learn_feats, segIdx, lambda_L,
                  forceAttack=False, thresh_lambda=7):
    """ return a candidate with or without penalizing the Delte Loudness """
    distance, cands = chroma_neigh.kneighbors(test_feats[segIdx,-n_feats:], return_distance=True)
    cands = np.squeeze(np.array(cands))
    distance = np.squeeze(np.array(distance))
    print "loudness: ", feats[0][segIdx,12:15] 
    DeltaL = feats[0][segIdx,13] -  feats[0][segIdx,12]
    n_distance = distance
    for c in range(n_distance.shape[0]):
        print "Cand %d: "%c, learn_feats[cands[c],12:15], distance[c]
        cand_DeltaL = learn_feats[cands[c],13] - learn_feats[cands[c],12]        
        n_distance[c] += lambda_L*np.abs(cand_DeltaL - DeltaL)
        if cand_DeltaL < thresh_lambda and forceAttack:
            n_distance[c] = 0
    b_c = np.argmin(n_distance)
    print "New best candidate is %d score of %1.4f"%(b_c, n_distance[b_c]) , n_distance
    return cands[b_c], n_distance[b_c]
    
    
# load the audio data and the features
audio_file_path = '/sons/rwc/Learn/rwc-g-m01_1.wav'
output_path = '/home/manu/workspace/audio-sketch/src/results/audio'
orig_sig = Signal(audio_file_path)

test_file = 'rwc-g-m01_1'
h5_file_path = '/sons/rwc/Learn/hdf5/rwc-g-m01_1.h5'
feats = []
segs = []
get_ten_features_from_file(feats, segs, [], h5_file_path)

# plot part of the audio and teh segmentation
seg_starts = segs[0][0]
seg_duration = np.diff(seg_starts)

nseg = 100
max_time = seg_starts[nseg] + seg_duration[nseg]
fs = orig_sig.fs

test_feats = feats[0][0:nseg,:]

# Load the learned features
learntype = 'Piano'
ext = '.WAV'
ref_audio_dir = '/sons/rwc/%s/'%learntype
h5_dir =  '/sons/rwc/%s/hdf5/'%learntype
# load the learned features and segments
# for each segment, we must keep a trace of the original corresponding file
learn_feats, learn_segs, learn_confidence = get_ten_features(h5_dir)

# get the candidates accodring to the chroma features
from sklearn.neighbors import NearestNeighbors
n_neighbs_chroma = 5
chroma_neigh = NearestNeighbors(n_neighbs_chroma)
n_feats = 12
lambda_L = 0.01
chroma_neigh.fit(learn_feats[:,-n_feats:])


thresh_lambda = 0 # In db the threshold above which a segment is considered to contain an attack
notanattack = []
for segIdx in range(nseg):
    print "loudness: ", feats[0][segIdx,12:15] 
    DeltaL = feats[0][segIdx,13] -  feats[0][segIdx,12]
    if DeltaL < thresh_lambda:
        print "Seg %d is not an attack : removing"%segIdx
        notanattack.append(segIdx)
    

# for all segments : get the nearest neighbor if and only if it's an attack
current_duration = 0.0
real_seg_starts = []
real_seg_durations = []
candidates = []

ongoing = False
dist = np.PINF
current_cand = None
for segIdx in range(nseg):    
    # It's an attack: get the candidate
    if segIdx not in notanattack:            
        # finish preceeding segment unless it's the first one
        if segIdx > 0:
            real_seg_durations.append(current_duration)
            current_duration = 0.0                    
            candidates.append(current_cand)
            dist = np.PINF
            
        # let's add a new candidate and associated distance
        newcand, newdist = get_candidate(chroma_neigh, feats,n_feats, learn_feats, 
                                         segIdx, lambda_L, forceAttack=False)        
        # if distance is below 
        if current_cand is None or newdist < dist:
            print "First candidate"
            dist = newdist
            current_cand = newcand
        
        real_seg_starts.append(seg_starts[segIdx])
        current_duration += seg_duration[segIdx]
        # finish ongoing segment if any
        if ongoing:                        
            ongoing = False                
    else:
        if not ongoing:
            # start an ongoing segment
            ongoing =True
        current_duration += seg_duration[segIdx]
        newcand, newdist = get_candidate(chroma_neigh, feats,n_feats, learn_feats,
                                         segIdx, lambda_L, forceAttack=False)        
        # if distance is below 
        if current_cand is None or newdist < dist:
            print "New candidate"
            dist = newdist
            current_cand = newcand
        
#when loop is over if ongoing segment: close it
real_seg_durations.append(current_duration)
candidates.append(current_cand)

candidates = np.squeeze(np.array(candidates))
sigout_onlyattacks =  resynth(candidates, real_seg_starts, real_seg_durations, learn_segs,
                  learn_feats, ref_audio_dir, ext,
                  dotime_stretch=True, normalize=True)
aud_str = "%s_%s_chroma_attacks_%1.2flambdaL_thresh_%1.1f_%dfeats_%dneighbs"%(learntype,
                                                                              test_file, lambda_L,
                                                                              thresh_lambda,
                                                                              n_feats,
                                                                              n_neighbs_chroma)
from tools.learning_tools import save_audio
save_audio(output_path, aud_str, sigout_onlyattacks, fs, norm_segments=False)  