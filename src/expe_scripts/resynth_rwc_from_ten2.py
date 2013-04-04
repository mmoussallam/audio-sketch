'''
expe_scripts.resynth_rwc_from_ten2  -  Created on Apr 4, 2013
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

def find_indexes(startIdx, array, stopvalue):
    """ get the indexes in the (sorted) array such that
    elements are smaller than value """
    idxset =[]
    idx = startIdx
    while idx <= array.shape[0]-1 and array[idx] < stopvalue:
        idxset.append(idx)
        idx +=1
#        print idx, array[idx]
    return idxset

def spec_morph(learn_specs, target_length, neighb_segments, l_seg_bounds):
    """ given a spectrogram and :
        - l_segments : list of (start, end) pairs of indices
        - neighb_segments :  list of (segment index, segment lengths)        
        do a morphing and build a candidate spectrogram """
    # total size of new spectrogram
    tstep = (learn_specs.shape[1]-1)    # Assume 50% overlap
    n_t_segments = len(neighb_segments)    
    # initialize new spectrogram
    morphed_spectro = (1e-5)*np.ones((target_length/tstep, learn_specs.shape[1]))
    cur_target_idx = 0  # current frame idx
    for segI in range(n_t_segments):
        ref_seg_idx = neighb_segments[segI][0]
        target_seg_length = neighb_segments[segI][1]
        # get indices of template frames
        ref_idx = range(l_seg_bounds[ref_seg_idx][0], l_seg_bounds[ref_seg_idx][1])      
        # get indices of target frames  
        target_idx = range(cur_target_idx, min(cur_target_idx + target_seg_length/tstep,  morphed_spectro.shape[0]))
        
        # now we need to compute the ratio of morphing        
        ratio = float(len(ref_idx))/float(len(target_idx))        
        # then 
        if ratio < 1:
            # case where we need to elongate the signal: use resize: duplicate end elements
            # TODO : maybe some other interpolation scheme would be better
            morphed_spectro[target_idx,:] = np.resize(np.abs(learn_specs[ref_idx,:]), (len(target_idx), learn_specs.shape[1]))
        else:
            # case where we need to compress the signal: build a redundant comb 
            morphed_spectro[target_idx,:] = np.abs(learn_specs[[ref_idx[int(j*ratio)] for j in range(len(target_idx))],:])             
        cur_target_idx += target_seg_length/tstep    
    return morphed_spectro
        
input_dir = '/sons/rwc/Learn/'
output_dir = '/sons/rwc/Learn/hdf5/'

audiofile = input_dir + 'rwc-g-m01_1.wav'
h5file = output_dir + 'rwc-g-m01_1.h5'

# load the Echo Nest features
h5 = hdf5_getters.open_h5_file_read(h5file)
timbre = hdf5_getters.get_segments_timbre(h5)
loudness_start = hdf5_getters.get_segments_loudness_start(h5)
C = hdf5_getters.get_segments_pitches(h5)
segments_all = hdf5_getters.get_segments_start(h5)

learn_feats_all = np.hstack((timbre,
                         loudness_start.reshape((loudness_start.shape[0],1)),
                        C))

# Ok That was the best possible case, now let us try to find the nearest neighbors, 
# get the segment back and resynthesize!


learn_duration = 200 # in seconds
test_start = 180
test_duration = 60

# Get learning data
learning = Signal(audiofile, mono=True)
learning.crop(0, learn_duration*learning.fs)

wsize = 1024
tstep = 512
# Get the magnitude spectrum for the given audio file
learn_specs = features.get_stft(learning.data, wsize, tstep)
learn_specs = learn_specs.T

max_l_seg_idx = np.where(segments_all < learn_duration)[0][-1]
l_segments = segments_all[:max_l_seg_idx]
l_segment_lengths = (l_segments[1:] - l_segments[0:-1])*learning.fs


learn_feats = learn_feats_all[:max_l_seg_idx,:]
# we must keep in mind for each segment index, the corresponding indices in the learn_spec mat
l_seg_bounds = []
ref_time = np.arange(0., float(learning.length)/float(learning.fs), float(tstep)/float(learning.fs))
for segI in range(len(l_segments)-1):
    startIdx = np.where(ref_time > l_segments[segI])[0][0]
    endIdx = np.where(ref_time > l_segments[segI+1])[0][0]
    l_seg_bounds.append((startIdx,endIdx))
l_seg_bounds.append((endIdx, ref_time.shape[0]))

# Get testing data
testing = Signal(audiofile, mono=True)
testing.crop(test_start*testing.fs, (test_start+test_duration)*learning.fs)

# get the testing features
min_t_seg_idx =  np.where(segments_all < test_start)[0][-1]
max_t_seg_idx =  np.where(segments_all < test_start + test_duration)[0][-1]
t_segments = segments_all[min_t_seg_idx:max_t_seg_idx]
t_segment_lengths = (t_segments[1:] - t_segments[0:-1])*testing.fs
test_feats = learn_feats_all[min_t_seg_idx:max_t_seg_idx,:]

# find the nearest neighbors
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(1)
# fit on the learning data
neigh.fit(learn_feats)
neighb_segments_idx = neigh.kneighbors(test_feats, return_distance=False)

# kneighs is a set of segment indices, we need to get the spectrogram back from the learning data
# then fit the new segment lengths

target_length = int(test_duration*testing.fs)

neighb_segments = zip(neighb_segments_idx[:,0], t_segment_lengths.astype(int))


###################### DEBUG ###################
# total size of new spectrogram
#tstep = (learn_specs.shape[1]-1)    # Assume 50% overlap
#n_t_segments = len(neighb_segments)    
## initialize new spectrogram
#morphed_spectro = (1e-5)*np.ones((target_length/tstep, learn_specs.shape[1]))
#cur_target_idx = 0  # current frame idx
#for segI in range(n_t_segments):
#    ref_seg_idx = neighb_segments[segI][0]
#    target_seg_length = neighb_segments[segI][1]
#    # get indices of template frames
#    ref_idx = range(l_seg_bounds[ref_seg_idx][0], l_seg_bounds[ref_seg_idx][1])      
#    # get indices of target frames  
#    target_idx = range(cur_target_idx, min(cur_target_idx + target_seg_length/tstep,  morphed_spectro.shape[0]))
#    
#    # now we need to compute the ratio of morphing        
#    ratio = float(len(ref_idx))/float(len(target_idx))        
#    # then 
#    if ratio < 1:
#        # case where we need to elongate the signal: use resize: duplicate end elements
#        # TODO : maybe some other interpolation scheme would be better
#        morphed_spectro[target_idx,:] = np.resize(np.abs(learn_specs[ref_idx,:]), (len(target_idx), learn_specs.shape[1]))
#    else:
#        # case where we need to compress the signal: build a redundant comb 
#        morphed_spectro[target_idx,:] = np.abs(learn_specs[[ref_idx[int(j*ratio)] for j in range(len(target_idx))],:])             
#    cur_target_idx += target_seg_length/tstep  
########################END DEBUG ######################
morphed_spectro = spec_morph(np.abs(learn_specs), target_length, neighb_segments, l_seg_bounds)


# retrieve true stft for comparison
test_specs = features.get_stft(testing.data, wsize, tstep)

plt.figure()
plt.subplot(121)
plt.imshow(np.log(np.abs(test_specs)), origin='lower')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log(morphed_spectro.T), origin='lower')
plt.colorbar()
plt.show()


init_vec = np.random.randn(morphed_spectro.shape[0]*tstep)
rec_method2 = transforms.gl_recons(morphed_spectro.T, init_vec, 10, wsize, tstep, display=False)
rec_sig_2 = Signal(rec_method2, testing.fs, mono=True, normalize=True)
rec_sig_2.write('/sons/tests/rec_sig2.wav')

