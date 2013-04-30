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

from tools.learning_tools import find_indexes, get_ten_features, get_ten_features_from_file, get_track_info
from feat_invert.transforms import spec_morph

def expe_1_synth_from_same_sample():        
    input_dir = '/sons/rwc/Learn/'
    output_dir = '/sons/rwc/Learn/hdf5/'
    
    audiofile = input_dir + 'rwc-g-m01_1.wav'
    h5file = output_dir + 'rwc-g-m01_1.h5'
    
    # load the Echo Nest features
    h5 = hdf5_getters.open_h5_file_read(h5file)
    timbre = hdf5_getters.get_segments_timbre(h5)
    loudness_start = hdf5_getters.get_segments_loudness_start(h5)
    loudness_max = hdf5_getters.get_segments_loudness_max(h5)
    loudness_max_time = hdf5_getters.get_segments_loudness_max_time(h5)
    C = hdf5_getters.get_segments_pitches(h5)
    segments_all = hdf5_getters.get_segments_start(h5)
    
    learn_feats_all = np.hstack((timbre,
                             loudness_start.reshape((loudness_start.shape[0],1)),
                            C))
    
    # Ok That was the best possible case, now let us try to find the nearest neighbors, 
    # get the segment back and resynthesize!
    
    
    learn_duration = 200 # in seconds
    test_start = 200
    test_duration = 5
    
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
#test_file = '/sons/rwc/Learn/hdf5/rwc-g-m01_1.h5'
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/B/E/TRABEMC128F148EF2D.h5'
test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/A/D/TRAADLN128F14832E9.h5'
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/R/A/TRARAAG128F42437FB.h5'
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/Z/W/TRAZWGK128F93141E3.h5'
#test_file ='/home/manu/workspace/databases/MillionSongSubset/data/A/D/D/TRADDXS12903CEDB38.h5'
#test_file = '/home/manu/workspace/databases/MillionSongSubset/data/A/R/T/TRARTEH128F423DBC1.h5'

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
max_synth_idx = 200
rescale = True
rescale_str = ''
#plt.figure()
#plt.plot(neighb_segments_idx)
#plt.show()
sigout = []
for num_neigh in range(n_neighbs):
    sigout.append([])

total_target_duration = 0
for test_seg_idx in range(max_synth_idx):
    for num_neigh in range(n_neighbs):
        ref_seg_idx = ref_seg_indices[neighb_segments_idx[test_seg_idx][num_neigh]]
        ref_audio_path = learn_segs[ref_seg_idx,1]
        ref_audio_start = l_seg_start[neighb_segments_idx[test_seg_idx][num_neigh]]
        ref_audio_duration = l_seg_duration[neighb_segments_idx[test_seg_idx][num_neigh]]    
        
#        ref_confidence = learn_confidence[neighb_segments_idx[test_seg_idx][num_neigh]]
#        test_confidence = test_confidence[test_seg_idx]
        
        target_audio_duration = t_seg_duration[test_seg_idx]  
        total_target_duration += target_audio_duration          
        
        # no need to worry about less than 1ms samples
        if ref_audio_duration < 0.001 or target_audio_duration < 0.001:
            continue
        
        tscale = float(ref_audio_duration)/float(target_audio_duration)
        print "seg %d / %dref is %d time scaling of %2.2f"%(test_seg_idx,len(neighb_segments_idx),
                                                        ref_seg_idx,
                                                        tscale)    
#        print "WARNING CHANGING THE EXTENSION!!"
        filepath = ref_audio_dir + ref_audio_path + ext
        print "Loading ", filepath
        signalin, fs = get_audio(filepath, ref_audio_start, ref_audio_duration)
        target_length = target_audio_duration*fs
        print "Loaded %s length of %d "%( filepath, len(signalin))
        print "Stretching to %2.2f"%target_length
        
        # adjust the Loudness ?
        if rescale:
            rescale_str = 'normed'
            signalin = signalin.astype(float)
            signalin /= 8192.0
            signalin /= np.max(signalin)
    #        N = float(len(signalin))
    #        target_loudness = test_feats[test_seg_idx, 13]
    #        adjust = target_loudness - 10*np.log10((1.0/N)*np.sum(signalin**2))
    #        signalin *= 10**(adjust/10.)
            signalin *= 8192.0
            signalin = signalin.astype(np.int16)
        sigout[num_neigh].append(time_stretch(signalin, tscale, wsize=1024, tstep=128)[128:-1024])


for num_neigh in range(n_neighbs):
    rec_sig = Signal(np.concatenate(sigout[num_neigh]), fs, normalize=True)
    rec_sig.write('/home/manu/workspace/audio-sketch/src/results/audio/%s_with%s_%dfeats_%s_neighbor_%d.wav'%(
                                                    os.path.split(test_file)[-1],
                                                    learntype,
                                                    n_feat,
                                                    rescale_str,
                                                    num_neigh))

