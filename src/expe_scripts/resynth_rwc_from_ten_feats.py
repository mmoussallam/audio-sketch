#encoding <utf-8>
'''
expe_scripts.resynth_rwc_from_ten_feats  -  Created on Apr 4, 2013
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

input_dir = '/sons/rwc/Learn/'
output_dir = '/sons/rwc/Learn/hdf5/'

audiofile = input_dir + 'rwc-g-m01_1.wav'
h5file = output_dir + 'rwc-g-m01_1.h5'

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
    

original = Signal(audiofile, mono=True)

max_duration = 20 # in seconds
original.crop(0, max_duration*original.fs)

wsize = 1024
tstep = 512

# Get the magnitude spectrum for the given audio file
learn_specs = features.get_stft(original.data, wsize, tstep)
learn_specs = learn_specs.T
# Read the features in the h5 file
h5 = hdf5_getters.open_h5_file_read(h5file)
timbre = hdf5_getters.get_segments_timbre(h5)
loudness_start = hdf5_getters.get_segments_loudness_start(h5)
C = hdf5_getters.get_segments_pitches(h5)
segments_all = hdf5_getters.get_segments_start(h5)

learn_feats_all = np.hstack((timbre,
                         loudness_start.reshape((loudness_start.shape[0],1)),
                        C))


max_seg_idx = np.where(segments_all < max_duration)[0][-1]
segments = segments_all[:max_seg_idx]

# now we have a 30202 x 512 matrix of magnitude spectrums and a 1147 x1 vector or time positions for 
# them.
# How can we combine the spectrums in order to resynthesize the audio?


ref_time = np.arange(0., float(original.length)/float(original.fs), float(tstep)/float(original.fs))

find_indexes(0, ref_time, 0.1)

# FIRST we need to build mean spectrums for the time slots defined by the segmentation
# so that each feature slice corresponds to a single spectrum
averaged_specs = np.zeros((segments.shape[0], learn_specs.shape[1]))
medianed_specs = np.zeros((segments.shape[0], learn_specs.shape[1]))
#x_filt = filter(lambda x: (0.12<x and 0.16>x), [0.08,0.11,0.121,0.145,0.161,0.8])
start_idx = 0
spec_idx = []
for i in range(1,segments.shape[0]):    
    spec_idx.append(find_indexes(start_idx, ref_time, segments[i]))
    if len(spec_idx[-1]) >0:
        start_idx = spec_idx[-1][-1]
    print "Segment %d index %d to %d"%(i,spec_idx[-1][0],spec_idx[-1][-1])
    averaged_specs[i,:] = np.mean(np.abs(learn_specs[spec_idx[-1],:]), axis=0)
    medianed_specs[i,:] = np.median(np.abs(learn_specs[spec_idx[-1],:]), axis=0)

plt.figure()
plt.subplot(131)
plt.imshow(np.log(np.abs(learn_specs.T)), origin='lower')
plt.subplot(132)
plt.imshow(np.log(averaged_specs.T), origin='lower')
plt.subplot(133)
plt.imshow(np.log(medianed_specs.T), origin='lower')
plt.show()

# Ok Now TO RECONSTRUCT we need to take the averaged spectrum for each segment
# and do a time stretching so as to reconstruct something that roughly has the
# rightful length... or duplicate the spectrums until it has the appropriate 
# length?

# METHOD 1 : reverse construct a full sized spectrogram from the averaged one
# based on the averaged spectrums : we should simulate a 50% overlap no?
reconstructed_averaged_specs = np.zeros_like(np.abs(learn_specs))
start_idx = 0
for i in range(1,segments.shape[0]):    
    indexes = find_indexes(start_idx, ref_time, segments[i])
    if len(indexes) >0:
        start_idx = indexes[-1]    
    reconstructed_averaged_specs[indexes,:] = averaged_specs[i,:].reshape(1, (wsize/2)+1).repeat(len(indexes),0)
    
plt.figure()
plt.subplot(131)
plt.imshow(np.log(np.abs(learn_specs.T)), origin='lower')
plt.subplot(132)
plt.imshow(np.log(averaged_specs.T), origin='lower')
plt.subplot(133)
plt.imshow(np.log(reconstructed_averaged_specs.T), origin='lower')
plt.show()

# time for resynthesis
init_vec = np.random.randn(original.data.shape[0])
rec_method1 = transforms.gl_recons(reconstructed_averaged_specs.T, init_vec, 10, wsize, tstep, display=False)
rec_sig = Signal(rec_method1, original.fs, mono=True, normalize=True)
# CONCLUSION: WE HAVE KEPT ONLY ONE SPECTRUM PER "NOTE" : DIGITALIZED SOUND

# METHOD 2: use the original waveform to resynthesize
# We have the segmentation and for each segment we have a waveform
# we should directly use the "potentially time-extended" original waveform
# to resynthesize !
#waveform_list = []
#for i in range(1,segments.shape[0]):
#    waveform_list.append(original.data)
# This is a little too easy in this context, we should try it after the nearest neighbors search

# METHOD 3 for each feature bag, we have a M x F spectogram (or magspec)
# given this a new feature bag, we need to resynthesize something that has a Ms x F spetcrogram
# which is a stretched version of the MxF one

#def morph(magspec, ref_idx, dist_idx):
    
    

# for practice, let us assume the ratio to be slightly changing from 0.9 to 1.2 for each segment
#segment_random_ratio = 0.9 + 0.3*np.random.rand(segments.shape[0]-1)
# perfect case
segment_random_ratio = np.ones((segments.shape[0]-1),)
# the new total duration is now:
distorted_segments = np.cumsum(segment_random_ratio * (segments[1:]-segments[0:-1]))
distorted_segments = np.concatenate((np.array([0]), distorted_segments))

nb_seg_distorted = np.max(find_indexes(0, ref_time, distorted_segments[-1]))-1
distorted_spec = np.zeros((nb_seg_distorted,(wsize/2)+1 ))
# now for each segment we need to build a spectr
start_dist_idx = 0
start_ref_idx = 0
for segI in range(1,nb_seg_distorted):
    ref_idx = find_indexes(start_ref_idx, ref_time, segments[segI])
    dist_idx = find_indexes(start_dist_idx, ref_time, distorted_segments[segI])
    ratio = float(len(ref_idx))/float(len(dist_idx))        
    if len(dist_idx) >0 and len(ref_idx) >0:        
        start_ref_idx = ref_idx[-1]
        start_dist_idx = dist_idx[-1]
        new_spec = np.zeros((len(dist_idx), (wsize/2)+1))        
        if ratio < 1:
            new_spec = np.resize(np.abs(learn_specs[ref_idx,:]), (len(dist_idx), (wsize/2)+1))
        else:
            if dist_idx[-1]*ratio > distorted_spec.shape[0]:
                break
            new_spec = np.abs(learn_specs[[int(j*ratio) for j in dist_idx],:])
        distorted_spec[dist_idx,:] = new_spec
    

plt.figure()
plt.imshow(np.log(distorted_spec.T), origin='lower')
plt.show()

# resynthesize
init_vec = np.random.randn(distorted_spec.shape[0]*tstep)
rec_method2 = transforms.gl_recons(distorted_spec.T, init_vec, 10, wsize, tstep, display=False)
rec_sig_2 = Signal(rec_method2, original.fs, mono=True, normalize=True)
rec_sig_2.write('/sons/tests/rec_sig2.wav')
# let us see if the spectrum joined together look alike
#seg_idx = 10
#plt.figure()
#plt.subplot(121)
#plt.imshow(np.log(np.abs(learn_specs[spec_idx[seg_idx],:]).T))
#plt.subplot(122)
#plt.plot(np.median(np.log(np.abs(learn_specs[spec_idx[seg_idx],:]).T), axis=1))
#plt.plot(np.log(np.mean(np.abs(learn_specs[spec_idx[seg_idx],:].T), axis=1)),'k')
#plt.plot(np.mean(np.log(np.abs(learn_specs[spec_idx[seg_idx],:]).T), axis=1),'r')
#plt.show()



