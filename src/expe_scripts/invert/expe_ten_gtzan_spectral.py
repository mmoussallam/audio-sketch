'''
expe_scripts.invert.expe_ten_gtzan_spectral  -  Created on May 7, 2013

Synthesis by taking the median of various magnitude spectrograms 

@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyMP import Signal
import sys
import os
from feat_invert import regression, transforms, features
from sklearn.neighbors import NearestNeighbors
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
from tools.learning_tools import resynth_sequence, save_audio

from expe_scripts.invert.com_imports import get_learns_and_test 

ref_audio_dir = '/home/manu/workspace/databases/genres/blues/'
outputpath = '/home/manu/workspace/audio-sketch/src/expe_scripts/audio/feat_invert'
data_path = '/home/manu/workspace/databases/genres/blues/hdf5/' # List all the files

# Global parameters
t_index = 100
filter_key = False
n_learn_max = 99
nbFeats = 27
max_synth_idx = 80
n_neighbs = 3

# loading learn and test base
args = get_learns_and_test(ref_audio_dir,
                             data_path,
                             t_index = t_index,
                             filter_key= filter_key,
                              n_learn_max = n_learn_max)  

l_feats, t_feats, t_seg_starts, t_seg_duration, l_segments, h5files, n_learn = args


knn = NearestNeighbors(n_neighbors=n_neighbs)    
knn.fit(l_feats[:,-nbFeats:])
distance, neigh = knn.kneighbors(t_feats[:,-nbFeats:], n_neighbors=n_neighbs, return_distance=True)


# first method: build 10 waveforms then stft and median of magnitude
magspecs = []
for neighIdx in range(neigh.shape[1]):
    # build waveform
    sigout = resynth_sequence(np.squeeze(neigh[:,neighIdx]), t_seg_starts, t_seg_duration,
                           l_segments, l_feats, ref_audio_dir, '.au', 22050,
                           dotime_stretch=True, max_synth_idx=max_synth_idx,  normalize=False)
    # stft
    magspecs.append(np.abs(stft.stft(sigout,512,128)[0,:,:]))

# take the median
magspecarr = np.array(magspecs)
#mean_magspec = np.mean(magspecarr, 0)
median_magspec = np.median(magspecarr, 0)
#min_magspec = np.mean(magspecarr, 0)
# load true data
orig =  Signal(ref_audio_dir + h5files[t_index-1][:-3] + '.au', normalize=False)

# Display the spectrograms
#plt.figure()
#plt.subplot(311)
#plt.imshow(np.log(mean_magspec), origin='lower')
#plt.subplot(312)
#plt.imshow(np.log(median_magspec), origin='lower')
#plt.subplot(313)
#plt.imshow(np.log(min_magspec), origin='lower')
#orig.spectrogram(512, 128, order=1, log=True, cmap=cm.jet, cbar=False)
#plt.show()

# Resynthesize using GL
#init_vec = np.random.randn(128*mean_magspec.shape[1])
#x_recon_mean = transforms.gl_recons(np.maximum(mean_magspec, 1e-10), init_vec, 50,
#                                   512, 128, display=False)
#
#sig_mean= Signal(x_recon_mean, orig.fs, normalize=True)
#sig_mean.write('%s/%s_spectral_%dmeans_%dFeats_%dLearns_Filter%d.wav' % (outputpath,h5files[t_index-1],
#                                                                   n_neighbs,nbFeats,n_learn,filter_key))

#sig_mean.play(prevent_too_long=False)

## same for median
init_vec = np.random.randn(128*median_magspec.shape[1])
x_recon_median = transforms.gl_recons(np.maximum(median_magspec, 1e-10), init_vec, 30,
                                   512, 128, display=False)

sig_median= Signal(x_recon_median, orig.fs, normalize=True)
sig_median.write('%s/%s_spectral_%dmedians_%dFeats_%dLearns_Filter%d.wav' % (outputpath,h5files[t_index-1],
                                                                   n_neighbs,nbFeats,n_learn,filter_key))


## same for min
#init_vec = np.random.randn(128*min_magspec.shape[1])
#x_recon_min = transforms.gl_recons(np.maximum(min_magspec, 1e-10), init_vec, 50,
#                                   512, 128, display=False)
#
#sig_min= Signal(x_recon_min, orig.fs, normalize=True)
#sig_min.write('%s/%s_spectral_%dmin_%dFeats_%dLearns_Filter%d.wav' % (outputpath,h5files[t_index-1],
#                                                                   n_neighbs,nbFeats,n_learn,filter_key))

#sig_median.play(prevent_too_long=False)