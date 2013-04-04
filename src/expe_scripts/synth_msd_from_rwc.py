'''
expe_scripts.synth_msd_from_rwc  -  Created on Apr 3, 2013

Ok so we have a collection of spectrums and associated chromas and mfccs
Can we reconstruct a file from the Million Song Dataset given its own chroma/mfccs using machine learning
and no reverse signal synthesis

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
sys.path.append('/home/manu/workspace/toolboxes/MSongsDB-master/PythonSrc')
sys.path.append('/usr/local/lib')
from feat_invert.features import load_data_one_audio_file



wintime = 0.016
steptime= 0.004
sr = 32000

# Load a random file info 
subset_path = '/home/manu/workspace/databases/MillionSongSubset/data/'
import hdf5_getters
#h5 = hdf5_getters.open_h5_file_read(subset_path+'A/V/A/TRAVAAN128F9359AAE.h5')
output_dir = '/sons/rwc/Learn/hdf5/'
output = output_dir + 'rwc-g-m01_1.h5'
h5 = hdf5_getters.open_h5_file_read(output)

duration = hdf5_getters.get_duration(h5)
title = hdf5_getters.get_title(h5)
n_segments_start = hdf5_getters.get_segments_start(h5)
n_segment = n_segments_start.shape[0]
artist_name = hdf5_getters.get_artist_name(h5)
timbre = hdf5_getters.get_segments_timbre(h5)
loudness = hdf5_getters.get_segments_loudness_start(h5)
C = hdf5_getters.get_segments_pitches(h5)
beattimes = hdf5_getters.get_segments_start(h5)

digital_id = hdf5_getters.get_track_7digitalid(h5)

# guess the duration of a segment
seg_dur = n_segments_start[1:] - n_segments_start[0:-1]


# Load the learning parts
from scipy.io import loadmat
full_path = '/home/manu/workspace/audio-sketch/matlab/'
savematname = 'learnbase_allfeats_2000000_seed_78.mat'
lstruct = loadmat(full_path + savematname)
learn_feats_all = lstruct['learn_feats_all']
learn_magspecs_all = lstruct['learn_magspecs_all']
learn_files = lstruct['learn_files']

# in this context we use only the Chroma vector
learn_feats = learn_feats_all[:,20:32]
learn_magspecs = learn_magspecs_all

# To resynthesize the audio we need to use a concatenation of existing spectrums stretched 
# to the corresponding size.. but first we need to find these spectrums
nb_median = 5
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(nb_median)
neigh.fit(learn_feats)
Xtest = C
# the easiest is to duplicate the features so as to fit a grid with pre-defined step_size
steptime = 0.016
full_seg_num = int(n_segments_start[-1]/steptime)
#full_feat_matrix = np.zeros((full_seg_num, C.shape[1]))
#cur_t = 0
#cur_seg_idx = 0
#for seg_idx in range(1,full_seg_num):
#    cur_t += steptime
#    if cur_t > n_segments_start[cur_seg_idx]:        
#        full_feat_matrix[seg_idx, : ] = C[cur_seg_idx, :]
#        cur_seg_idx += 1
#    else:
#        full_feat_matrix[seg_idx, : ] = full_feat_matrix[seg_idx-1, : ]


#kneighs = neigh.kneighbors(full_feat_matrix, return_distance=False)

# ok now we have a certain amount of corresponding spectrums per segment
# we need to recreate a waveform by inverting the spectrum

estimated_spectrum, neighbors = regression.ann(learn_feats.T,
                                               learn_magspecs.T,
                                               C.T,
                                               None,
                                               K=nb_median)



sr = 16000
win_size = steptime*2*sr
step_size = steptime*sr
# sliding median filtering ?
from scipy.ndimage.filters import median_filter
estimated_spectrum_filt = median_filter(estimated_spectrum, (1, 20))

plt.figure()
plt.imshow(np.log(estimated_spectrum_filt), origin='lower')
plt.show()
# reconstruction    

#init_vec = np.random.randn(step_size*Y_hat.shape[1])
init_vec = np.random.randn(step_size*estimated_spectrum.shape[1])
#x_recon = transforms.gl_recons(estimated_spectrum_filt, init_vec, 20,
#                               win_size, step_size, display=False)

x_recon = transforms.gl_recons_vary_size(estimated_spectrum, n_segments_start, 20, win_size, step_size, display=False)

output_path = '/home/manu/workspace/audio-sketch/src/results/'
res_sig = Signal(x_recon, sr, mono=True, normalize=True)


res_sig.write(output_path+'audio/resynth_%s_%dmedian.wav'%(title,nb_median))
#sig_test_ref.write(output_path+'audio/resynth_%s_learn_%d.wav'%int(100*learn_ratio))