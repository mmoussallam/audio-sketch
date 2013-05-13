'''
expe_scripts.invert.expe_ten_gtzan_hybrid  -  Created on May 7, 2013
@author: M. Moussallam
'''

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

from expe_scripts.invert.com_imports import get_learns_multidir , get_test


ref_audio_dirs = ['/home/manu/workspace/databases/genres/blues/',
                  '/home/manu/workspace/databases/genres/pop/',
                  '/home/manu/workspace/databases/genres/rock/',
                  '/home/manu/workspace/databases/genres/disco/',
                  '/home/manu/workspace/databases/genres/classical/',
                  '/home/manu/workspace/databases/genres/reggae/',
                  '/home/manu/workspace/databases/genres/hiphop/',
                  '/sons/rwc/Piano/',
                  '/sons/rwc/Learn/',
                  '/sons/rwc/Test/']

ref_audio_dirs = ['/sons/rwc/Piano/',]
#                  '/sons/rwc/Learn/',
#                  '/sons/rwc/Test/']

#ref_audio_dirs = ['/home/manu/workspace/databases/genres/hiphop/',]

outputpath = '/home/manu/workspace/audio-sketch/src/expe_scripts/audio/feat_invert'


# Global parameters
t_name = 'blues.00056'
t_path = '/home/manu/workspace/databases/genres/blues/'
test_file_target = '%shdf5/%s.h5'%(t_path,t_name)

#t_name = 'TRADPIA128E078EE1B'
#t_path = '/home/manu/workspace/databases/MillionSongSubset/data/A/D/P/'
#test_file_target = '%s%s.h5'%(t_path,t_name)

test_key, t_feats, t_seg_starts, t_seg_duration = get_test(test_file_target)

filter_key = False
if filter_key:
    t_key = test_key
else:
    t_key = None
n_learn_max = 99
nbFeats = 27
max_synth_idx = 10
n_neighbs = 5

# loading learn and test base
l_feats,l_segments, n_learn = get_learns_multidir(ref_audio_dirs,
                                                  filter_key= t_key,
                                                  t_name=t_name,
                                                  n_learn_max = 1000)


knn = NearestNeighbors(n_neighbors=n_neighbs)    
knn.fit(l_feats[:,-nbFeats:])
distance, neigh = knn.kneighbors(t_feats[:,-nbFeats:], n_neighbors=n_neighbs, return_distance=True)



# first method: build 10 waveforms then stft and median of magnitude
magspecs = []
for neighIdx in range(neigh.shape[1]):
    # build waveform
    sigout = resynth_sequence(np.squeeze(neigh[:,neighIdx]), t_seg_starts, t_seg_duration,
                           l_segments, l_feats, '', '.au', 22050,
                           dotime_stretch=False, max_synth_idx=max_synth_idx,  normalize=True)
    # stft
    magspecs.append(np.abs(stft.stft(sigout,512,128)[0,:,:]))

# take the median
magspecarr = np.array(magspecs)
max_magspec = np.max(magspecarr, 0)
mean_magspec = np.mean(magspecarr, 0)
median_magspec = np.median(magspecarr, 0)

#min_magspec = np.mean(magspecarr, 0)
# load true data
#orig =  Signal(t_path + t_name + '.au', normalize=False)


######### Second do the Viterbi decoding
n_neighbs_viterbi = 20
nbFeats_viterbi = 12
knn = NearestNeighbors(n_neighbors=n_neighbs)    
knn.fit(l_feats[:,-nbFeats_viterbi:])
distance, neigh = knn.kneighbors(t_feats[:,-nbFeats_viterbi:], n_neighbors=n_neighbs_viterbi, return_distance=True)


from tools.learning_tools import Viterbi
vit_path = Viterbi(neigh, distance, t_penalty=0.01, c_value=20)
vit_cands = [neigh[ind,neighbind] for ind, neighbind in enumerate(vit_path)]
#
sig_out_viterbi = resynth_sequence(np.squeeze(vit_cands), t_seg_starts, t_seg_duration,
                           l_segments, l_feats, '', '.au', 22050,
                           dotime_stretch=True,max_synth_idx=max_synth_idx,  normalize=True)
sig_viterbi = Signal(sig_out_viterbi, 22050, normalize=True)
sig_viterbi.write('%s/%s_viterbi_%dneighbs_%dFeats_%dLearns_Filter%d.wav' % (outputpath,t_name,
                                                                   n_neighbs,nbFeats_viterbi,n_learn,filter_key))

viterbi_spec = np.abs(stft.stft(sig_out_viterbi,512,128)[0,:,:])

# Normalize the spectrums
hybrid_magspec = np.zeros_like(viterbi_spec)
for cc in range(viterbi_spec.shape[1]):
    v_spec = viterbi_spec[:,cc] #/ np.sum(viterbi_spec[:,cc])
    m_spec = mean_magspec[:,cc] #/ np.sum(mean_magspec[:,cc])
    hybrid_magspec[:,cc] = np.maximum(v_spec, m_spec)



#hybrid_magspec = np.maximum(viterbi_spec, median_magspec)
# Display the spectrograms
plt.figure()
plt.subplot(311)
plt.imshow(np.log(viterbi_spec), origin='lower')
plt.subplot(312)
plt.imshow(np.log(median_magspec), origin='lower')
plt.subplot(313)
plt.imshow(np.log(hybrid_magspec), origin='lower')
plt.show()
#plt.imshow(np.log(min_magspec), origin='lower')
#orig.spectrogram(512, 128, order=1, log=True, cmap=cm.jet, cbar=False)
#plt.show()



## same for median
init_vec = np.random.randn(128*median_magspec.shape[1])
x_recon_median = transforms.gl_recons(np.maximum(median_magspec, 1e-10), init_vec, 30,
                                   512, 128, display=False)

sig_median= Signal(x_recon_median,22050, normalize=True)
sig_median.write('%s/%s_spectral_%dmedians_%dFeats_%dLearns_Filter%d.wav' % (outputpath,t_name,
                                                                   n_neighbs,nbFeats,n_learn,filter_key))


## same for max
init_vec = np.random.randn(128*max_magspec.shape[1])
x_recon_max = transforms.gl_recons(np.maximum(max_magspec, 1e-10), init_vec, 30,
                                   512, 128, display=False)

sig_max= Signal(x_recon_max,22050, normalize=True)
sig_max.write('%s/%s_spectral_%dmax_%dFeats_%dLearns_Filter%d.wav' % (outputpath,t_name,
                                                                      n_neighbs,nbFeats,n_learn,filter_key))

# same for hybrid
init_vec = np.random.randn(128*hybrid_magspec.shape[1])
x_recon_hybrid = transforms.gl_recons(np.maximum(hybrid_magspec, 1e-10), init_vec, 30,
                                   512, 128, display=False)

sig_hybrid= Signal(x_recon_hybrid,22050, normalize=True)
sig_hybrid.write('%s/%s_hybrid_%dmedians_%d-%dFeats_%dLearns_Filter%d.wav' % (outputpath,t_name,
                                                                   n_neighbs,nbFeats,nbFeats_viterbi,n_learn,filter_key))
