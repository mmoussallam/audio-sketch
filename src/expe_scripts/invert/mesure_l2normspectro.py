'''
expe_scripts.invert.mesure_l2normspectro  -  Created on May 13, 2013

Take a segment, find it's nearest neighbor in base, mesure the distance 
as the Frobenius norm of the STFT

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
from tools.learning_tools import resynth_single_seg, save_audio
from expe_scripts.invert.com_imports import get_learns_multidir , get_test
from feat_invert.transforms import get_audio, time_stretch
def frob_norm(A,B):
    return np.linalg.norm((A-B),'fro')

def db_mse(A,B):
    return 20*np.log10(np.linalg.norm(((A-B)/np.linalg.norm(A)),'fro'))

def rel_frob_norm(A,B):
    return np.linalg.norm(((A/np.linalg.norm(A))-(B/np.linalg.norm(B))),'fro')

ref_audio_dirs = ['/home/manu/workspace/databases/genres/blues/',
                  '/home/manu/workspace/databases/genres/pop/',
                  '/home/manu/workspace/databases/genres/rock/',
                  '/home/manu/workspace/databases/genres/disco/',
                  '/home/manu/workspace/databases/genres/classical/',
                  '/home/manu/workspace/databases/genres/reggae/',
                  '/home/manu/workspace/databases/genres/hiphop/',
                  '/home/manu/workspace/databases/genres/country/',
                  '/sons/rwc/Piano/',
                  '/sons/rwc/Learn/',
                  '/sons/rwc/Test/']

#ref_audio_dirs = ['/sons/rwc/Piano/',]

outputpath = '/home/manu/workspace/audio-sketch/src/expe_scripts/audio/feat_invert'

# Global parameters
t_name = 'blues.00056'
t_path = '/home/manu/workspace/databases/genres/blues/'
test_file_target = '%shdf5/%s.h5'%(t_path,t_name)

test_key, t_feats, t_seg_starts, t_seg_duration = get_test(test_file_target)

filter_key = False
if filter_key:
    t_key = test_key
else:
    t_key = None
n_learn_max = 99
nbFeats = 27
max_synth_idx = 1

# loading learn and test base
l_feats,l_segments, n_learn = get_learns_multidir(ref_audio_dirs,
                                                  filter_key= t_key,
                                                  t_name=t_name,
                                                  n_learn_max = 1000)

target_seg_idx = 0;
marge = 0.0
orig, fs = get_audio(t_path + t_name + '.au',
                     t_seg_starts[target_seg_idx],
                     t_seg_duration[target_seg_idx]+marge, targetfs=22050)    
    #orig =  Signal(t_path + t_name + '.au', normalize=True)
orig_spec = np.abs(stft.stft(orig, 512,128)[0,:,:])

for n_neighbs in [1, 5, 10]:    
    # first method: build waveforms then stft and median of magnitude
    knn = NearestNeighbors(n_neighbors=n_neighbs)    
    knn.fit(l_feats[:,-nbFeats:])
    distance, neigh = knn.kneighbors(t_feats[target_seg_idx,-nbFeats:], n_neighbors=n_neighbs, return_distance=True)        
    magspecs = []    
    for neighIdx in range(neigh.shape[1]):
        # build waveform
        sigout = resynth_single_seg(neigh[0,neighIdx],
                                    t_seg_starts[target_seg_idx],
                                    t_seg_duration[target_seg_idx],
                                    l_segments, l_feats, '', '.au', 22050,
                                    dotime_stretch=False, 
                                    normalize=True, marge=marge)
        # stft
        magspecs.append(np.abs(stft.stft(sigout,512,128)[0,:,:]))    
    # take the median
    magspecarr = np.array(magspecs)
    max_magspec = np.max(magspecarr, 0)
    mean_magspec = np.mean(magspecarr, 0)
    median_magspec = np.median(magspecarr, 0)    
    print "MAX : ",frob_norm(orig_spec, max_magspec), db_mse(orig_spec, max_magspec)
    print "MEAN : ",frob_norm(orig_spec, mean_magspec), db_mse(orig_spec, mean_magspec)
    print "MEDIAN : ",frob_norm(orig_spec, median_magspec), db_mse(orig_spec, median_magspec)

# recenter energy
sigout /= np.linalg.norm(sigout)
sigout *= np.linalg.norm(orig)


#############"" for many segments, compare value of zero-mfcc coeff with actual norm
true_norm = []
feat_norm = []
for segIdx in range(0,30):
    orig, fs = get_audio(t_path + t_name + '.au',
                     t_seg_starts[segIdx],
                     t_seg_duration[segIdx]+0.0, targetfs=22050)    
    true_norm.append(np.linalg.norm(orig))
    feat_norm.append(t_feats[segIdx,0])

plt.figure()
plt.plot(true_norm)
plt.plot(feat_norm)
plt.show()

plt.figure()
plt.subplot(311)
plt.imshow(np.log(orig_spec), origin='lower')
plt.subplot(312)
plt.imshow(np.log(median_magspec), origin='lower')
plt.subplot(313)
plt.imshow(np.log(mean_magspec), origin='lower')
plt.show()

plt.figure()
plt.plot(orig_spec[:,0])
plt.plot(median_magspec[:,0])
plt.plot(max_magspec[:,0])
plt.show()