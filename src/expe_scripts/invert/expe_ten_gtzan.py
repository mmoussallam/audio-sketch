'''
expe_scripts.invert.expe_ten_gtzan  -  Created on May 6, 2013

We have the features for 100 blues songs, can we built one of them using 
99 of them as the learning base ?

@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
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

ref_audio_dir = '/home/manu/workspace/databases/genres/blues/'
outputpath = '/home/manu/workspace/audio-sketch/src/expe_scripts/audio/feat_invert'
data_path = '/home/manu/workspace/databases/genres/blues/hdf5/'
# List all the files
h5files = [name  for name in os.listdir(data_path) if 'h5' in name]

# isolate one of them 
learn_feats_list = []
learn_segs_list = []

n_learn = 10
for fileIdx in range(n_learn):
    get_ten_features_from_file(learn_feats_list, learn_segs_list, [],
                               os.path.join(data_path, h5files[fileIdx]))

l_feats = np.concatenate(learn_feats_list, axis=0)
l_segments = np.vstack(learn_segs_list)    

for h5file in h5files:
    h5 = hdf5_getters.open_h5_file_read(os.path.join(data_path, h5file))
    print h5file,  hdf5_getters.get_tempo(h5),  hdf5_getters.get_key(h5)


t_index = len(h5files)-1

# get the test
test_feats_list = []
test_segs_list = []
test_confidence_list = []
get_ten_features_from_file(test_feats_list, test_segs_list, test_confidence_list,
                           os.path.join(data_path, h5files[t_index]))

t_feats = test_feats_list[0]
t_seg_starts = test_segs_list[0][0]
t_seg_duration = np.diff(t_seg_starts)

# find the nearest neighbors
knn = NearestNeighbors(n_neighbors=10)    

nbFeats = 12

knn.fit(l_feats[:,-nbFeats:])
distance, neigh = knn.kneighbors(t_feats[:,-nbFeats:], n_neighbors=10, return_distance=True)

#from tools.learning_tools import resynth
#sig_out = resynth(np.squeeze(neigh[:,0]), t_seg_starts, t_seg_duration, 
#            l_segments, l_feats, ref_audio_dir, '.wav',
#            dotime_stretch=True, max_synth_idx=50, normalize=True)
#sig = save_audio(outputpath, h5files[t_index], sig_out, 22050, norm_segments=False)
#sig.crop(0, 9.5*sig.fs)

#sig_out = resynth_sequence(np.squeeze(neigh[:,0]), t_seg_starts, t_seg_duration,
#                           l_segments, l_feats, ref_audio_dir, '.wav', 22050,
#                           dotime_stretch=True,max_synth_idx=50,  normalize=True)
#
#sig = Signal(sig_out, 22050, normalize=True)
#sig.write('%s/%s.wav' % (outputpath,h5files[t_index]))
#sig.crop(0, 9.5*sig.fs)


# now try to viterbi decode this shit
from tools.learning_tools import Viterbi
vit_path = Viterbi(neigh, distance, trans_penalty=0.01, c_value=20)
vit_cands = [neigh[ind,neighbind] for ind, neighbind in enumerate(vit_path)]
#
sig_out_viterbi = resynth_sequence(np.squeeze(vit_cands), t_seg_starts, t_seg_duration,
                           l_segments, l_feats, ref_audio_dir, '.wav', 22050,
                           dotime_stretch=True,max_synth_idx=80,  normalize=True)
sig_viterbi = Signal(sig_out_viterbi, 22050, normalize=True)
sig_viterbi.write('%s/%s_viterbi_%dFeats_%dLearns_.wav' % (outputpath,h5files[t_index], nbFeats,n_learn))
sig_viterbi.crop(0, 9.5*sig_viterbi.fs)
#
#sig_viterbi = save_audio(outputpath, '%s_viterbi'%h5files[t_index], sig_out_viterbi, 22050, norm_segments=False)