'''
expe_scripts.invert.mesure_l2norm_big_expe  -  Created on May 13, 2013

in the spirit of ACM:
- vary the size of the learn, database (after shuffling?)
- vary the number of neighbors
- vary the combination of feature used

- for each configuration: make 100 random tests and average out the results

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

from joblib import Memory
memory = Memory(cachedir='/tmp/joblib/feat_invert/', verbose=0)

def frob_norm(A,B):
    return np.linalg.norm((A-B),'fro')

def db_mse(A,B):
    return 20*np.log10(np.linalg.norm(((A-B)/np.linalg.norm(A)),'fro'))

def rel_frob_norm(A,B):
    return np.linalg.norm(((A/np.linalg.norm(A))-(B/np.linalg.norm(B))),'fro')

def get_feat_idx_from_name(comb_name):
    if comb_name.find('Chroma')>=0:
        ret = range(15,27)
        if comb_name.find('Timbre')>=0:
            ret.extend(range(0,12))
        if comb_name.find('Loudness')>=0:
            ret.extend(range(12,15))
    elif comb_name.find('Timbre')>=0:
        ret = range(0,12)
        if comb_name.find('Loudness')>=0:
            ret.extend(range(12,15))
    elif comb_name == 'Loudness':
        ret = range(12,15)
    elif comb_name == 'All':
        ret = range(27)
    else:
        raise ValueError('Unrecognized %s'%comb_name)
    return ret

ref_audio_dirs = ['/home/manu/workspace/databases/genres/blues/',
#                  '/home/manu/workspace/databases/genres/pop/',
#                  '/home/manu/workspace/databases/genres/rock/',
#                  '/home/manu/workspace/databases/genres/disco/',
#                  '/home/manu/workspace/databases/genres/metal/',
#                  '/home/manu/workspace/databases/genres/jazz/',
#                  '/home/manu/workspace/databases/genres/classical/',
#                  '/home/manu/workspace/databases/genres/reggae/',
#                  '/home/manu/workspace/databases/genres/hiphop/',
#                  '/home/manu/workspace/databases/genres/country/',
#                  '/sons/rwc/Piano/',
#                  '/sons/rwc/Learn/',
                  '/sons/rwc/Test/']

######### main loops 
n_frames_list = [1000,10000,100000] # logarithmically scaled
feat_combinations = ['Chroma','Timbre','Loudness',
                     'Chroma-Timbre','Chroma-Loudness',
                     'Timbre-Loudness','All']
n_knn = [1,5,10,20]
nbtest = 10
nb_max_seg = 50
# BUGFIX NO METAL OR JAZZ (missing features)
genre_list = [s for s in os.listdir('/home/manu/workspace/databases/genres/') if s not in ['jazz','metal']]

# preload the complete database
for tidx in range(nbtest):
    # Randomly select a file in the base
    t_genre_idx = np.random.randint(0,len(genre_list))
    t_name_idx = np.random.randint(0,100)
    t_path = '/home/manu/workspace/databases/genres/'+genre_list[t_genre_idx]
    name_list = os.listdir(t_path)
    t_name = os.path.splitext(name_list[t_name_idx])[0]
    test_file_target = '%s/hdf5/%s.h5'%(t_path,t_name)
    
    test_key, t_feats, t_seg_starts, t_seg_duration = get_test(test_file_target)
    
    # Load the base and make sure test file is avoided
    l_feats_all, l_segments_all, n_learn = get_learns_multidir(ref_audio_dirs,
                                                  filter_key= None,
                                                  t_name=t_name,
                                                  n_learn_max = 1000)
        
    marge = 0.0
    orig, fs = get_audio(t_path + '/' + t_name + '.au',
                         0,
                         np.sum(t_seg_duration[:nb_max_seg])+marge, targetfs=22050)    
        #orig =  Signal(t_path + t_name + '.au', normalize=True)
    print "Working on %s duration of %2.2f"%(t_name, np.sum(t_seg_duration[:nb_max_seg]))
    orig_spec = np.abs(stft.stft(orig, 512,128)[0,:,:])
    
    # for each combination
    for Nidx, n_frames in enumerate(n_frames_list):
        print "Starting work on N=",n_frames
        for Midx, feat_comb in enumerate(feat_combinations):
            print "Starting work on comb : ",feat_comb
#            print get_feat_idx_from_name(feat_comb)
            # Limit the number of 
            