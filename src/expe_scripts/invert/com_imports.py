'''
expe_scripts.invert.com_imports  -  Created on May 7, 2013
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

from tools.learning_tools import find_indexes, get_ten_features_from_file, get_ten_features, get_track_info
from tools.learning_tools import resynth_sequence, save_audio


def get_ten_feats_fullpath(feats_all, segments_all,  h5file):
    h5 = hdf5_getters.open_h5_file_read(h5file)
    timbre = hdf5_getters.get_segments_timbre(h5)
    loudness_start = hdf5_getters.get_segments_loudness_start(h5)
    loudness_max = hdf5_getters.get_segments_loudness_max(h5)
    loudness_max_time = hdf5_getters.get_segments_loudness_max_time(h5)
    C = hdf5_getters.get_segments_pitches(h5)        
    
    (hdf5path, filename) = os.path.split(h5file)
    target_audio_dir = os.path.split(hdf5path)[0]
    target_audio_path = os.path.join(target_audio_dir, os.path.splitext(filename)[0])
    segments_all.append(np.array([hdf5_getters.get_segments_start(h5), target_audio_path]))
    
    feats_all.append(np.hstack((timbre, loudness_start.reshape((loudness_start.shape[0], 1)),
                                loudness_max.reshape((loudness_max.shape[0], 1)),
                                loudness_max_time.reshape((loudness_max_time.shape[0], 1)),
                                C)))
    h5.close()


def get_test(test_path):
    test_feats_list = []
    test_segs_list = []
    test_confidence_list = []
    get_ten_features_from_file(test_feats_list, test_segs_list, test_confidence_list, test_path)
    t_feats = test_feats_list[0]
    t_seg_starts = test_segs_list[0][0]
    t_seg_duration = np.diff(t_seg_starts)
    h5 = hdf5_getters.open_h5_file_read(test_path)
    test_key = hdf5_getters.get_key(h5)
    h5.close()
    return test_key, t_feats, t_seg_starts, t_seg_duration

def get_test_params(t_index, data_path):
    h5files = [name for name in os.listdir(data_path) if 'h5' in name] # isolate one of them
    t_index = len(h5files) - 1 # get the test
    test_feats_list = []
    test_segs_list = []
    test_confidence_list = []
    get_ten_features_from_file(test_feats_list, test_segs_list, test_confidence_list, os.path.join(data_path, h5files[t_index]))
    t_feats = test_feats_list[0]
    t_seg_starts = test_segs_list[0][0]
    t_seg_duration = np.diff(t_seg_starts)
    test_key = hdf5_getters.get_key(hdf5_getters.open_h5_file_read(os.path.join(data_path, h5files[t_index])))
    return t_index, h5files, test_key, t_feats, t_seg_starts, t_seg_duration


def get_learns_and_test(ref_audio_dir, data_path, t_index = 100, filter_key= True, n_learn_max = 99):
    
    # load the test data
    t_index, h5files, test_key, t_feats, t_seg_starts, t_seg_duration = get_test_params(t_index, data_path) 
    
    # Now load all the others    
    learn_feats_list = []
    learn_segs_list = []    
    n_learn = 0    
    for fileIdx in range(t_index):
        if filter_key and n_learn < n_learn_max:
            if hdf5_getters.get_key(hdf5_getters.open_h5_file_read(os.path.join(data_path, h5files[fileIdx]))) == test_key:
                get_ten_features_from_file(learn_feats_list, learn_segs_list, [], os.path.join(data_path, h5files[fileIdx]))
                n_learn += 1
        elif n_learn < n_learn_max:
            get_ten_features_from_file(learn_feats_list, learn_segs_list, [], os.path.join(data_path, h5files[fileIdx]))
            n_learn += 1
    
    l_feats = np.concatenate(learn_feats_list, axis=0)
    l_segments = np.vstack(learn_segs_list)
    for h5file in h5files:
        h5 = hdf5_getters.open_h5_file_read(os.path.join(data_path, h5file))
        print h5file, hdf5_getters.get_tempo(h5), hdf5_getters.get_key(h5)
    
    return l_feats, t_feats, t_seg_starts, t_seg_duration, l_segments, h5files, n_learn





def get_learns_multidir(ref_audio_dirs, filter_key= None, t_name=None, n_learn_max = 99):
    """ Load the features for a whole lotta directories """
        
    # Now load all the others
    learn_feats_list = []
    learn_segs_list = []    
    n_learn = 0    
    for dir in ref_audio_dirs:
        
        target_path = os.path.join(dir,'hdf5')
        print "loading from %s"% target_path
        h5files = [name for name in os.listdir(target_path) if 'h5' in name]
        for fileIdx in range(len(h5files)):
            if t_name in h5files[fileIdx]:
                print "Excluding %s from learn"%t_name
                continue
            if filter_key is not None and n_learn < n_learn_max:
                h5 = hdf5_getters.open_h5_file_read(os.path.join(target_path, h5files[fileIdx]))
                target_key = hdf5_getters.get_key(h5)
                h5.close()
                if target_key == filter_key:
                    get_ten_feats_fullpath(learn_feats_list, learn_segs_list, os.path.join(target_path, h5files[fileIdx]))
                    n_learn += 1
            elif n_learn < n_learn_max:
                get_ten_feats_fullpath(learn_feats_list, learn_segs_list,  os.path.join(target_path, h5files[fileIdx]))
                n_learn += 1
        print "Reached %d",n_learn
                

    l_feats = np.concatenate(learn_feats_list, axis=0)
    l_segments = np.vstack(learn_segs_list)
    
    return l_feats,l_segments, n_learn
