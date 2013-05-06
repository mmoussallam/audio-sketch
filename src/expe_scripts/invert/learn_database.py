'''
expe_scripts.invert.learn_database  -  Created on May 3, 2013
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
from feat_invert.features import load_yaafedata
#sys.path.append('/usr/local/python_packages')
from sklearn.neighbors import NearestNeighbors
from yaafelib import *
from scipy.ndimage.filters import median_filter


learn_directories = ['cmu_us_jmk_arctic',
                     'cmu_us_ksp_arctic',
                     'cmu_us_rms_arctic',
                     'cmu_us_slt_arctic']

for speakerIdx in range(len(learn_directories)):
    learn_directory = '/sons/voxforge/main/Learn/%s'%learn_directories[speakerIdx]
    base_str = 'VoxMonoSpeaker%d'%speakerIdx
    for wintime in [0.032, 0.064]:
        startpoint = 0  # in seconds
        learn_seed = 123
        params = {}
        params['n_frames'] = 400000
        #params.sigma = 0.00001;
        params['shuffle'] = learn_seed
        params['wintime'] = wintime
        params['steptime'] = 0.008
        params['sr'] = 16000
        params['frame_num_per_file'] = 5000
        params['features'] = ['zcr','OnsetDet','energy','specstats','mfcc','chroma','pcp']
        params['location'] = learn_directory
        
        full_path = '/home/manu/workspace/audio-sketch/data/bases/'
        output_path = '/home/manu/workspace/audio-sketch/src/results/'
        savematname = 'learnbase_%s_allfeats_%d_seed_%d_win%d.mat'%(base_str,
                                                                    params['n_frames'],
                                                                 params['shuffle'],
                                                                 int(params['wintime']*params['sr']))
        
        if not os.path.exists(full_path + savematname):
            from scipy.io import savemat
            [learn_feats_all, learn_magspecs_all,
                n_f_learn, ref_learn_data, learn_files] = features.load_yaafedata(params)
            savemat(full_path + savematname, {'learn_feats_all':learn_feats_all,
                                               'learn_magspecs_all':learn_magspecs_all,
                                                'learn_files':learn_files})