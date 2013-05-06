'''
expe_scripts.get_feats_for_gtzan  -  Created on May 3, 2013
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
import pyechonest
from pyechonest import config
from pyechonest import track
config.ECHO_NEST_API_KEY="5TSYCVEZEIQ9R3HEO"
#os.environ['ECHONEST_API_KEY'] = config.ECHO_NEST_API_KEY

# the goal is to get the feature information for the learn subbase of rwc audio samples
input_dir = '/home/manu/workspace/databases/genres/pop/'
output_dir = '/home/manu/workspace/databases/genres/pop/hdf5/'

# Single file is working, now loop on all files from the learning directory
#from pyechonest import track
for audiofile in features.get_filepaths(input_dir, ext='.au'):
    print "Starting work on ", audiofile    
    output = output_dir + os.path.splitext(os.path.split(audiofile)[-1])[0] + '.h5'
    if os.path.exists(output):
        continue
    file_object = open(audiofile)
    curtrack = track.track_from_file(file_object, 'au', force_upload=True)
#
    HDF5.create_song_file(output,force=False)
    h5 = HDF5.open_h5_file_append(output)
    # HACK we need to fill missing values
    curtrack.__setattr__('foreign_id','')
    curtrack.__setattr__('foreign_release_id','')
    curtrack.__setattr__('audio_md5','')
    HDF5.fill_hdf5_from_track(h5,curtrack)
    h5.close()
    del h5
    
# first testing on a single song
#audiofile = input_dir + 'rwc-g-m01_1.wav'
#output = output_dir + 'rwc-g-m01_1.h5'

# MSdB API not convenient need to identify the song
#from enpyapi_to_hdf5 import convert_one_song
#res = convert_one_song(audiofile,output,mbconnect=None,verbose=1,DESTROYAUDIO=False)

# let us use directly the echo nest API
#from pyechonest import track
#file_object = open(audiofile)
#track = track.track_from_file(file_object, 'wav', force_upload=True)
#
#HDF5.create_song_file(output,force=False)
#h5 = HDF5.open_h5_file_append(output)
## HACK we need to fill missing values
#track.__setattr__('foreign_id','')
#track.__setattr__('foreign_release_id','')
#track.__setattr__('audio_md5','')
#HDF5.fill_hdf5_from_track(h5,track)
#h5 = hdf5_getters.open_h5_file_read(output)
##
#timbre = hdf5_getters.get_segments_timbre(h5)
#loudness = hdf5_getters.get_segments_loudness_start(h5)
#C = hdf5_getters.get_segments_pitches(h5)
#n_segments_start = hdf5_getters.get_segments_start(h5)
#
#plt.figure()
#plt.hist(np.diff(n_segments_start),100)
#plt.show()

