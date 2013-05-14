'''
expe_scripts.mesure_klnorm_readresults  -  Created on May 14, 2013
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
from tools.learning_tools import resynth_sequence, save_audio

out_dir = '/home/manu/workspace/audio-sketch/data/scores/'
n_frames_list = [100000,] # logarithmically scaled
feat_combinations = ['Chroma','Timbre','Loudness',
                     'Chroma-Timbre','Chroma-Loudness',
                     'Timbre-Loudness','All']
n_knn = [1,5,10]
nbtest = 40
# BUGFIX NO METAL OR JAZZ (missing features)

error_matrix = np.zeros((nbtest,
                         len(n_frames_list),
                         len(feat_combinations),
                         len(n_knn),3))
kl_matrix = np.zeros((nbtest,
                         len(n_frames_list),
                         len(feat_combinations),
                         len(n_knn),3))
# preload the complete database
for tidx in range(nbtest):
    for Nidx, n_frames in enumerate(n_frames_list):
        print "Starting work on N=",n_frames
        for Midx, feat_comb in enumerate(feat_combinations):
            for kidx, k in enumerate(n_knn):
                
                magarray_name = 'magarray_Trial%d_%dFrames_%s_%dNN.npy'%(tidx,n_frames,feat_comb,k)
#                magspecarr = np.load(os.path.join(out_dir,magarray_name))
                                
#                max_magspec = np.max(magspecarr, 0)
#                mean_magspec = np.mean(magspecarr, 0)
#                median_magspec = np.median(magspecarr, 0)
                
                loc_name = 'dbscore_Trial%d_%dFrames_%s_%dNN.npy'%(tidx,n_frames,feat_comb,k)
                loc_klname = 'KLscore_Trial%d_%dFrames_%s_%dNN.npy'%(tidx,n_frames,feat_comb,k)
                loc_error_vector = np.load(os.path.join(out_dir,loc_name))
                                
                loc_kl_vector = np.load(os.path.join(out_dir,loc_klname))
                
                error_matrix[tidx,Nidx,Midx,kidx,:] = loc_error_vector
                kl_matrix[tidx,Nidx,Midx,kidx,:] = loc_kl_vector
                
                arglist = [tidx,n_frames,feat_comb,k]
                arglist.extend(loc_error_vector)
                

                arglist.extend(loc_kl_vector)
                scorestr= """  Trial : %d %d Frames %s %dNN
             MEAN   MEDIAN   MAX
    MSE :    %2.2f - %2.2f - %2.2f
    KL :     %2.2f - %2.2f - %2.2f"""%(tuple(arglist))
                print scorestr

# Great now plotting the results
klmat_mean = kl_matrix[...,0]
klmat_median = kl_matrix[...,1]
klmat_max = kl_matrix[...,2]

plt.figure()
plt.errorbar(range(len(feat_combinations)),
             np.mean(np.reshape(np.swapaxes(klmat_max, 0,2),(len(feat_combinations),-1) ), 1),
             yerr=np.std(np.reshape(np.swapaxes(klmat_max, 0,2),(len(feat_combinations),-1) ), 1))
ax = plt.gca()
ax.set_xticklabels(feat_combinations, rotation=45)
plt.subplots_adjust(bottom=0.5)
plt.show()


plt.figure()
plt.boxplot(np.reshape(np.swapaxes(klmat_max, 0,2),(len(feat_combinations),-1) ).T)             
ax = plt.gca()
ax.set_xticklabels(feat_combinations, rotation=45)
plt.ylabel('KL divergence')
plt.subplots_adjust(bottom=0.3)
plt.show()


plt.figure()
plt.plot(n_knn, np.mean(np.reshape(np.swapaxes(klmat_mean, 0,3),(len(n_knn),-1) ), 1))
plt.plot(n_knn, np.mean(np.reshape(np.swapaxes(klmat_median, 0,3),(len(n_knn),-1) ), 1))
plt.plot(n_knn, np.mean(np.reshape(np.swapaxes(klmat_max, 0,3),(len(n_knn),-1) ), 1))
plt.show()

plt.figure()
plt.plot(n_knn, np.mean(klmat_mean[:,-1,6,:],0))
plt.plot(n_knn, np.mean(klmat_median[:,-1,6,:],0))
plt.plot(n_knn, np.mean(klmat_max[:,-1,6,:],0))
plt.show()

plt.figure()
plt.imshow(np.mean(klmat_max[:,-1,:,:],0))
plt.colorbar()
plt.show()

min_idx = np.unravel_index(klmat_max.argmin(), klmat_max.shape)
print klmat_max[min_idx]
min_idx = np.unravel_index(klmat_mean.argmin(), klmat_mean.shape)
print klmat_mean[min_idx]

# Loading the corresponding mean
magarray_name = 'magarray_Trial%d_%dFrames_%s_%dNN.npy'%(min_idx[0],
                                                         100000,
                                                         feat_combinations[min_idx[2]],
                                                         n_knn[min_idx[3]])
magspecarr = np.load(os.path.join(out_dir,magarray_name))
max_magspec = np.max(magspecarr, 0)


#plt.figure()
#plt.imshow(np.log(max_magspec), origin='lower')
#plt.show() 
# influence of the number of NN
