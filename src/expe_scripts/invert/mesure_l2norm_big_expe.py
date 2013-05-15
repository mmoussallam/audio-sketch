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
from tools.learning_tools import resynth_sequence, save_audio
from joblib import Memory
memory = Memory(cachedir='/tmp/joblib/feat_invert/', verbose=0)

def frob_norm(A,B):
    return np.linalg.norm((A-B),'fro')

def db_mse(A,B):
    return 20*np.log10(np.linalg.norm(((A-B)/np.linalg.norm(A)),'fro'))

def rel_frob_norm(A,B):
    return np.linalg.norm(((A/np.linalg.norm(A))-(B/np.linalg.norm(B))),'fro')

def KLspec(A,B):
    """ KL divergence: we want to use only non nan values in B"""
    mask = B==0
    Ama = np.ma.masked_array(A, mask=mask)
    Anorm = Ama/np.sum(Ama)
    Bma = np.ma.masked_array(B, mask=mask)
    Bnorm = Bma/np.sum(Bma)
    return np.sum(Anorm*np.log(Anorm/Bnorm))

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
                  '/home/manu/workspace/databases/genres/pop/',
                  '/home/manu/workspace/databases/genres/rock/',
                  '/home/manu/workspace/databases/genres/disco/',
                  '/home/manu/workspace/databases/genres/metal/',
                  '/home/manu/workspace/databases/genres/jazz/',
                  '/home/manu/workspace/databases/genres/classical/',
                  '/home/manu/workspace/databases/genres/reggae/',
                  '/home/manu/workspace/databases/genres/hiphop/',
                  '/home/manu/workspace/databases/genres/country/',
                  '/sons/rwc/Piano/',
                  '/sons/rwc/Learn/',
                  '/sons/rwc/Test/']
out_dir = '/home/manu/workspace/audio-sketch/data/scores/'
######### main loops 
rndseed = 2
np.random.seed(rndseed)
n_frames_list = [10000,100000,] # logarithmically scaled
feat_combinations = ['Chroma','Timbre','Loudness',
                     'Chroma-Timbre','Chroma-Loudness',
                     'Timbre-Loudness','All']
n_knn = [1,5,10,20]#5,10]
nbtest = 25
nb_max_seg = 20
# BUGFIX NO METAL OR JAZZ (missing features)
genre_list = [s for s in os.listdir('/home/manu/workspace/databases/genres/') if s not in ['jazz','metal','country']]

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
    # Randomly select a file in the base
    
    # check whether this has already been computed
    magarray_name_end = 'magarray_Trial%d_%dFrames_%s_%dNN_seed%d.npy'%(tidx,n_frames_list[-1],
                                                                        feat_combinations[-1],
                                                                        max(n_knn),rndseed)
    if os.path.exists(os.path.join(out_dir,magarray_name_end)):
        print "Already Done!"
        continue
    
    
    t_genre_idx = np.random.randint(0,len(genre_list))
    t_name_idx = np.random.randint(0,100)
    t_path = '/home/manu/workspace/databases/genres/'+genre_list[t_genre_idx]
    name_list = [n for n in os.listdir(t_path) if 'au' in n]
    t_name = os.path.splitext(name_list[t_name_idx])[0]
    test_file_target = '%s/hdf5/%s.h5'%(t_path,t_name)
    print test_file_target
    test_key, t_feats_all, t_seg_starts, t_seg_duration = get_test(test_file_target)
    
    # Load the base and make sure test file is avoided
    l_feats_all, l_segments_all, n_learn = get_learns_multidir(ref_audio_dirs,
                                                  filter_key= None,
                                                  t_name=genre_list[t_genre_idx],
                                                  n_learn_max = 1000)
        
    marge = 5.0
    print t_path + '/' + t_name + '.au'
    orig, fs = get_audio(t_path + '/' + t_name + '.au',
                         0,
                         np.sum(t_seg_duration[:nb_max_seg])+marge, targetfs=22050)    
        #orig =  Signal(t_path + t_name + '.au', normalize=True)
    print "Working on %s duration of %2.2f"%(t_name, np.sum(t_seg_duration[:nb_max_seg]))
    orig_spec = np.abs(stft.stft(orig, 512,128)[0,:,:])
    orig_spec_name = 'origrray_%s_Trial%d_seed%d.npy'%(t_name,tidx,rndseed)
    np.save(os.path.join(out_dir,orig_spec_name), orig_spec)

    
    # for each combination
    for Nidx, n_frames in enumerate(n_frames_list):
        print "Starting work on N=",n_frames
        for Midx, feat_comb in enumerate(feat_combinations):
            print "Starting work on comb : ",feat_comb
            feat_idxs = get_feat_idx_from_name(feat_comb)
            # limit the size of the l_features vector:
            l_feats = l_feats_all[:n_frames, feat_idxs]
            # and the t_feats vector
            t_feats = t_feats_all[:nb_max_seg, feat_idxs]
            
            # standardizing the features            
            for f in range(l_feats.shape[1]):
                mu = np.mean(l_feats[:,f])
                sigma = np.std(l_feats[:,f])
#                print "Mu %2.4f Sigma: %2.4f"%(mu,sigma)
                l_feats[:,f] = (l_feats[:,f] - mu)/sigma
                t_feats[:,f] = (t_feats[:,f] - mu)/sigma
            
            # Find the Nearest neighbors (maximum number)
            knn = NearestNeighbors(n_neighbors=np.max(n_knn))    
            knn.fit(l_feats)
            neigh = knn.kneighbors(t_feats,
                                         n_neighbors=np.max(n_knn),
                                         return_distance=False)

            # rebuild the magnitude spectrogram
            for kidx, k in enumerate(n_knn):
                magspecs = []
                for neighIdx in range(k):
                    # build waveform
                    sigout = resynth_sequence(np.squeeze(neigh[:,neighIdx]),
                                              t_seg_starts,
                                              t_seg_duration,
                                           l_segments_all, l_feats, '', '.au', 22050,
                                           dotime_stretch=False,
                                           max_synth_idx=nb_max_seg, marge=marge, normalize=True)
                    # stft
                    magspecs.append(np.abs(stft.stft(sigout,512,128)[0,:,:]))
                
                # take the median
                magspecarr = np.array(magspecs)
                
                # save the magnitude spectrums chosen
                magarray_name = 'magarray_Trial%d_%dFrames_%s_%dNN_seed%d.npy'%(tidx,n_frames,feat_comb,k,rndseed)
                np.save(os.path.join(out_dir,magarray_name), magspecarr)
                
                max_magspec = np.max(magspecarr, 0)
                mean_magspec = np.mean(magspecarr, 0)
                median_magspec = np.median(magspecarr, 0)
                
                loc_name = 'dbscore_Trial%d_%dFrames_%s_%dNN_seed%d.npy'%(tidx,n_frames,feat_comb,k,rndseed)
                loc_klname = 'KLscore_Trial%d_%dFrames_%s_%dNN_seed%d.npy'%(tidx,n_frames,feat_comb,k,rndseed)
                loc_error_vector = [db_mse(orig_spec, mean_magspec),
                                    db_mse(orig_spec, median_magspec),
                                    db_mse(orig_spec, max_magspec)]
                np.save(os.path.join(out_dir,loc_name), loc_error_vector)
                
                loc_kl_vector = [KLspec(orig_spec, mean_magspec),
                                    KLspec(orig_spec, median_magspec),
                                    KLspec(orig_spec, max_magspec)]
                
                
                np.save(os.path.join(out_dir,loc_klname), loc_kl_vector)
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
#                plt.figure()
#                plt.subplot(411)
#                plt.imshow(np.log(orig_spec), origin='lower')
#                plt.subplot(412)
#                plt.imshow(np.log(mean_magspec), origin='lower')
#                plt.subplot(413)
#                plt.imshow(np.log(median_magspec), origin='lower')
#                plt.subplot(414)
#                plt.imshow(np.log(max_magspec), origin='lower')
#                plt.show() 
                



# Plotting the results
plt.figure()
plt.errorbar(n_frames_list,
             np.mean(np.reshape(kl_matrix,(len(n_frames_list),-1) ), 1),
             yerr=np.std(np.reshape(kl_matrix,(len(n_frames_list),-1) ), 1))
plt.show()


plt.figure()
plt.errorbar(range(len(feat_combinations)),
             np.mean(np.reshape(np.swapaxes(kl_matrix, 0,2),(len(feat_combinations),-1) ), 1),
             yerr=np.std(np.reshape(np.swapaxes(kl_matrix, 0,2),(len(feat_combinations),-1) ), 1))
ax = plt.gca()
ax.set_xticklabels(feat_combinations, rotation=45)
plt.show()

plt.figure()
plt.plot(np.mean(np.reshape(np.swapaxes(kl_matrix, 0,1),(len(feat_combinations),-1) ), 1))
ax = plt.gca()
ax.set_xticklabels(feat_combinations)
plt.show()

plt.figure()
plt.plot(n_knn, np.mean(np.reshape(np.swapaxes(kl_matrix, 0,2),(len(n_knn),-1) ), 1))
plt.show()

# Result for 100 000 database only
mean_db_error_mat = error_matrix[...,0]

plt.figure()
plt.plot(np.mean(np.squeeze(mean_db_error_mat[0,2,...]),1))
plt.show()   