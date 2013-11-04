'''
expe_scripts.invert.large_tests  -  Created on May 3, 2013
Let's do something modular and avoid recomputing stuff
define the parameters we want to vary and use joblib not te rerun the same 
test multiple times 

@author: M. Moussallam
'''

import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from PyMP import Signal
import itertools
import sys
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
sys.path.append('/usr/local/lib')
sys.path.append('/home/manu/workspace/toolboxes/sti-wav-master/src')
from feat_invert import regression, transforms, features
from joblib import Memory
memory = Memory(cachedir='/tmp/joblib/feat_invert/', verbose=0)
from sklearn.neighbors import NearestNeighbors
sys.path.append('/home/manu/workspace/toolboxes/sti-wav-master/src')
import sti


def load_learned_database(n_frames, nb_speakers, wintime, n_feats):
    """ Loading the learned bases - for voxforge data only"""
    learn_feats_all = []
    learn_magspecs_all = []
    for speakIdx in range(nb_speakers):
        params = {}
        params['n_frames'] = 400000
        base_str = 'VoxMonoSpeaker%d'%speakIdx
        params['shuffle'] = 123
        params['wintime'] = wintime
        params['steptime'] = 0.008
        params['sr'] = 16000
        params['frame_num_per_file'] = 2000
        params['features'] = ['zcr','OnsetDet','energy','specstats','mfcc','chroma','pcp']
        params['location'] = 'learn_directory'
        
        full_path = '/home/manu/workspace/audio-sketch/data/bases/'    
        savematname = 'learnbase_%s_allfeats_%d_seed_%d_win%d.mat'%(base_str,
                                                                    params['n_frames'],
                                                                 params['shuffle'],
                                                                 int(params['wintime']*params['sr']))
        
        print "LOADING - ",base_str
        from scipy.io import loadmat
        lstruct = loadmat(full_path + savematname)
        learn_feats = lstruct['learn_feats_all']
        learn_magspecs = lstruct['learn_magspecs_all']
        
        # Keep only the part we need, clean the rest
        learn_feats_all.append(np.array(learn_feats[:n_frames,:n_feats]))
        learn_magspecs_all.append(np.array(learn_magspecs[:n_frames,:]))
        
        del lstruct
    
    # now concatenate results
    return np.vstack(learn_magspecs_all), np.vstack(learn_feats_all)

def load_test_datas(test_filepath, wintime, n_feats):
    """ Load the data for the test file """
    tested_features = ['zcr','OnsetDet','energy','specstats','mfcc','chroma','pcp']
    [MagSpectrums, Feats, Datas] = features.load_data_one_audio_file(test_filepath, 
                                                     16000,
                                                     sigma_noise=0,
                                                     wintime=wintime,
                                                     steptime=0.008,
                                                     max_frame_num_per_file=5000,
                                                     startpoint = 0,
                                                     features=tested_features)
    
    return MagSpectrums, Feats[:,:n_feats], Datas


def reconstruct(l_specs, t_specs, neighbs, k, winsize, n_iter_gl):
    """ reconstruct by taking the median of the Knn and GL """
    Y_hat = np.zeros_like(t_specs)
    T = neighbs.shape[0]
    for t in range(T):
        Y_hat[t,:] = np.median(l_specs[neighbs[t, :k],:], 0)    
    init_vec = np.random.randn(128*Y_hat.shape[0])
    x_recon = transforms.gl_recons(Y_hat.T, init_vec, n_iter_gl,
                                       winsize, 128, display=False)
    return x_recon

@memory.cache
def do_feat_invert_test1(n_speakers, n_frames, n_feats, wintime,
                         sr, n_iter_gls,
                         knn_s, test_filepath):
    """ Load the database and the test file then
    
    -perform the nearest neighbor search
    -do the median computation for all k 
        - resynthesize using Griffin/Lim
        - compute STI score 
        - That should do the trick
        
    does it suffice to relauch everything    
    """
    n_neighbs_max = max(knn_s)
    # load the dev datas
    l_specs, l_feats = load_learned_database(n_frames, n_speakers, wintime, n_feats)
    
    print l_specs.shape, l_feats.shape
    # Fit the nearest neighbor model    
    knn = NearestNeighbors(n_neighbors=n_neighbs_max)    
    knn.fit(l_feats)
    
    # Load the test file
    t_specs, t_feats, t_data = load_test_datas(test_filepath, wintime, n_feats)
    print t_specs.shape, t_feats.data
    # Find the nearest neighbors
    distance, neighbs = knn.kneighbors(t_feats,
                                       n_neighbors=n_neighbs_max,
                                       return_distance=True)
    
    # for increasing number of examples
    x_recons = []
    for n_iter_gl in n_iter_gls:
        for k in knn_s:
            x_recons.append(reconstruct(l_specs, t_specs, neighbs,
                                        k, int(wintime*sr), n_iter_gl))


    # evaluate the STI 
    scores = np.array(sti.stiFromAudio(t_data, x_recons, sr,
                         calcref=False, downsample=None, name="unnamed"))
    
    return scores.reshape((len(n_iter_gls), len(knn_s)))

if __name__ == '__main__':
    
    # what are the parameters of a test
    n_l_speakers = [1,2,3,4]             # The number of speakers in the learned database
    n_l_frames = [10000, 50000, 100000, 200000, 400000] # number of frames per speaker to load
    nb_features = [7,20, 32, 68]              # number of features on which to do the neighbor search
    wintimes = [0.032,]          # size of sliding windows (overlap is always 128)
    n_iter_gls = [10,]             # number of Griffin&Lim iterations
    knn_s = [1,3,5,7,10,15,20]
    process = True
#    test_filename = 'voicemale'
#    test_filepath = '/sons/sqam/%s.wav'%test_filename
    test_filename = 'arctic_a0001'
    test_filepath = '/sons/voxforge/main/Test/cmu_us_awb_arctic/wav/%s.wav'%test_filename
    
    out_dir = '/home/manu/workspace/audio-sketch/data/scores/'
#    # initaliaze the STI scores
    sti_scores = np.zeros((len(n_l_speakers),
                       len(n_l_frames),
                       len(nb_features),
                       len(wintimes),
                       len(n_iter_gls),
                       len(knn_s)))
    
    for i, n_speaker in enumerate(n_l_speakers):
        for j, n_frames in enumerate(n_l_frames):
            for k, n_feats in enumerate(nb_features):
                for l, wintime in enumerate(wintimes):
                    out_name = '%sscores_%s_%d_%d_%d_%d.npy'%(out_dir, test_filename,
                                                   n_speaker,n_frames,n_feats,
                                                   int(wintime*16000))
                    print out_name
                    if process:
                        scores  = do_feat_invert_test1(n_speaker,
                                                       n_frames, n_feats, wintime,
                                                       16000, n_iter_gls,
                                                       knn_s, test_filepath)
                        
                        np.save(out_name, scores)
                    else:
                        scores = np.load(out_name)
                    sti_scores[i,j,k,l,:,:] = scores

    
    # Now we can see the influence of different variations
    # first transform it in a masked array
    masked_sti_scores = np.ma.masked_array(sti_scores, mask=np.isnan(sti_scores))
    
    # Influence of the number of speakers
    plt.figure()
    plt.plot(n_l_speakers, np.mean(np.reshape(masked_sti_scores,(len(n_l_speakers),-1) ), 1))
    plt.show()
    
    # influence of the size of database
    plt.figure()
    plt.plot(n_l_frames, np.mean(np.reshape(np.swapaxes(masked_sti_scores, 0,1),(len(n_l_frames),-1) ), 1))
    plt.show()
    
    # influence of sliding window
    plt.figure()
    plt.plot(wintimes, np.mean(np.reshape(np.swapaxes(masked_sti_scores, 0,3),(len(wintimes),-1) ), 1))
    plt.show()
    
    # influence of number of features
    plt.figure()
    plt.plot(nb_features, np.mean(np.reshape(np.swapaxes(masked_sti_scores, 0,2),(len(nb_features),-1) ), 1))
    plt.show()
    
    # influence of number GL iterations
    # NONE AT ALL
#    plt.figure()
#    plt.plot(n_iter_gls, np.mean(np.reshape(np.swapaxes(masked_sti_scores, 0,4),(len(n_iter_gls),-1) ), 1))
#    plt.show()

    # INfluence of K
    plt.figure()
    plt.plot(knn_s, np.mean(np.reshape(np.swapaxes(masked_sti_scores, 0,5),(len(knn_s),-1) ), 1))
    plt.show()
    
    
############## DEBUG Part
# why do we have some Nans ?
#scores  = do_feat_invert_test1(1,
#                               100000, 20, 0.032,
#                               16000, [5,],
#                               [5], test_filepath)

l_specs, l_feats = load_learned_database(50000, 1, 0.032, 7)
knn = NearestNeighbors(n_neighbors=3)    
knn.fit(l_feats)

t_specs, t_feats, t_data = load_test_datas(test_filepath, 0.032, 7)
distance, neighbs = knn.kneighbors(t_feats,
                                       n_neighbors=5,
                                       return_distance=True)

x_recon = reconstruct(l_specs, t_specs, neighbs,
                                        5, int(0.032*16000), 10)

sti.stiFromAudio(t_data, x_recon, 16000,
                         calcref=False, downsample=None, name="unnamed")

sig = Signal(x_recon, 16000, normalize=True)
#### the best is:
np.unravel_index(np.argmax(masked_sti_scores), masked_sti_scores.shape)
