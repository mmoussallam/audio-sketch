'''
expe_scripts.invert.expe_features_clustering  -  Created on May 2, 2013
@author: M. Moussallam
'''

# Suppose you load some voice example and compute features of it
# Can you segment in phonemes/syllabes by looking at features?

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

from yaafelib import *
from scipy.ndimage.filters import median_filter

win_size = 1024
step_size = 128

# Learning phase
learn_audiofilepath = '/sons/sqam/voicemale.wav'

#all_feat_name_list = ['zcr','OnsetDet','energy','specstats','specflux','mfcc','pcp']
tested_features = ['zcr','OnsetDet','energy','specstats','specflux','mfcc']

[MagSpectrums, Feats, Datas] = features.load_data_one_audio_file(learn_audiofilepath, 
                                                     16000,
                                                     sigma_noise=0,
                                                     wintime=0.064,
                                                     steptime=0.008,
                                                     max_frame_num_per_file=5000,
                                                     startpoint = 0,
                                                     features=tested_features)



#plt.figure()
#plt.subplot(311)
#plt.imshow(np.log10(MagSpectrums.T), origin='lower')
#plt.subplot(312)
#plt.plot(Datas)
#plt.subplot(313)
#plt.imshow(np.log10(np.abs(Feats.T)))

from sklearn.decomposition import PCA
from sklearn.lda import LDA
pca = PCA(n_components=100)
X_r = pca.fit(Feats).transform(Feats)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(Feats)
#
#plt.figure()
#plt.imshow(np.log10(np.abs(kmeans.cluster_centers_.T)))
#plt.show()
#
#plt.figure()
#plt.subplot(211)
#plt.imshow(np.log10(np.abs(Feats.T)))
#plt.subplot(212)
#plt.plot(kmeans.labels_)
#plt.show()


############### Loading a database
params = {}
params['n_frames'] = 400000
base_str = 'VoxMonoSpeaker0'
params['shuffle'] = 123
params['wintime'] = 0.064
params['steptime'] = 0.008
params['sr'] = 16000
params['frame_num_per_file'] = 2000
params['features'] = ['zcr','OnsetDet','energy','specstats','mfcc','chroma','pcp']
params['location'] = 'learn_directory'

full_path = '/home/manu/workspace/audio-sketch/data/bases/'
output_path = '/home/manu/workspace/audio-sketch/src/results/'
savematname = 'learnbase_%s_allfeats_%d_seed_%d_win%d.mat'%(base_str,
                                                            params['n_frames'],
                                                         params['shuffle'],
                                                         int(params['wintime']*params['sr']))

print "LOADING"
from scipy.io import loadmat
lstruct = loadmat(full_path + savematname)
learn_feats_all = lstruct['learn_feats_all']
learn_magspecs_all = lstruct['learn_magspecs_all']
learn_files = lstruct['learn_files']
print learn_magspecs_all.shape

wsize = 2*(learn_magspecs_all.shape[1]-1)

######## How are the candidates distributed ?
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=3)
Learnidxs = [0,1,2,3,4,5,6]
knn.fit(learn_feats_all[:,Learnidxs])

distance, neighbs = knn.kneighbors(Feats[:,Learnidxs],
                                   n_neighbors=3,
                                   return_distance=True)

# reconstruct
Y_hat = np.zeros_like(MagSpectrums)
T = neighbs.shape[0]
for t in range(T):
    Y_hat[t,:] = np.median(learn_magspecs_all[neighbs[t, :],:], 0)

init_vec = np.random.randn(128*Y_hat.shape[0])
x_recon = transforms.gl_recons(Y_hat.T, init_vec, 50,
                                   wsize, 128, display=False)

import sti

orig_sig = Signal(learn_audiofilepath, mono=True, normalize=True)
orig_sig.downsample(16000)
sig = Signal(x_recon, 16000, normalize=True)

score = sti.stiFromAudio(orig_sig.data, x_recon, 16000,
                         calcref=False, downsample=None, name="unnamed")


# can we perform viterbi decoding ?
n_candidates = neighbs.shape[1]
n_states = neighbs.shape[0]
transition_cost = np.ones((n_candidates,))
cum_scores = np.zeros((n_candidates,))
paths = []
# initalize the paths and scores
for candIdx in range(n_candidates):
    paths.append([0,])
    cum_scores = distance[0,:]

for stateIdx in range(1, n_states):
    for candIdx in range(n_candidates):
        trans_penalty = [1 if not abs(neighbs[stateIdx-1,i]-neighbs[stateIdx,candIdx])<5 else 0.05 for i in range(n_candidates)]
        trans_score = trans_penalty * cum_scores # to be replaced by a penalty of moving far from previous index         
        best_prev_ind = np.argmin(trans_score)        
        paths[candIdx].append(best_prev_ind)
        cum_scores[candIdx] = distance[stateIdx,candIdx] + trans_score[best_prev_ind]

best_score_ind = np.argmin(cum_scores)
best_path = paths[best_score_ind]

# compare the two reconstructions
Y_hat_1nn = np.zeros_like(MagSpectrums)
T = neighbs.shape[0]
for t in range(T):
    Y_hat_1nn[t,:] = learn_magspecs_all[neighbs[t, 0],:]
init_vec = np.random.randn(128*Y_hat_1nn.shape[0])
x_recon_1nn = transforms.gl_recons(Y_hat_1nn.T, init_vec, 50,
                                   win_size, 128, display=False)
sig_1nn = Signal(x_recon_1nn, 16000, normalize=True)

Y_hat_viterbi = np.zeros_like(MagSpectrums)
T = neighbs.shape[0]
for t in range(T):
    Y_hat_viterbi[t,:] = learn_magspecs_all[neighbs[t, best_path[t]],:]

init_vec = np.random.randn(128*Y_hat_viterbi.shape[0])
x_recon_viterbi = transforms.gl_recons(Y_hat_viterbi.T, init_vec, 50,
                                   win_size, 128, display=False)

sig_viterbi = Signal(x_recon_viterbi, 16000, normalize=True)

score_viterbi = sti.stiFromAudio(orig_sig.data, [x_recon, x_recon_1nn, x_recon_viterbi],
                                 16000,
                                 calcref=False, downsample=None, name="unnamed")

plt.figure()
plt.plot(x_recon_viterbi - x_recon_1nn)
plt.show()
# is the reconstructed spectrogram relevantly statistically distributed ?
import matplotlib.cm as cm
from scipy.ndimage import median_filter
plt.figure()
plt.subplot(311)
plt.imshow(np.log10(median_filter(Y_hat.T,[3,1])), origin='lower')
plt.subplot(312)
plt.imshow(np.log10(Y_hat_viterbi.T), origin='lower')
plt.plot([i*2 for i in best_path],'k')
plt.subplot(313)
orig_sig.spectrogram(win_size, 128, order=1, log=True,cmap=cm.jet, cbar=False)
plt.show()


filt_sizes = [[1,1],[3,1],[5,1],[10,1],[1,3],[1,5],[1,10],[3,3],[5,5],[10,10]]
x_recons = []
for filt_size in filt_sizes:
    guess_yhat = median_filter(Y_hat_viterbi,filt_size)
    init_vec = np.random.randn(128*guess_yhat.shape[0])
    x_recons.append(transforms.gl_recons(guess_yhat.T, init_vec, 50,
                                   win_size, 128, display=False))

scores = sti.stiFromAudio(orig_sig.data, x_recons,
                                 16000,
                                 calcref=False, downsample=None, name="unnamed")

# we need to verify..
sig_filtered = Signal(x_recons[1], orig_sig.fs, normalize=True, mono=True)


##
#plt.figure()
#plt.plot(np.log10(MagSpectrums[70,:]/np.sum(MagSpectrums[70,:])))
#plt.plot(np.log10(learn_magspecs_all[neighbs[70,0],:]))
#plt.show()
#
##  impact of taking more dev spetrums
#plt.figure()
#plt.plot(np.log10(MagSpectrums[70,:]))
#plt.plot(np.log10(learn_magspecs_all[neighbs[70,0],:]))
#plt.plot(np.log10(np.median(learn_magspecs_all[neighbs[70,0:3],:],0)))
#plt.plot(np.log10(np.median(learn_magspecs_all[neighbs[70,0:10],:],0)))
#plt.plot(np.log10(np.median(learn_magspecs_all[neighbs[70,0:30],:],0)))
#plt.show()
#
## What is the distance between next frames
#np.sqrt(np.sum((Feats[71,0:7] - learn_feats_all[neighbs[71,0],0:7])**2))
#plt.figure()
##plt.plot(Feats[70,0:7])
##plt.plot(Feats[71,0:7])
##plt.plot(Feats[72,0:7])
#plt.plot(learn_feats_all[neighbs[71,0]-1,0:7])
#plt.plot(learn_feats_all[neighbs[71,0],0:7])
#plt.plot(learn_feats_all[neighbs[71,0]+1,0:7])
#plt.show()

#
#plt.figure()
#plt.plot(np.log(MagSpectrums[70,:]))
#plt.plot(np.log(MagSpectrums[71,:]))
#plt.plot(np.log(MagSpectrums[72,:]))
#plt.plot(np.log(learn_magspecs_all[neighbs[71,0]-1,:]),':')
#plt.plot(np.log(learn_magspecs_all[neighbs[71,0],:]),':')
#plt.plot(np.log(learn_magspecs_all[neighbs[71,0]+1,:]),':')
#plt.show()
