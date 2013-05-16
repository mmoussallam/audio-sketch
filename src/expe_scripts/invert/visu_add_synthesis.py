'''
expe_scripts.invert.visu_add_synthesis  -  Created on May 16, 2013
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
import stft
from sklearn.neighbors import NearestNeighbors
from tools.learning_tools import resynth_sequence, get_ten_features, get_ten_features_from_file, get_track_info
from expe_scripts.invert.com_imports import get_learns_multidir , get_test
from feat_invert.transforms import get_audio
output_audio_path = '/home/manu/Documents/Articles/ISMIR2013/ListeningMSD/Audio/'
output_fig_path = '/home/manu/Documents/Articles/ISMIR2013/ListeningMSD/Figures/'
import matplotlib.cm as cm

def recons_save_fig_audio(magspec, target_name,n_max_frames,fs=22050, format=(8,3), nb_gl_iter = 30):
    
    init_vec = np.random.randn(128*n_max_frames)
    x_recon = transforms.gl_recons(magspec[:,:n_max_frames], init_vec, nb_gl_iter,
                                               512, 128, display=False)
    rec_sig = Signal(x_recon, fs, normalize=True)
    
    rec_sig.write(os.path.join(output_audio_path, '%s.wav'%target_name))
    plt.figure(figsize=format)
    rec_sig.spectrogram(512, 128, order=1, log=True, cmap=cm.jet, cbar=False)
    plt.savefig(os.path.join(output_fig_path, '%s.png'%target_name))

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

Ps = [20,]
feat_combs = ['All',]
M = 150000
genre = 'reggae'
t_name = 'reggae.00011'
t_path = '/home/manu/workspace/databases/genres/%s'%genre
test_file_target = '%s/hdf5/%s.h5'%(t_path,t_name)
test_key, t_feats_all, t_seg_starts, t_seg_duration = get_test(test_file_target)

# Load the base and make sure test file is avoided
l_feats_all, l_segments_all, n_learn = get_learns_multidir(ref_audio_dirs,
                                              filter_key= None,
                                              t_name=genre,
                                              n_learn_max = 1000)

l_feats_all, l_segments_all, n_learn = get_learns_multidir(ref_audio_dirs,
                                                  filter_key= None,
                                                  t_name=genre,
                                                  n_learn_max = 1000)
        
marge = 2.0
nb_max_seg = 20
print t_path + '/' + t_name + '.au'
from feat_invert.transforms import get_audio
orig, fs = get_audio(t_path + '/' + t_name + '.au',
                     0,
                     np.sum(t_seg_duration[:nb_max_seg])+marge, targetfs=22050)    
    #orig =  Signal(t_path + t_name + '.au', normalize=True)
print "Working on %s duration of %2.2f"%(t_name, np.sum(t_seg_duration[:nb_max_seg]))
orig_spec = np.abs(stft.stft(orig, 512,128)[0,:,:])

sig_ellis = Signal('%sellis_resynth%s.wav'%(output_audio_path,t_name), normalize=True)
#sig_ellis.crop(0,orig_spec.shape[1]*128 + 512-128)
plt.figure(figsize=(8,3))
sig_ellis.spectrogram(512, 128, order=1, log=True, cmap=cm.jet, cbar=False)
plt.savefig(os.path.join(output_fig_path, '%s_ellis.png'%t_name))


feat_idxs = get_feat_idx_from_name(feat_combs[0])
# limit the size of the l_features vector:
l_feats = l_feats_all[:M, feat_idxs]
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
knn = NearestNeighbors(n_neighbors=np.max(Ps))    
knn.fit(l_feats)
neigh = knn.kneighbors(t_feats,
                             n_neighbors=np.max(Ps),
                             return_distance=False)

# rebuild the magnitude spectrogram
for kidx, k in enumerate(Ps):
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
    magspecarr = np.array(magspecs)
    max_magspec = np.max(magspecarr, 0)
    mean_magspec = np.mean(magspecarr, 0)
    median_magspec = np.median(magspecarr, 0)
    n_max_frames = orig_spec.shape[1]
    nb_gl_iter = 20
    recons_save_fig_audio(max_magspec, '%s_max_%s_P%d'%(t_name,feat_combs[0],k),
                          n_max_frames,
                          fs=22050, format=(8,3), nb_gl_iter = nb_gl_iter)
    
    recons_save_fig_audio(mean_magspec, '%s_mean_%s_P%d'%(t_name,feat_combs[0],k),
                          n_max_frames,
                          fs=22050, format=(8,3), nb_gl_iter = nb_gl_iter)
    
    recons_save_fig_audio(median_magspec, '%s_median_%s_P%d'%(t_name,feat_combs[0],k),
                          n_max_frames,
                          fs=22050, format=(8,3), nb_gl_iter = nb_gl_iter)
