'''
expe_scripts.invert.Visu_all_synthesis  -  Created on May 17, 2013

generate the big picture end of the ismir paper
summarizes visu_add and visu_cross
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyMP import Signal
import sys
import os
import os.path as op
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
from expe_scripts.invert.com_imports import get_learns_multidir , get_test
from feat_invert.transforms import get_audio, gl_recons
output_audio_path = '/home/manu/Documents/Articles/ISMIR2013/ListeningMSD/Audio/'
output_fig_path = '/home/manu/Documents/Articles/ISMIR2013/ListeningMSD/Figures/'
from tools.learning_tools import Viterbi, resynth_sequence

from expe_scripts.invert.visu_add_synthesis import ref_audio_dirs, get_feat_idx_from_name

def KLspec(A,B):
    """ KL divergence: we want to use only non nan values in B"""
    mask = B==0
    Ama = np.ma.masked_array(A, mask=mask)
    Anorm = Ama/np.sum(Ama)
    Bma = np.ma.masked_array(B, mask=mask)
    Bnorm = Bma/np.sum(Bma)
    return np.sum(Anorm*np.log(Anorm/Bnorm))    

solo_piano_dirs = ['/sons/rwc/Piano',]

# Parameters
M = 150000
genre = 'classical'
t_name = 'classical.00019'
recons_audio_path = op.join(output_audio_path,t_name)
if not op.exists(recons_audio_path):
    os.mkdir(recons_audio_path)
marge = 5.0
nb_max_seg = 50
P = 20
feat_comb = 'Chroma'
t_path = '/home/manu/workspace/databases/genres/%s'%genre
test_file_target = '%s/hdf5/%s.h5'%(t_path,t_name)
test_key, t_feats_all, t_seg_starts, t_seg_duration = get_test(test_file_target)

target_duration = np.sum(t_seg_duration[:nb_max_seg]) + marge

# Recover the dev data for complete base
l_feats_all, l_segments_all, n_learn = get_learns_multidir(ref_audio_dirs,
                                                      filter_key= None,
                                                      t_name=genre,
                                                      n_learn_max = 1000)

# Recover the dev data in Solo Piano case
l_feats_piano, l_segments_piano, n_learn_piano = get_learns_multidir(solo_piano_dirs,
                                                      filter_key= None,
                                                      t_name=genre,
                                                      n_learn_max = 1000)

# Loading the reference and Ellis reconstruction
print t_path + '/' + t_name + '.au'
orig, fs = get_audio(t_path + '/' + t_name + '.au', 0,
                     target_duration, targetfs=22050)    

sig_orig = Signal(orig, fs, normalize=True)
sig_orig.write(op.join(recons_audio_path, '_original.wav'))
print "Working on %s duration of %2.2f"%(t_name, np.sum(t_seg_duration[:nb_max_seg]))
orig_spec = np.abs(stft.stft(orig, 512,128)[0,:,:])
Lmin = orig_spec.shape[1]

sig_ellis = Signal('%sellis_resynth%s.wav'%(output_audio_path,t_name), normalize=True)
sig_ellis.write(op.join(recons_audio_path, '_ellisrec.wav'))
magspec_ellis = np.abs(stft.stft(sig_ellis.data[:,0], 512,128)[0,:,:])
Lmin = min(Lmin,magspec_ellis.shape[1])
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], magspec_ellis[:,:Lmin])

#sig_ellis.crop(0,orig_spec.shape[1]*128 + 512-128)
#plt.figure(figsize=(8,3))
#sig_ellis.spectrogram(512, 128, order=1, log=True, cmap=cm.jet, cbar=False)
#plt.savefig(os.path.join(output_fig_path, '%s_ellis.png'%t_name))


# Filter the features by the selected combination:
feat_idxs = get_feat_idx_from_name(feat_comb)
feat_idxs_piano = get_feat_idx_from_name('Chroma')
# limit the size of the l_features vector:
l_feats = l_feats_all[:M, feat_idxs]

l_feats_piano = l_feats_piano[:, feat_idxs_piano]
# and the t_feats vector
t_feats = t_feats_all[:nb_max_seg, feat_idxs]
t_feats_piano = t_feats_all[:nb_max_seg, feat_idxs_piano]


# standardizing the features            
for f in range(l_feats.shape[1]):
    mu = np.mean(l_feats[:,f])
    sigma = np.std(l_feats[:,f])
    l_feats[:,f] = (l_feats[:,f] - mu)/sigma    
    t_feats[:,f] = (t_feats[:,f] - mu)/sigma

knn = NearestNeighbors(n_neighbors=P)    
knn.fit(l_feats)
distance, neigh = knn.kneighbors(t_feats,
                         n_neighbors=P,
                         return_distance=True)

knnpiano = NearestNeighbors(n_neighbors=1)
knnpiano.fit(l_feats_piano)
distance_piano, neigh_piano = knnpiano.kneighbors(t_feats_piano,
                         n_neighbors=1,
                         return_distance=True)

# With this neigh vector we can perform all the synthesis type
print " CROSS-PLAIN - Piano"
data_cross_plain_piano =  resynth_sequence(neigh_piano[:,0], t_seg_starts,
                                        t_seg_duration,
                                        l_segments_piano, l_feats_piano,
                                        '', '.au', 22050,
                                        dotime_stretch=False,
                                        max_synth_idx=nb_max_seg,
                                        normalize=False,
                                        marge=5.0, verbose=False)
magspec_cross_plain_piano = np.abs(stft.stft(data_cross_plain_piano,512,128)[0,:,:])
Lmin = min(Lmin, magspec_cross_plain_piano.shape[1])
sig_cross_plain_piano = Signal(data_cross_plain_piano, 22050, normalize=True)
sig_cross_plain_piano.write(op.join(recons_audio_path, '_cross_plain_piano.wav'))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], magspec_cross_plain_piano[:,:Lmin])

print " CROSS-PLAIN - full dev"
data_cross_plain =  resynth_sequence(neigh[:,0], t_seg_starts,
                                        t_seg_duration,
                                        l_segments_all, l_feats,
                                        '', '.au', 22050,
                                        dotime_stretch=False,
                                        max_synth_idx=nb_max_seg,
                                        normalize=False,
                                        marge=marge, verbose=False)
magspec_cross_plain = np.abs(stft.stft(data_cross_plain,512,128)[0,:,:])
Lmin = min(Lmin, magspec_cross_plain.shape[1])
sig_cross_plain = Signal(data_cross_plain, 22050, normalize=True)
sig_cross_plain.write(op.join(recons_audio_path, '_cross_plain.wav'))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], magspec_cross_plain[:,:Lmin])


print " CROSS-NORMALIZED - Piano"
data_cross_normalized_piano =  resynth_sequence(neigh_piano[:,0], t_seg_starts,
                                        t_seg_duration,
                                        l_segments_piano, l_feats_piano,
                                        '', '.au', 22050,
                                        dotime_stretch=True,
                                        max_synth_idx=nb_max_seg,
                                        normalize=True,
                                        marge=marge, verbose=False)
magspec_cross_normalized_piano = np.abs(stft.stft(data_cross_normalized_piano,512,128)[0,:,:])
Lmin = min(Lmin, magspec_cross_normalized_piano.shape[1])
sig_cross_normalized_piano = Signal(data_cross_normalized_piano, 22050, normalize=True)
sig_cross_normalized_piano.write(op.join(recons_audio_path, '_cross_normalized_piano.wav'))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], magspec_cross_normalized_piano[:,:Lmin])

print " CROSS-NORMALIZED - full dev"
data_cross_normalized =  resynth_sequence(neigh[:,0], t_seg_starts,
                                        t_seg_duration,
                                        l_segments_all, l_feats,
                                        '', '.au', 22050,
                                        dotime_stretch=True,
                                        max_synth_idx=nb_max_seg,
                                        normalize=True,
                                        marge=marge, verbose=False)
magspec_cross_normalized = np.abs(stft.stft(data_cross_normalized,512,128)[0,:,:])
Lmin = min(Lmin, magspec_cross_normalized.shape[1])
sig_cross_normalized = Signal(data_cross_normalized, 22050, normalize=True)
sig_cross_normalized.write(op.join(recons_audio_path, '_cross_normalized.wav'))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], magspec_cross_normalized[:,:Lmin])

sig_data_cross_normalized = Signal(data_cross_normalized, 22050, normalize=True)

print "CROSS-Penalized - full dev"
vit_path = Viterbi(neigh, distance, t_penalty=0.01, c_value=20)
vit_cands = [neigh[ind,neighbind] for ind, neighbind in enumerate(vit_path)]
data_cross_penalized =  resynth_sequence(np.squeeze(vit_cands), t_seg_starts,
                                        t_seg_duration,
                                        l_segments_all, l_feats,
                                        '', '.au', 22050,
                                        dotime_stretch=True,
                                        max_synth_idx=nb_max_seg,
                                        normalize=True,
                                        marge=marge, verbose=False)
magspec_cross_penalized = np.abs(stft.stft(data_cross_penalized,512,128)[0,:,:])
Lmin = min(Lmin, magspec_cross_penalized.shape[1])
sig_cross_penalized = Signal(data_cross_penalized, 22050, normalize=True)
sig_cross_penalized.write(op.join(recons_audio_path, '_cross_penalized.wav'))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], magspec_cross_penalized[:,:Lmin])


print " ADDITIVE SYNTHESIS "
magspecs = []
for neighIdx in range(P):
    # build waveform
    sigout = resynth_sequence(np.squeeze(neigh[:,neighIdx]),
                              t_seg_starts,
                              t_seg_duration,
                               l_segments_all, l_feats, '', '.au', 22050,
                               dotime_stretch=False,
                               max_synth_idx=nb_max_seg,
                               marge=marge, normalize=True)
    # stft
    magspecs.append(np.abs(stft.stft(sigout,512,128)[0,:,:]))
magspecarr = np.array(magspecs)


Lmin = min(Lmin, magspecarr.shape[2])
print "Add-Max"
max_magspec = np.max(magspecarr, 0)
init_vec = np.random.randn(128*Lmin)
x_recon = gl_recons(max_magspec[:,:Lmin], init_vec, 20,
                                           512, 128, display=False)
sig_add_max = Signal(x_recon, fs, normalize=True)
sig_add_max.write(op.join(recons_audio_path, '_add_max_%s_P%d.wav'%(feat_comb,P)))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], max_magspec[:,:Lmin])
print "Add-Mean"
mean_magspec = np.mean(magspecarr, 0)
init_vec = np.random.randn(128*Lmin)
x_recon = gl_recons(mean_magspec[:,:Lmin], init_vec, 20,
                                           512, 128, display=False)
sig_add_mean = Signal(x_recon, fs, normalize=True)
sig_add_mean.write(op.join(recons_audio_path, '_add_mean_%s_P%d.wav'%(feat_comb,P)))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], mean_magspec[:,:Lmin])
print "Add-Median"
median_magspec = np.median(magspecarr, 0)
init_vec = np.random.randn(128*Lmin)
x_recon = gl_recons(median_magspec[:,:Lmin], init_vec, 20,
                                           512, 128, display=False)
sig_add_median = Signal(x_recon, fs, normalize=True)
sig_add_median.write(op.join(recons_audio_path, '_add_median_%s_P%d.wav'%(feat_comb,P)))
print "KL value %2.2f"%KLspec(orig_spec[:,:Lmin], median_magspec[:,:Lmin])

N = Lmin*128;
yticksvalues = np.arange(0.0,8001.0,4000.0).astype(int)
xticksvalues = np.arange(0.0,(N/22050),2).astype(int)
fs = 22050.0
Fmax = 257.0
yticks = np.floor((yticksvalues/(0.5*fs))*Fmax).astype(int)
xticks = (xticksvalues*fs/128).astype(int)


plt.figure(figsize=(14,10))
ax1 = plt.subplot(421)
plt.imshow(np.log(orig_spec[:,:Lmin]), origin='lower');
plt.xticks([]);plt.yticks(yticks,yticksvalues)
plt.ylabel('Frequency (Hz)',fontsize=16.0)
plt.title('(a)')
plt.subplot(422)
plt.imshow(np.log(magspec_ellis[:,:Lmin]), origin='lower');
plt.xticks([]);plt.yticks([])
plt.title('(b) : %2.2f'%KLspec(orig_spec[:,:Lmin], magspec_ellis[:,:Lmin]))
plt.subplot(423)
plt.imshow(np.log(magspec_cross_plain_piano[:,:Lmin]), origin='lower');
plt.xticks([]);plt.yticks(yticks,yticksvalues)
plt.ylabel('Frequency (Hz)',fontsize=16.0)
plt.title('(c) : %2.2f'%KLspec(orig_spec[:,:Lmin], magspec_cross_plain_piano[:,:Lmin]))
plt.subplot(424)
plt.imshow(np.log(median_magspec[:,:Lmin]), origin='lower');
plt.xticks([]);plt.yticks([])
plt.title('(d) : %2.2f'%KLspec(orig_spec[:,:Lmin], median_magspec[:,:Lmin]))
plt.subplot(425)
plt.imshow(np.log(magspec_cross_normalized_piano[:,:Lmin]), origin='lower');
plt.xticks([]);plt.yticks(yticks,yticksvalues)
plt.ylabel('Frequency (Hz)',fontsize=16.0)
plt.title('(e) : %2.2f'%KLspec(orig_spec[:,:Lmin], magspec_cross_normalized_piano[:,:Lmin]))
plt.subplot(426)
plt.imshow(np.log(mean_magspec[:,:Lmin]), origin='lower');
plt.xticks([]);plt.yticks([])
plt.title('(f) : %2.2f'%KLspec(orig_spec[:,:Lmin], mean_magspec[:,:Lmin]))
plt.subplot(427)
plt.imshow(np.log(magspec_cross_penalized[:,:Lmin]), origin='lower');
plt.xticks(xticks, xticksvalues);plt.yticks(yticks,yticksvalues)
plt.xlabel('Time (s)',fontsize=16.0)
plt.ylabel('Frequency (Hz)',fontsize=16.0)
plt.title('(g) : %2.2f'%KLspec(orig_spec[:,:Lmin], magspec_cross_penalized[:,:Lmin]))
plt.subplot(428)
plt.imshow(np.log(max_magspec[:,:Lmin]), origin='lower');
plt.xticks(xticks, xticksvalues);plt.yticks([])
plt.xlabel('Time (s)',fontsize=16.0)
plt.title('(h) : %2.2f'%KLspec(orig_spec[:,:Lmin], max_magspec[:,:Lmin]))
plt.subplots_adjust(left=0.08,bottom=0.07,top=0.97,right=0.97,hspace=0.14,wspace=0.03)
plt.savefig(os.path.join(output_fig_path,
                         '%s_%dsegments_%s_logspectro_P%d.pdf'%(t_name,nb_max_seg,feat_comb,P)))
plt.show()



