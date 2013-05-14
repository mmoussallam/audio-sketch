'''
expe_scripts.invert.visu_segmentation  -  Created on Apr 30, 2013
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
worpspace_path = '/home/manumouss/workspace/git/'
sys.path.append(worpspace_path+'PyMP')
from PyMP import Signal
import sys
import os



sys.path.append(worpspace_path+'audio-sketch')
sys.path.append(worpspace_path+'PyMP')
sys.path.append(worpspace_path+'MEG_Denoise')
sys.path.append('/usr/local/lib')
#sys.path.append('/home/manu/workspace/toolboxes/MSongsDB-master/PythonSrc')
sys.path.append('/home/manumouss/workspace/git/MSongsDB-master/PythonSrc')
sys.path.append('/usr/local/python_packages')

#import stft
sys.path.append(worpspace_path+'audio-sketch/src')
from tools.learning_tools import find_indexes, get_ten_features, get_ten_features_from_file, get_track_info, resynth
from feat_invert import regression, transforms, features


# load the audio data and the features
audio_file_path = '/home/manumouss/workspace/git/audio-sketch/src/expe_scripts/invert/rwc-g-m01_1.wav'
output_path = '/home/manu/workspace/audio-sketch/src/results/audio'
orig_sig = Signal(audio_file_path, normalize=True)

test_file = 'rwc-g-m01_1'
h5_file_path = '/home/manumouss/workspace/git/audio-sketch/src/expe_scripts/invert/rwc-g-m01_1.h5'

#test_file = '011PFNOF.WAV'
#h5_file_path = '/sons/rwc/Piano/hdf5/011PFNOF.h5'
feats = []
segs = []
get_ten_features_from_file(feats, segs, [], h5_file_path)

# plot part of the audio and teh segmentation
seg_starts = segs[0][0]
seg_duration = np.diff(seg_starts)
loud_start = feats[0][:,12]
loud_max = feats[0][:,13]
loud_max_time = feats[0][:,14]
feats =  feats[0]

nseg = 5
max_time = seg_starts[nseg-1]# + seg_duration[nseg-1]
fs = orig_sig.fs

X,Y = np.meshgrid(seg_starts[:nseg],range(12))

fig=plt.figure(figsize=(10,3))
ax1 = plt.subplot(311)
plt.plot(np.linspace(0, max_time, int(max_time*orig_sig.fs)),orig_sig.data[0: int(max_time*orig_sig.fs)])
plt.stem(seg_starts[:nseg], 0.15*np.ones((nseg,)), linefmt='k-', markerfmt='s')    
plt.stem(seg_starts[:nseg] + loud_max_time[:nseg], 2.0**(loud_max[:nseg]/10.0), linefmt='r-', markerfmt='o')    
#plt.legend(('Waveform','Segment Starts', 'Loudness Peaks'),loc='lower right')
plt.xlim([-0.05,1.2])
plt.subplot(312, sharex=ax1)
ax2 = plt.pcolor(X,Y, feats[:nseg,-12:].T, cmap=cm.copper_r)
#ax_col = fig.add_axes([0.6, 0.35, 0.1, 0.2])
#plt.colorbar( mappable=ax2, ax=ax_col)
plt.subplot(313, sharex=ax1)
plt.pcolor(X,Y,feats[:nseg,:12].T, cmap=cm.copper_r)
#plt.colorbar()
plt.xlabel('Time (s)')
plt.show()

# How is Loudness related to energy of my elements?
segIdx = 0
subdata = orig_sig.data[seg_starts[segIdx]*fs: seg_starts[segIdx]*fs+seg_duration[segIdx]*fs]
print "True Loudness max of : ", feats[0][segIdx,12:15]


#######
#test_feats = feats[0][0:nseg,:]
#
## Load the learned features
#learntype = 'Piano'
#ext = '.WAV'
#ref_audio_dir = '/sons/rwc/%s/'%learntype
#h5_dir =  '/sons/rwc/%s/hdf5/'%learntype
## load the learned features and segments
## for each segment, we must keep a trace of the original corresponding file
#learn_feats, learn_segs, learn_confidence = get_ten_features(h5_dir)
#
## get the candidates accodring to the chroma features
#from sklearn.neighbors import NearestNeighbors
#n_neighbs_chroma = 10
#chroma_neigh = NearestNeighbors(n_neighbs_chroma)
#
#n_feats = 27
#
#chroma_neigh.fit(learn_feats[:,-n_feats:])
#distance, chroma_candidates = chroma_neigh.kneighbors(test_feats[:,-n_feats:])
#
## what about the loudness on the candidates ? penalizes it!
#ref_ind_no_penalty = chroma_candidates[:,0]
#ref_ind_pen_loud = np.zeros_like(ref_ind_no_penalty)
#
#lambda_L = 0.1
#for segIdx in range(nseg):
#    print "loudness: ", feats[0][segIdx,12:15] 
#    DeltaL = feats[0][segIdx,13] -  feats[0][segIdx,12]
#    n_distance = np.array(distance[segIdx,:])
#    for c in range(n_neighbs_chroma):
#        print "Cand %d: "%c, learn_feats[chroma_candidates[segIdx,c],12:15], distance[segIdx,c]
#        cand_DeltaL = learn_feats[chroma_candidates[segIdx,c],13] - learn_feats[chroma_candidates[segIdx,c],12]
#        n_distance[c] += lambda_L*np.abs(cand_DeltaL - DeltaL)
#    b_c = np.argmin(n_distance)
#    ref_ind_pen_loud[segIdx] = chroma_candidates[segIdx, b_c]
#    print "New best candidate is %d score of %1.4f"%(b_c, n_distance[b_c]) , n_distance
#
##plt.figure()
##plt.plot(ref_ind_no_penalty)
##plt.plot(ref_ind_pen_loud,'r')
##plt.show()
#
################# Direct resynthesis
#from tools.learning_tools import resynth, save_audio
## No penalty no stretching 
##sigout_no_pen =  resynth(ref_ind_no_penalty, seg_starts, seg_duration, learn_segs,
##                  learn_feats, ref_audio_dir, ext,
##                  dotime_stretch=False)
##aud_str = learntype+'_' +test_file +'_chrom_testing_nopen_nostretch'
##save_audio(output_path, aud_str, sigout_no_pen, fs, norm_segments=False)
#
###  No penalty but time stretching
#sigout_no_pen_stretch =  resynth(ref_ind_no_penalty, seg_starts, seg_duration, learn_segs,
#                  learn_feats, ref_audio_dir, ext,
#                  dotime_stretch=True)
#aud_str = learntype+'_' +test_file +'_chrom_testing_nopen_stretch'
#save_audio(output_path, aud_str, sigout_no_pen_stretch, fs, norm_segments=False)  
#
###  With penalty no stretching
##sigout_pen_loud =  resynth(ref_ind_pen_loud, seg_starts, seg_duration, learn_segs,
##                  learn_feats, ref_audio_dir, ext,
##                  dotime_stretch=False)
##aud_str_pen = learntype+'_' +test_file +'_chrom_testing_penloud_nostretch'
##save_audio(output_path, aud_str_pen, sigout_pen_loud, fs, norm_segments=False)
##
###  With penalty with stretching
#sigout_pen_loud_stretch =  resynth(ref_ind_pen_loud, seg_starts, seg_duration, learn_segs,
#                  learn_feats, ref_audio_dir, ext,
#                  dotime_stretch=True)
#aud_str_pen = learntype+'_' +test_file +'_chrom_testing_penloud_stretch'
#save_audio(output_path, aud_str_pen, sigout_pen_loud_stretch, fs, norm_segments=False)      
#
#
############### we can also take the median spectrogram of all candidates
#magspeccand = []
#for cand in range(n_neighbs_chroma):
#    ref_ind = chroma_candidates[:,cand]
#    sigout = resynth(ref_ind, seg_starts, seg_duration, learn_segs,
#                  learn_feats, ref_audio_dir, ext,
#                  dotime_stretch=True)
#    magspeccand.append(np.abs(stft.stft(np.concatenate(sigout), wsize=1024, tstep=256)[0,:,:]))
#
#spectensor = np.array(magspeccand) 
#orig_sig.crop(0, int((seg_starts[nseg]+seg_duration[nseg])*orig_sig.fs))
#plt.figure()
#plt.subplot(311)
#orig_sig.spectrogram(1024, 256, order=1, log=True, cbar=False)
#plt.subplot(312)
#plt.imshow(np.log10(np.mean(spectensor, axis=0)), origin='lower')
#plt.subplot(313)
#plt.imshow(np.log10(np.median(spectensor, axis=0)), origin='lower')
#
#plt.show()
#
##### resynthesize 'em
#from feat_invert.transforms import gl_recons
#init_vec = np.random.randn(spectensor.shape[-1]*256)
#
#mean_recons = gl_recons(np.mean(spectensor, axis=0), init_vec, niter=20, wsize=1024, tstep=256)
#sig_mean_recons = Signal(mean_recons, orig_sig.fs, mono=True)
#sig_mean_recons.write(output_path +'/'+ learntype+'_'+str(n_feats)+'feats_' +test_file +'_chrom_mean.wav')