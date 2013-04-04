'''
expe_scripts.expe_synth_from_30s  -  Created on Apr 2, 2013

OK so we want to synthesize a complete file based on spectrums from its first 30 seconds
(can't we even keep the phase from the original spectrums?)

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
from feat_invert.features import load_data_one_audio_file


wintime = 0.016
steptime= 0.004
sr = 32000
frame_num_per_file = 200000
features = ['zcr','OnsetDet','energy','specstats','mfcc','chroma','pcp']
filepath = '/sons/rwc/Test/rwc-g-m04_4.wav'



[MagSpectrums, Feats, Datas] = load_data_one_audio_file(filepath, 
                                                     sr,
                                                     sigma_noise=0,
                                                     wintime=wintime,
                                                     steptime=steptime,
                                                     max_frame_num_per_file=frame_num_per_file,
                                                     startpoint = 0,
                                                     features=features)

# Ok se now we keep only the first 10 % as training
start_l_ratio = 0.1
learn_ratio = 0.20
start_t_ratio = 0.7
test_ratio =  0.1

nb_learn_frames = int(learn_ratio * MagSpectrums.shape[0])
start_l_frame = int(start_l_ratio * MagSpectrums.shape[0])
nb_test_frames = int(test_ratio * MagSpectrums.shape[0])
start_t_frame = int(start_t_ratio * MagSpectrums.shape[0])

# ok now we add samples from the pre-computed collection


learn_feats = np.zeros((0,68))
learn_magspecs = np.zeros((0,257))

AddCollection  = True
AddSample = True
add_col_str = ''
add_sample_str = ''

if AddSample:
    add_sample_str = 'add_%d_sample'%int(100*learn_ratio)
    learn_feats = Feats[start_l_frame:start_l_frame+nb_learn_frames,:]
    learn_magspecs = MagSpectrums[start_l_frame:start_l_frame+nb_learn_frames,:]
    
# concatenate 
if AddCollection:
    from scipy.io import loadmat
    full_path = '/home/manu/workspace/audio-sketch/matlab/'
    savematname = 'learnbase_allfeats_2000000_seed_78.mat'
    lstruct = loadmat(full_path + savematname)
    learn_feats_all = lstruct['learn_feats_all']
    learn_magspecs_all = lstruct['learn_magspecs_all']
    learn_files = lstruct['learn_files']
    add_col_str = 'add_%s_col'%learn_magspecs_all.shape[0]
    learn_feats = np.concatenate((learn_feats, learn_feats_all))
    learn_magspecs = np.concatenate((learn_magspecs, learn_magspecs_all))



test_feats = Feats[start_t_frame:start_t_frame+nb_test_frames,:]
test_magspecs = Feats[start_t_frame:start_t_frame+nb_test_frames,:]

learn_sample = learn_ratio * Datas.shape[0]
start_l_sample  = start_l_ratio * Datas.shape[0] 
test_sample = test_ratio * Datas.shape[0] 
start_t_sample  = start_t_ratio * Datas.shape[0]

ref_learn_data = Datas[start_l_sample:start_l_sample+learn_sample]
sig_learn_ref = Signal(ref_learn_data, sr)
ref_test_data = Datas[start_t_sample:start_t_sample+test_sample]
sig_test_ref = Signal(ref_test_data, sr)


nb_median = 5
nb_iter_gl = 20
l_medfilt = 1
params = {}
params['win_size'] = int(wintime*sr)
params['step_size'] = int(steptime*sr)

res_array = regression.eval_knn( learn_feats,
                                 learn_magspecs,
                                 test_feats , 
                                 test_magspecs,
                                 ref_test_data, 
                                 nb_median, nb_iter_gl,
                                 l_medfilt, params)


output_path = '/home/manu/workspace/audio-sketch/src/results/'
res_sig = Signal(res_array[1], sr, mono=True, normalize=True)


res_sig.write(output_path+'audio/test_rwc-g-m01_4_learn_%s%s_%dmedian.wav'%(add_sample_str,add_col_str,nb_median))
sig_test_ref.write(output_path+'audio/ref_rwc-g-m04_4_learn_%d.wav'%int(100*learn_ratio))

#plt.figure()
#plt.plot(res_array[2])
#plt.show()
## terrible idea: use the waveforms directly?
#Xdev = learn_feats
#Ydev = learn_feats
#estimated_windowed_wf = regression.ann(Xdev, Ydev, X, Y, display=False, K=1)
