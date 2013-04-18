'''
expe_scripts.expe_knn_fi_mfcc_vs_chroma  -  Created on Apr 12, 2013

Mix two best candidates: the one based on the mfcc and the one based on the chroma

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

display = True

# for all combinations of these parameters
nb_learns = [2000000,]
nb_medians = [1,]
nb_features = [68,]
nb_trials = 1
method = 1

# evaluate using these parameters
nb_iter_gl = 10
nb_test = 5000
l_medfilt = 3


#learn_directory = '/sons/voxforge/main/Learn'
#test_directory = '/sons/voxforge/main/Test'
#startpoint = 0
#learn_seed = 13

learn_directory = '/sons/rwc/Learn'
test_directory = '/sons/rwc/Test'
startpoint = 0 # in seconds
learn_seed = 78
force_recompute = False

params = {}
params['n_frames'] = max(nb_learns)
#params.n_frames = nb_learn;
#params.sigma = 0.00001;
params['shuffle'] = learn_seed
params['wintime'] = 0.032
params['steptime'] = 0.008
params['sr'] = 16000
params['frame_num_per_file'] = 2000
params['features'] = ['zcr','OnsetDet','energy','specstats','mfcc','chroma','pcp']
params['location'] = learn_directory

full_path = '/home/manu/workspace/audio-sketch/matlab/'
output_path = '/home/manu/workspace/audio-sketch/src/results/'
savematname = 'learnbase_allfeats_%d_seed_%d.mat'%(params['n_frames'], params['shuffle'])

if not os.path.exists(full_path + savematname) or force_recompute:
    from scipy.io import savemat
    [learn_feats_all, learn_magspecs_all,
        n_f_learn, ref_learn_data, learn_files] = features.load_yaafedata(params)
    savemat(full_path + savematname, {'learn_feats_all':learn_feats_all,
                                       'learn_magspecs_all':learn_magspecs_all,
                                        'learn_files':learn_files})
else:
    print "LOADING"
    from scipy.io import loadmat
    lstruct = loadmat(full_path + savematname)
    learn_feats_all = lstruct['learn_feats_all']
    learn_magspecs_all = lstruct['learn_magspecs_all']
    learn_files = lstruct['learn_files']
    print learn_magspecs_all.shape

# Processing the requested elements
for trialIdx in range(nb_trials):
    isinbase = True
    
    while isinbase:
        # get the test data
        params['n_frames'] = nb_test
        params['sigma'] = 0.0
        params['shuffle'] =  321
        params['startpoint'] = startpoint
        # very important: need to look for test that is not in learn                        
        params['location'] = test_directory
        params['forbidden_names'] = [os.path.basename(i) for i in learn_files]
        [test_feats_all, test_magspecs, n_f_test,
            ref_t_data, test_files] = load_yaafedata(params);
            
        # search for any test file that is already in the learning set
        isinbase = any([os.path.basename(p) in params['forbidden_names'] for p in test_files])    
    
    
    save_test_name = 'test_audio_seed_%d_%d_trial%s'% (learn_seed,
                                           params['shuffle'], trialIdx)
 
    # also save the audio
    res_sig = Signal(ref_t_data, params['sr'], mono=True, normalize=True)
    res_sig.write(output_path+'audio/'+save_test_name+'.wav')
    for nli in range(len(nb_learns)):
        nb_learn = nb_learns[nli]
        
        for mfi in range(len(nb_features)):
            nb_feat = nb_features[mfi]
            
            
            learn_feats = learn_feats_all[0:nb_learn,0:nb_feat]
            learn_magspecs = learn_magspecs_all[:, 0:nb_learn]

            test_feats = test_feats_all[:,0:nb_feat]
            
            for nmi in range(len(nb_medians)):
                nb_median = nb_medians[nmi]                
                
                # Getting the spectrum with all features considered
                print "Getting the full part"                
                estimated_spectrum_full, neighbors = regression.ann(learn_feats[:,0:20].T, learn_magspecs.T,
                             test_feats[:,0:20].T, test_magspecs.T,
                             K=nb_median)

                print "Getting the harmonic part"
                estimated_spectrum_harmo, neighbors = regression.ann(learn_feats[:,-48:-36].T, learn_magspecs.T,
                             test_feats[:,-48:-36].T, test_magspecs.T,
                             K=nb_median)

                win_size = params['wintime']*params['sr']
                step_size = params['steptime']*params['sr']
                # sliding median filtering ?
                if l_medfilt > 1:
                    estimated_spectrum = median_filter(estimated_spectrum_full + estimated_spectrum_harmo, (1, l_medfilt))
            
                print "reconstruction"    
                
                #init_vec = np.random.randn(step_size*Y_hat.shape[1])
                init_vec = np.random.randn(step_size*estimated_spectrum.shape[1])
                x_recon = transforms.gl_recons(estimated_spectrum, init_vec, nb_iter_gl,
                                               win_size, step_size, display=False)
                
                # Get the rythmic part by using all coefficients  
#                res_array = regression.eval_knn( learn_feats[:,0:20], learn_magspecs,
#                                                 test_feats[:,0:20] , 
#                                                 test_magspecs, ref_t_data, 
#                                                 nb_median, nb_iter_gl,
#                                                 l_medfilt, params)
#                
#                # now get a harmonic candidate bu using only the chroma coefficients
#                res_array_harmo = regression.eval_knn( learn_feats[:,-48:], learn_magspecs,
#                                                 test_feats[:,-48:] , 
#                                                 test_magspecs, ref_t_data, 
#                                                 nb_median, nb_iter_gl,
#                                                 l_medfilt, params)
                
                # save the data
                save_res_name = 'result_harmo_spec_%d_%d_%d_seed_%d_%d_trial%s'% (nb_learn,
                                           nb_feat, nb_median, learn_seed,
                                           params['shuffle'], trialIdx)
#                np.save(output_path+save_res_name+'.npy', res_array[0])
                
                # also save the audio
                res_sig = Signal(x_recon, params['sr'], mono=True, normalize=True)
                res_sig.write(output_path+'audio/'+save_res_name+'.wav')
                
                
                if display:
                    plt.figure()
                    plt.subplot(211)
                    plt.imshow(np.log10(test_magspecs.T), origin='lower')
                    plt.title('Original')
                    plt.subplot(212)
                    plt.imshow(np.log10(estimated_spectrum), origin='lower')
                    plt.title('Reconstructed, base: %d - feats: %d feats - medians: %d'%(
                                                        nb_learn, nb_feat, nb_median))
                    

plt.show()