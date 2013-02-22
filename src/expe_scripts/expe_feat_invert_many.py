'''
expe_scripts.expe_feat_invert_many  -  Created on Feb 21, 2013
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
from feat_invert import regression, transforms, features


learn_audiofilepath = '/sons/sqam/voicemale.wav'
test_audiofilepath = '/sons/sqam/voicefemale.wav'

audiosave_path ='/home/manu/workspace/audio-sketch/src/expe_scripts/audio/feat_invert/'

def RMSE(x,y):
    return 10.0*np.log10(np.sum((y**2))/np.sum((x - y)**2))

def test_feat_configuration(featlist, 
                            cov_fun,
                            reg_fun,
                            K=1, medfilt_width=10,
                            display=True, ngliter=5,
                            saveAudio=True,
                            win_size=512, step_size=128,
                            method='median'):
    """ run a feature inversion test on the desired configuration """
    
    YaafeDict = features.get_yaafe_dict(win_size, step_size)
    
    print "Starting test on features: %s with func %s"%(featlist, cov_fun.__name__)
    featuresList = [YaafeDict[feat] for feat in featlist]

    # Load the feature for train file
    # Load the features for test file 
    learn_feats = features.get_yaafe_features(featuresList, learn_audiofilepath)
    test_feats = features.get_yaafe_features(featuresList, test_audiofilepath)
    
    # get the features back
    devseq = []
    testseq = [];
    for key in learn_feats.keys():
        devseq.append(learn_feats[key].T)
        testseq.append(test_feats[key].T)

    # stacking all the descriptors
    Xdev = np.vstack(devseq)
    X = np.vstack(testseq)
    
    # get the magnitude spectrums
    Ydev = np.abs(transforms.get_stft(learn_audiofilepath, win_size,step_size, sigma = 0.001))
    Y = np.abs(transforms.get_stft(test_audiofilepath, win_size,step_size, sigma = 0.001))
    
    # Now use a regression technique 
    Y_hat, Ktest_dev = reg_fun(Xdev,Ydev,X,Y,
                                   cov_fun,
                                   display=False,
                                   K = K,
                                   method=method)
    
    # optionnal step: median filtering for smoothing the data:
    Y_hat = median_filter(Y_hat,(1,medfilt_width))
    
    if display:
        plt.figure()
        plt.subplot(211)
        plt.imshow(np.log(Y),
                   origin='lower')
        plt.colorbar()
        plt.title('Original')
        plt.subplot(212)
        plt.imshow(np.log(Y_hat),
                   origin='lower')
        plt.colorbar()
        plt.title('Estimation from %s - %s %d features'%(reg_fun.__name__,
                                                         cov_fun.__name__,
                                                        Xdev.shape[0]))
        
        
    sig_orig = Signal(test_audiofilepath,  normalize=True, mono=True)
    init_vec = np.random.randn(sig_orig.length)
    x_recon = transforms.gl_recons(Y_hat, init_vec, ngliter, win_size,
                                   step_size, display=display)
    
    
    sig = Signal(x_recon, 32000, normalize=True)
    
    if saveAudio:
        sig.write(op.join(audiosave_path,
                          'recons_%s_%s_%d%s_%dfeats_.wav'%(cov_fun.__name__,
                                                            reg_fun.__name__,
                                                        K,method,
                                                        Xdev.shape[0])))
    
#    err = 10.0*np.log10(np.sum((sig.data - sig_orig.data)**2)/np.sum((sig_orig.data**2)))
    return RMSE(sig.data, sig_orig.data), Xdev.shape[0]
#        plt.show()
    
        
###########################################
# listons les parameters
all_features = ['mfcc_d1','lpc-4','loudness','OnsetDet','mfcc']
#all_features = ['mfcc_d1','mfcc']
all_corr_func = {'cov':regression.corrcoeff_correl, 
                 'proj':regression.innerprod_correl}
all_reg_fun ={'gp':regression.nadaraya_watson,
             'odl':regression.online_learning}

cov_fun = all_corr_func['cov']
reg_fun = all_reg_fun['odl']


# Now we want all possible combinations of features:
from itertools import chain, combinations
s = list(all_features)
combis =  chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

results = [];

combis = [all_features,]

for feat_combi in combis:
    if len(feat_combi)==0:
        continue
    print feat_combi
    try:
        rmse, n_feats = test_feat_configuration(feat_combi,
                                                cov_fun,
                                                reg_fun,
                                                K=3,
                                                medfilt_width=10,
                                                display=True,
                                                ngliter=5,
                                                saveAudio=True, 
                                                win_size=512, step_size=128,
                                                method='median')
        
        results.append({'features':feat_combi,
                        'n_features':n_feats,
                        'cov_fun':cov_fun.__name__,
                        'reg_fun':reg_fun.__name__,
                        'Score':rmse})
        print rmse
    except:
        continue
    
max_index = np.argmax(np.array([i['Score'] for i in results]))

max_res = results[max_index]
print "Best score of %1.2f obtained for %s"%(max_res['Score'],list(max_res['features']))
plt.show()