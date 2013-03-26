'''
expe_scripts.expe_feature_inversion1  -  Created on Feb 21, 2013
@author: M. Moussallam
'''
# How could we learn an invert transform from the feature space to the data?
import numpy as np
import matplotlib.pyplot as plt
from PyMP import Signal
import sys
from feat_invert import regression, transforms, features
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
sys.path.append('/usr/local/lib')
#sys.path.append('/usr/local/python_packages')

from yaafelib import *
from scipy.ndimage.filters import median_filter

win_size = 512;
step_size = 128;

# Learning phase
learn_audiofilepath = '/sons/sqam/voicemale.wav'
test_audiofilepath = '/sons/sqam/voicefemale.wav'
#featuresList = [
##                {'name':'mfcc',
##                 'featName':'MFCC',
##                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
#                {'name':'zcr',
#                 'featName':'ZCR',
#                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
#                {'name':'Loudness',
#                 'featName':'Loudness',
#                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
#                {'name':'lpc',
#                 'featName':'LPC',
#                 'params':'LPCNbCoeffs=8 blockSize=%d stepSize=%d'%(win_size,step_size)},
#                {'name':'ComplexDomainOnsetDetection',
#                 'featName':'ComplexDomainOnsetDetection',
#                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)}
#                ]
yaafe_dict = features.get_yaafe_dict(win_size,step_size)
all_feat_name_list = ['zcr','OnsetDet','energy','specstats','specflux','mfcc','magspec']
featuresList = []
for feat_name in all_feat_name_list:
    featuresList.append(yaafe_dict[feat_name])

tested_features = ['zcr','OnsetDet','energy','specstats']

learn_feats = features.get_yaafe_features(featuresList, learn_audiofilepath)
test_feats = features.get_yaafe_features(featuresList, test_audiofilepath)

# get the features back
devseq = []
testseq = [];
for key in tested_features:
    devseq.append(learn_feats[key].T)
    testseq.append(test_feats[key].T)

# stacking all the descriptors
Xdev = np.vstack(devseq)
X = np.vstack(testseq)

Ydev = learn_feats['magspec'].T
Y = test_feats['magspec'].T

# we also want to use the transforms
Ydev_old = np.abs(transforms.get_stft(learn_audiofilepath, win_size,step_size, sigma = 0.001))
#Y = np.abs(transforms.get_stft(test_audiofilepath, win_size,step_size, sigma = 0.001))


# Now use a regression technique 
import time
t = time.time()
Y_hat, Ktest_dev = regression.knn(Xdev,Ydev,X,Y,
                                   regression.corrcoeff_correl,
                                   display=False,
                                   K = 2)
print "Took ", time.time() - t , " secs"

t = time.time()
Y_hat = regression.ann(Xdev, Ydev, X, Y, K=5)
print "Took ", time.time() - t , " secs"
#plt.figure();plt.imshow(Ktest_dev); 
#plt.colorbar()
#plt.show()

# optionnal step: median filtering for smoothing the data:
Y_hat = median_filter(Y_hat,(1,10))

#plt.figure()
#plt.subplot(211)
#plt.imshow(np.log(Y),
#           origin='lower')
#plt.colorbar()
#plt.title('Original')
#plt.subplot(212)
#plt.imshow(np.log(Y_hat),
#           origin='lower')
#plt.colorbar()
#plt.title('Estimation from Nadaraya-Watson')
#plt.show()

sig_orig = Signal(test_audiofilepath,  normalize=True, mono=True)
#init_vec = np.random.randn(step_size*Y_hat.shape[1])
init_vec = np.random.randn(sig_orig.length)
x_recon = transforms.gl_recons(Y_hat, init_vec, 10, win_size, step_size, display=False)
plt.show()

sig_recon = Signal(x_recon, 32000, normalize=True)

err = 10.0*np.log10(np.sum((sig_recon.data - sig_orig.data)**2)/np.sum((sig_orig.data**2)))
