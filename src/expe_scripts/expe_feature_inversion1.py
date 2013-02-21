'''
expe_scripts.expe_feature_inversion1  -  Created on Feb 21, 2013
@author: M. Moussallam
'''
# How could we learn an invert transform from the feature space to the data?
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
import sys
import os.path as op
import os
import cv, cv2
from feat_invert import regression, transforms, features
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
sys.path.append('/usr/local/lib')
#sys.path.append('/usr/local/python_packages')

from yaafelib import *
import stft
import spams

from scipy.ndimage.filters import median_filter

win_size = 512;
step_size = 128;

# Learning phase
learn_audiofilepath = '/sons/sqam/voicemale.wav'
test_audiofilepath = '/sons/sqam/voicefemale.wav'
featuresList = [
#                {'name':'mfcc',
#                 'featName':'MFCC',
#                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
                {'name':'zcr',
                 'featName':'ZCR',
                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
                {'name':'Loudness',
                 'featName':'Loudness',
                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
                {'name':'lpc',
                 'featName':'LPC',
                 'params':'LPCNbCoeffs=8 blockSize=%d stepSize=%d'%(win_size,step_size)},
                {'name':'ComplexDomainOnsetDetection',
                 'featName':'ComplexDomainOnsetDetection',
                 'params':'blockSize=%d stepSize=%d'%(win_size,step_size)}
                ]


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
#Xdev = np.hstack((learn_feats['zcr'].T, learn_feats['mfcc'].T,
#                       learn_feats['Loudness'].T, learn_feats['lpc'].T))
#X =  np.hstack((test_feats['zcr'].T, test_feats['mfcc'].T,
#                    test_feats['Loudness'].T, test_feats['lpc'].T))
#Xdev = learn_feats['mfcc'].T
#X = test_feats['mfcc'].T

# we also want to use the transforms
Ydev = np.abs(transforms.get_stft(learn_audiofilepath, win_size,step_size, sigma = 0.001))
Y = np.abs(transforms.get_stft(test_audiofilepath, win_size,step_size, sigma = 0.001))


#Kdev = regression.load_correl(X,Y)
# truncate to fit
#Ydev = Ydev[:,0:Kdev.shape[1]]
#Y = Y[:,0:Kdev.shape[0]]
#Xdev = Xdev[:,0:Kdev.shape[1]]
#X = X[:,0:Kdev.shape[0]]

# Now use a regression technique 
Y_hat, Ktest_dev = regression.nadaraya_watson(Xdev,Ydev,X,Y,
                                   regression.corrcoeff_correl,
                                   display=False,K = 1)

#plt.figure();plt.imshow(Ktest_dev); 
#plt.colorbar()
#plt.show()

# optionnal step: median filtering for smoothing the data:
Y_hat = median_filter(Y_hat,(1,10))

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
plt.title('Estimation from Nadaraya-Watson')
plt.show()

sig_orig = Signal(test_audiofilepath,  normalize=True, mono=True)
#init_vec = np.random.randn(step_size*Y_hat.shape[1])
init_vec = np.random.randn(sig_orig.length)
x_recon = transforms.gl_recons(Y_hat, init_vec, 10, win_size, step_size, display=True)
plt.show()

sig = Signal(x_recon, 32000, normalize=True)

err = 10.0*np.log10(np.sum((sig.data - sig_orig.data)**2)/np.sum((sig_orig.data**2)))
