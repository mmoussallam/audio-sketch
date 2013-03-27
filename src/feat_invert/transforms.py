'''
feat_invert.transforms  -  Created on Feb 21, 2013
@author: M. Moussallam
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
import sys
import os.path as op
import os
import cv
import cv2
import stft
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
sys.path.append('/usr/local/lib')
sys.path.append('/usr/local/python_packages')


def get_stft(x, wsize=512, tstep=256, sigma=None):
    """ if necessary load the wav file and get the stft"""
    if isinstance(x, str):
        sig = Signal(x, mono=True, normalize=True)
        x = sig.data

    if sigma is not None:
        x += sigma*np.random.randn(*x.shape)

    return np.squeeze(stft.stft(x, wsize, tstep))


def get_istft(spect, wsize=512, tstep=256, L=None):
    """ reshape the spectrum and get the inverse Fourier transform """

    if len(spect.shape) < 3:
        spect = spect.reshape((1, spect.shape[0], spect.shape[1]))
    if L is not None:
        return np.squeeze(stft.istft(spect, tstep, L))

    return stft.istft(spect, tstep)


def gl_recons(magspec, init_vec, niter=10, wsize=512, tstep=256, display=False):
    """ A Griffin and Lim Based reconstruction method
        % reconstruct from a power spectrum
        % uses Signal class stft
        % must be initialized with a random vector or anything closer
        to the original target"""

    # initialize signal        
    x_rec = init_vec
    (K, P) = magspec.shape

    for n in range(niter):

        # compute stft of candidate
        S = get_stft(x_rec, wsize, tstep)        

        # estimate error        
        err = np.sum((np.abs(S[:]) - magspec[:]) ** 2) / np.sum(magspec[:] ** 2)
        print "Iteration %d: error of %1.6f " % (n, err)

        P_min = min(S.shape[1], P)

        # normalize its spectrum by target spectrum
        S *= magspec / np.abs(S)

        # resynthesize using inverse stft
        x_rec = get_istft(S, wsize, tstep, L=x_rec.shape[0])

    if display:
        plt.figure()
        plt.subplot(211)
        plt.plot(x_rec)
        plt.subplot(212)
        plt.imshow(np.log(np.abs(S)),
                   aspect='auto',
                   origin='lower',
                   interpolation='nearest')

    return x_rec
