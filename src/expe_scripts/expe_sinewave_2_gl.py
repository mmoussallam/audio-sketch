'''
expe_scripts.expe_sinewave_2_gl  -  Created on Apr 4, 2013
what happens if we use a spectrogram from a sine wave speech in griffin and lim?
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
import stft
# load the sinewave speech
sinewave = Signal('/sons/sqam/vegaSWS.wav', mono=True)

spectro = stft.stft(sinewave.data, wsize=1024, tstep=256)[0,:,:]

init_vec = np.random.randn(sinewave.data.shape[0])

rec_gl_data = transforms.gl_recons(np.abs(spectro), init_vec, niter=20, wsize=1024, tstep=256)

sig_rec = Signal(rec_gl_data, sinewave.fs, mono=True, normalize=True)
sig_rec.write('/sons/sqam/vegaSWS_gl.wav')

# ok it's working just fine'
# now compare with reconstruction from original spectrogram
original = Signal('/sons/sqam/vega.wav', mono=True)
spectro = stft.stft(original.data, wsize=1024, tstep=256)[0,:,:]
init_vec = np.random.randn(original.data.shape[0])
rec_gl_data = transforms.gl_recons(np.abs(spectro), init_vec, niter=20, wsize=1024, tstep=256)
sig_rec = Signal(rec_gl_data, sinewave.fs, mono=True, normalize=True)
sig_rec.write('/sons/sqam/vega_gl.wav')