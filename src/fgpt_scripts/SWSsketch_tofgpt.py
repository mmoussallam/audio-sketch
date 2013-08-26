'''
fgpt_scripts.SWSsketch_tofgpt  -  Created on Jul 29, 2013
@author: M. Moussallam


How to build a fingerprint with a SineWave Speech Representation?

'''

import os
import sys
import numpy as np
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

#from classes import pydb, sketch
from classes.pydb import *
from classes.sketches.misc import SWSSketch
from PyMP.signals import LongSignal, Signal
import os.path as op
import matplotlib.pyplot as plt

single_test_file1 = '/sons/jingles/panzani.wav'

sk = SWSSketch()
sk.recompute(single_test_file1)

#sk.sparsify(3000)
#sk.represent(sparse=True)
#
#synth3000 = sk.synthesize(sparse=True)
#
#sk.sparsify(300)   
#sk.represent(sparse=True)
#synth300 = sk.synthesize(sparse=True)

sk.sparsify(3000)
print sk.params   
y_vec = sk.formants[1]
print y_vec.shape
import stft

sig = Signal(single_test_file1, mono=True)
#wsize= floor(sk.params['windowSize']*sig.fs)
#tstep = floor(sk.params['time_step']*sig.fs)
#X = sig.spectrogram(4096, 512,log=False, cbar=False)
wsize = 1024
tstep = 512
X = stft.stft(sig.data, wsize, tstep)

mask = np.zeros_like(X)
deg = 10
appmask = np.zeros_like(X)
binappmask = np.zeros_like(X)
#

plt.figure()
ax2 = plt.imshow(np.abs(X[0,:,:]), extent=[0,y_vec.shape[0],0,sig.fs/2], origin='lower')
for n_form in range(3):
    y_vec = sk.formants[n_form]
    x_vec = np.arange(0.0,sk.orig_signal.get_duration(),sk.orig_signal.get_duration()/len(y_vec))
    plt.plot(y_vec, linewidth=3.0)
    xxes = np.floor(np.linspace(0, X.shape[2]-1, y_vec.shape[0])).astype(int)
    yxes = ((y_vec / 44100.0)*wsize).astype(int)
    mask[0,yxes, xxes]=X[0,yxes, xxes]
    
    # now the approximated version
    p = np.polyfit(x_vec, y_vec, deg)
    poly = np.zeros_like(y_vec)
    for i in range(deg+1):
        poly += p[-i-1]*(x_vec ** i)
    appyxes = ((poly / 44100.0)*wsize).astype(int)
    appmask[0,appyxes, xxes]=X[0,appyxes, xxes]
    binappmask[0,appyxes, xxes]=1


#plt.figure()
#
#plt.imshow(np.abs(mask[0,:,:]))
#plt.show()

plt.figure()
plt.plot(sk.formants[2]-sk.formants[1])
plt.plot(sk.formants[1]-sk.formants[0],'r')
plt.show()

rec = stft.istft(appmask, tstep)
rec_sig = Signal(rec[0,:], sig.fs, mono=True, normalize=True)
binrec = stft.istft(binappmask, tstep)
binrec_sig = Signal(binrec[0,:], sig.fs, mono=True, normalize=True)
#sk.represent(sparse=True)
#synth30 = sk.synthesize(sparse=True)
#
##sk.params['n_formants']=5
##sk.sparsify(30000)   
##sk.represent(sparse=True)
##synth30000 = sk.synthesize(sparse=True)
#
#plt.show()
#
#plt.figure()
#for formIdx in range(3):
#    y_vec = sk.formants[formIdx]
#    x_vec = np.arange(0.0,sk.orig_signal.get_duration(),sk.orig_signal.get_duration()/len(y_vec))    
#    # 
#    deg = 30
#    p = np.polyfit(x_vec, y_vec, deg)
#    rec = np.zeros_like(y_vec)
#    for i in range(deg+1):
#        rec += p[-i-1]*(x_vec ** i)
#        
        
#    plt.plot(x_vec, y_vec)
#    plt.plot(x_vec, rec)
#plt.show()
