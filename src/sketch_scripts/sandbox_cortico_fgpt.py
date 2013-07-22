'''
sketch_scripts.sandbox_cortico_fgpt  -  Created on Jul 22, 2013
@author: M. Moussallam

Ok so we want to see how the cortiocogram reacts to time shifts or translations
Is the sparsity pattern robust or How does it evolve?
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PyMP import Signal
import cProfile
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

from classes import sketch 
from tools.cochleo_tools import _cor2aud, Corticogram


audio_test_file = '/sons/jingles/panzani.wav'
fs = 8000
L = 2*fs

def expe1():
    shifts = [0,] # in samples
    fgpts = []
    for shift in shifts:
        sig =  Signal(audio_test_file, normalize=True, mono=True)
        sig.crop(shift, shift+L)
    
        sk = sketch.CorticoIHTSketch()    
        sk.recompute(sig)
        
        sk.sparsify(100)
        fgpts.append(sk.fgpt())
        
    #    sk.represent()
    #    plt.suptitle("Shift of %2.2f sec"%(float(shift)/float(fs)))
    colors = ['b', 'r', 'c','m']
    score = []
    bin_nnz_ref = np.flatnonzero(fgpts[0])
    #plt.figure()
    for i, fgpt in enumerate(fgpts):
        bin_nnz = np.flatnonzero(fgpt)
    #    plt.stem(bin_nnz,[1]*len(bin_nnz), colors[i])    
        score.append(len(np.intersect1d(bin_nnz_ref, bin_nnz, assume_unique=True)))
    
    print score


def gen_harmo_sig(freqs, L, fs):
    x = np.arange(0.0,float(L)/float(fs),1.0/float(fs))
    data = np.zeros(x.shape)
    for f in freqs:
        data += np.sin(2.0*np.pi*f*x)
    return Signal(data, fs, normalize=True, mono=True)


def gen_chirp_sig(freqs, L, fs, octave=2):
    x = np.arange(0.0,float(L)/float(fs),1.0/float(fs))
    data = np.zeros(x.shape)
    for fbase in freqs:
        f = np.linspace(fbase, fbase*(2**octave), L)
        data += np.sin(2.0*np.pi*f*x)
    return Signal(data, fs, normalize=True, mono=True)

sig = gen_chirp_sig([440.0, 512.0], L, fs)
sk = sketch.CorticoIHTSketch()
sk.recompute(sig)
sk.represent()

#sk.cort.plot_cort( cor=fgpts[1]-fgpts[0])
plt.show()