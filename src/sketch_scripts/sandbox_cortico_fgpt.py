'''
sketch_scripts.sandbox_cortico_fgpt  -  Created on Jul 22, 2013
@author: M. Moussallam

Ok so we want to see how the cortiocogram reacts to time shifts or translations
Is the sparsity pattern robust or How does it evolve?
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyMP import Signal
import cProfile
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

from classes.sketches.cortico import * 
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

def gen_vibrato_sig(freqs, L, fs, octave=2, rate=0.1, ratio=0.1):
    x = np.arange(0.0,float(L)/float(fs),1.0/float(fs))
    data = np.zeros(x.shape)
    from scipy.special import sici
    for fbase in freqs:
        # merci wolfram
        f = fbase*(1.0 + ratio*sici(2.0*np.pi*rate*x)[1])   
        data += np.sin(2.0*np.pi*f*x)
    return Signal(data, fs, normalize=True, mono=True), f*x

def expe2():
    sig = gen_chirp_sig([440.0, 512.0], L, fs)
    
    sig = Signal('/sons/sqam/voicemale.wav', mono=True, normalize=True)
    #sig, c = gen_vibrato_sig([440.0,], L, fs, rate=10,ratio=0.1)
    sk = sketch.CorticoSketch()
    sk.recompute(sig)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.abs(sk.cort.cor[-1,0,:,:]))
    plt.subplot(122)
    plt.imshow(np.abs(sk.cort.cor[0,-1,:,:]))
    plt.show()
    
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.abs(sk.cort.cor[-1,0,:,:]))
    plt.subplot(212)
    plt.plot(np.sum(np.abs(sk.cort.cor[-1,0,:,:]), axis=0))
    #plt.show()
    
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.abs(sk.cort.cor[0,-1,:,:].T))
    plt.subplot(212)
    plt.plot(np.sum(np.abs(sk.cort.cor[0,-1,:,:]), axis=1))
    plt.show()

sig = Signal('/sons/jingles/panzani.wav', mono=True, normalize=True)
#sig.crop(0, 2*sig.fs)
sk = CorticoSubPeaksSketch(**{'downsample':8000, 'n_inv_iter':10})
sk.recompute(sig)

combis = [(0,6),(4,6),(0,11),(4,11)]
for combi in combis:
    sk.sp_rep = np.zeros_like(sk.rep)
    sk.sp_rep[combi[0], combi[1], :,:] = sk.rep[combi[0], combi[1], :,:]
    sk.represent(sparse=True)
    synth_sig = sk.synthesize(sparse=True)
    synth_sig.write('SubCortico_%d_%d.wav'%(combi[0], combi[1]))
#sk.sparsify(1000)
#sk.represent(sparse=True)
#sk.represent()

#sk.cort.plot_cort(cor=sk.sp_rep, binary=False)


#
#plt.figure()
#plt.imshow(np.abs(sk.sp_rep[0,6,:,:]), cmap=cm.bone_r)
#plt.colorbar()
#plt.show()
#
##sk.cort.plot_cort( cor=fgpts[1]-fgpts[0])
plt.show()