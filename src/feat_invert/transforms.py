'''
feat_invert.transforms  -  Created on Feb 21, 2013
@author: M. Moussallam
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
from PyMP.signals import LongSignal
import sys
import os.path as op
import os
import cv
import cv2
import stft
from math import pi
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


def spec_morph(learn_specs, target_length, neighb_segments, l_seg_bounds):
    """ given a spectrogram and :
        - l_segments : list of (start, end) pairs of indices
        - neighb_segments :  list of (segment index, segment lengths)        
        do a morphing and build a candidate spectrogram """
    # total size of new spectrogram
    tstep = (learn_specs.shape[1]-1)    # Assume 50% overlap
    n_t_segments = len(neighb_segments)    
    # initialize new spectrogram
    morphed_spectro = (1e-5)*np.ones((target_length/tstep, learn_specs.shape[1]))
    cur_target_idx = 0  # current frame idx
    for segI in range(n_t_segments):
        ref_seg_idx = neighb_segments[segI][0]
        target_seg_length = neighb_segments[segI][1]
        # get indices of template frames
        ref_idx = range(l_seg_bounds[ref_seg_idx][0], l_seg_bounds[ref_seg_idx][1])      
        # get indices of target frames  
        target_idx = range(cur_target_idx, min(cur_target_idx + target_seg_length/tstep,  morphed_spectro.shape[0]))
        
        # now we need to compute the ratio of morphing        
        ratio = float(len(ref_idx))/float(len(target_idx))        
        # then 
        if ratio < 1:
            # case where we need to elongate the signal: use resize: duplicate end elements
            # TODO : maybe some other interpolation scheme would be better
            morphed_spectro[target_idx,:] = np.resize(np.abs(learn_specs[ref_idx,:]), (len(target_idx), learn_specs.shape[1]))
        else:
            # case where we need to compress the signal: build a redundant comb 
            morphed_spectro[target_idx,:] = np.abs(learn_specs[[ref_idx[int(j*ratio)] for j in range(len(target_idx))],:])             
        cur_target_idx += target_seg_length/tstep    
    return morphed_spectro
    
def time_stretch(signalin, tscale, wsize=512, tstep=128):
    """ take the time serie, perform a highly overlapped STFT
        and change the hop size at reconstruction while adapting 
        the phase accordingly """
    
    # read input and get the timescale factor    
    L = signalin.shape[0]    
    
    # signal blocks for processing and output
    phi  = np.zeros(wsize)
    out = np.zeros(wsize, dtype=complex)
    sigout = np.zeros(L/tscale+wsize)
    
    # max input amp, window
    amp = signalin.max()
    win = np.hanning(wsize)
    p = 0  # position in the original (increment by tstep*tscale)
    pp = 0 # position in the target   (increment by tstep)
    
    # TODO change this: the algorithm is stopping too soon: many zeroes on the edges
    while p < L-(wsize+tstep):    
        # take the spectra of two consecutive windows
        p1 = int(p)
        spec1 =  np.fft.fft(win*signalin[p1:p1+wsize])
        spec2 =  np.fft.fft(win*signalin[p1+tstep:p1+wsize+tstep])
        # take their phase difference and integrate
        phi += (np.angle(spec2) - np.angle(spec1))
        # bring the phase back to between pi and -pi
        for phiI in range(len(phi)):
            while phi[phiI] < -pi: phi[phiI] += 2*pi
            while phi[phiI] >= pi: phi[phiI] -= 2*pi
        out.real, out.imag = np.cos(phi), np.sin(phi)
        # inverse FFT and overlap-add
        sigout[pp:pp+wsize] += win*np.fft.ifft(np.abs(spec2)*out)
        pp += tstep
        p += tstep*tscale
    
    return sigout

def get_audio(filepath, seg_start, seg_duration):
    """ for use only with wav files from rwc database """
    
    if filepath[-3:] == 'wav' or filepath[-3:] == 'WAV':
        import wave
        wavfile = wave.open(filepath, 'r')
    elif filepath[-2:] == 'au':
        import sunau
        wavfile = sunau.open(filepath, 'r')
    fs = wavfile.getframerate()
    bFrame = int(seg_start*fs)
    nFrames = int(seg_duration*fs)    
    wavfile.setpos(bFrame)
#        print "Reading ",bFrame, nFrames, wavfile._framesize
    str_bytestream = wavfile.readframes(nFrames)
    
    sample_width = wavfile.getsampwidth()
    print filepath, sample_width, wavfile.getnchannels() , fs, nFrames
    if sample_width == 1:
        typeStr = 'int8'
    elif sample_width == 2:
        typeStr = 'int16'
    elif sample_width == 3:
        typeStr ='int24' # WARNING NOT SUPPORTED BY NUMPY
    elif sample_width == 4:
        typeStr = 'uint32'    
    
    audiodata = np.fromstring(str_bytestream, dtype=typeStr)
    wavfile.close()
    return audiodata, fs

