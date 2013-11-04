# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:53:00 2013

@author: loa-guest
"""

import scipy.signal as scps
import cmath
import math
import numpy as np
from numpy import floor
import matplotlib.pyplot as plt
from PyMP import Signal
import scipy.io as sio
import scipy.sparse as sp
import sys
import profile


#sys.path.append('/Users/loa-guest/Documents/Laure/audio-sketch-master/src/')
#son = '/Users/loa-guest/Documents/MATLAB/test_small_mono.wav'
#signal = Signal(son, normalize=True, mono=True)
     
     

def noyau(fs,freq_min,freq_max,fen_type,bins): #x = resize(signal) 
    
    K = math.ceil(bins*np.log2(freq_max/freq_min))
    if fen_type==1:
       seuil = 0.0075 # hamming  
    elif fen_type==2:
       seuil = 0.005 # blackman 
    elif fen_type==3:
       seuil  = -1.0   
    Q=1/(2**(1/bins)-1)
    Nfft=2**nextpow2(math.ceil(Q*fs/freq_min))
    fftvec = np.arange(Nfft)
    #B = [cmath.exp(2*np.pi*1j*Q*a) for a in range(int(Nfft))]
    noyautmp = np.zeros((Nfft,K), dtype = complex)
    for k in np.arange(K-1,-1,-1):
        N=math.ceil(Q*fs/(freq_min*2**(k/bins)))
        if N%2 ==1:
            N=N-1
        n0=Nfft/2-N/2 
        listn0 = [0]*n0
        listn1 = range(int(N))
        if fen_type ==1:
            fen = np.hamming(N+1)       
        elif fen_type ==2:
            fen = np.blackman(N+1)
        elif fen_type ==3:
            fen = tukeywin(N,0.5)  #used for the inverse
        norm_fact = sum(fen[0:N])
        listn1 = [fen[a]/norm_fact for a in listn1]
        listn1.extend(listn0)
        listn0.extend(listn1)
        #B = [cmath.exp(2*np.pi*1j*Q*a/N) for a in range(int(Nfft))]
        B = np.exp(2*np.pi*1j*Q*fftvec/N)
        #B2 = np.array(B)**(1./float(N))
        noyautmp[:,k] = np.multiply(listn0,B)
    specnoyau= np.fft.fft(noyautmp.T)
    if seuil>0:
        specnoyau[abs(specnoyau)<=seuil] = 0
        noyauatrous=np.conj(sp.csc_matrix(specnoyau.T))
    else:
        noyauatrous=np.conj(specnoyau.T)
    return noyauatrous,K
    
    
def cqtS(signal, noyauatrous, K, freq_min, bins,overl):
#    signal.data = np.array(signal.data).flatten(1)
    signal.data = signal.data.flatten(1)
    wdw_size = noyauatrous.shape[0]    
    cqt_l = int((signal.length-wdw_size)/overl)+1
    t_cal = np.zeros(cqt_l)
    cqt = np.zeros((K,cqt_l),dtype=complex)
    l=0
    for indice in np.arange(cqt_l):
        bit = signal.data[int(indice*overl):int(indice*overl+wdw_size)]     
        cq = constQ(bit.T, noyauatrous)
        cqt[:,indice] = cq
        t_cal[l] = float((indice*overl+1)/signal.fs)
        l += 1  
    f_cal=freq_min*2**(np.arange(int(K))/bins)
    return cqt,f_cal,t_cal
    
def cqt_d(signal_data,noyauatrous,K,freq_min,bins,overl):
    signal_data = signal_data.flatten(1)
    wdw_size = noyauatrous.shape[0]    
    cqt_l = int((signal_data.shape[0]-wdw_size)/overl)+1
    cqt = np.zeros((K,cqt_l),dtype=complex)
    l=0
    for indice in np.arange(cqt_l):
        bit = signal_data[int(indice*overl):int(indice*overl+wdw_size)]     
        cq = constQ(bit.T, noyauatrous)
        cqt[:,indice] = cq
        l += 1  
    return cqt
    
def inverseS(cqt,fs,freq_min,freq_max,bins,overl):
    nblocks = cqt.shape[1]
    [noyau_inv,K] = noyau(fs,freq_min,freq_max,3,bins)
    Nfft = noyau_inv.shape[0]
    print noyau_inv.shape, cqt.shape
    yf = overl * np.dot(noyau_inv, cqt)
    y_oct = 2*np.real(np.fft.ifft(yf.T))
    lensig = (nblocks-1)*overl + Nfft
    x_oct = np.zeros(lensig)
    for n in range(int(nblocks)):
        x_oct[int(n*overl):int(n*overl+Nfft)] += y_oct[n,:]
    return x_oct
    
    
#def cqtP(noyau,fs_new,avance,Nfft,K,signal,freq_max,bandwidth, freq_min,bins,overl):
#    
#    nbo = np.log2(freq_max/freq_min)
#    b= importB()
#    nk = math.ceil(np.log2(bandwidth*signal.fs)-np.log2(freq_max)) -1
#    # prétraitement à faire si nk!=O
#    for k in np.arange(nk):
#        x = scps.filtfilt(b, np.array([1]),signal.x)
#        x = scps.decimate(x,2)
#
#    x_oct = [0]*1
#    x_oct[0] = x 
#    for k in range(K-1):
#        x_oct.append(downsample(scps.filtfilt(b, np.array([1]),x_oct[k]),2))
#    x_oct.reverse()
#    
#
#    val=[2**(nbo-1)]*1
#    [val.extend([val[a]/2]) for a in range(nbo-1)]
#    A = range(nbo)
#    x_oct2 = [bufferND(x_oct[a],Nfft,Nfft-overl/c) for (a,c) in zip(A,val)]
#    
#    cqt_non_sync = np.array(np.zeros((nbo*bins,x_oct2[nbo-1].shape[1])),dtype=complex)
#    for nb_sub in range(nbo):
#        for n_buf in range(x_oct2[nb_sub].shape[1]):
#            index = np.arange(int(nb_sub*bins),int((nb_sub+1)*bins))
#            cqt_non_sync[index,n_buf] = constQ(x_oct2[nb_sub][:,n_buf],noyau)
#        
##Resynchronise les coefficients cqt aux instants de calcul par rapport au signal
## fonction resync
#    nl = cqt_non_sync.shape[0]
#    nc = cqt_non_sync.shape[1]
#    decmin = Nfft/(2*overl)
#    nc_ns=nc+Nfft*(2**(nbo-2)/overl)     
#            
#    cqt_sync=np.zeros((nl,nc_ns),dtype = complex)
#    for k in range(nbo-4):
#        ind = int(k*bins)
#        dec = int(Nfft*(2**(nbo-2-k)/overl))
#        if int(k)==1 and decmin == 0.5:
#            dec = 0
#        cqt_sync[ind:int(ind+bins),:][:,dec:int(dec+nc)] = cqt_non_sync[np.arange(ind,int(ind+bins)),:]
#    t_cal = Nfft/2-decmin+overl*np.arange(nc_ns-1)/fs_new
#    f_cal = [freq_min*2**a/bins for a in range(int(K))]
#    return cqt_sync, f_cal, t_cal
#    

    
def nextpow2(i):
    n = 2
    while n < i: n = n * 2
    return np.log2(n)
    
def importMat(nameOfvar):
    adresse = '/Users/loa-guest/Documents/MATLAB/'+ nameOfvar + '.mat'
    fullVar = sio.loadmat(adresse)
    var = fullVar[nameOfvar]
    del fullVar
    return var

def importB():  ####### dois je mettre self?
    fullB = sio.loadmat('b.mat')
    b = fullB['b']
    del fullB
    b = b.reshape(b.shape[1],)
    return b
    
def constQ(x,noyauatrous):
    cq = np.fft.fft(x,noyauatrous.shape[0])*noyauatrous/noyauatrous.shape[0]
   #np.dot(np.fft.fft(x,noyauatrous.shape[0]),noyauatrous)/noyauatrous.shape[0]
    return cq
    
def downsample(signal,pas):
    # signal de taille (L,)
    L = signal.shape[0]
    A = range(0,L,pas)
    return signal[A]
    
def bufferND(x, n, p):
    modulo = (n-p) - (x.shape[0]-p)%(n-p)
    x = np.concatenate((x,np.zeros(modulo)))
    q = floor((x.shape[0]-p)/(n-p)+0.99)
    x1 = np.zeros((n,q))
    x1[:,0] = x[0:n]
    for k in np.arange(1,int(q)): 
        x1[:,k] = x[k*(n-p):n+k*(n-p)]
    return x1

def resize(signal):
   # signal est de la classe signal
   nb = signal.length//signal.fs           ####### est ce sur que le signal est en mono (length)
   signal.x = np.concatenate((np.zeros(floor(1.2*signal.fs),),signal.data[0:floor(2*signal.fs)], np.zeros(floor(1.2*signal.fs),)), axis = 0)
   del nb
   return signal
    
def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w

