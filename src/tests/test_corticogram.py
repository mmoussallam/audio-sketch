'''
tests.test_corticogram  -  Created on Jun 27, 2013
@author: M. Moussallam
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
import sys
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
from src.tools import cochleo_tools
#from classes import sketch
import matplotlib.pyplot as plt
from PyMP import Signal
from scipy.signal import lfilter, hann
from scipy.io import loadmat
#from scipy.fftpack import fft, ifft
from numpy.fft import fft, ifft

plt.switch_backend('Agg')
audio_test_file = '/home/manu/workspace/recup_angelique/Sketches/NLS Toolbox/nsltools/_done.au'
audio_test_file = '/sons/jingles/panzani.wav'

############################### Inversion
sig = Signal(audio_test_file, mono=True, normalize=True)
sig.downsample(8000)
# convert to auditory
params = {'frmlen':8,'shift':0,'fac':-2,'BP':1}

gram = cochleo_tools.Cochleogram(sig.data, **params )

import cProfile
cProfile.runctx('gram.build_aud()',globals(), locals())
cProfile.runctx('gram.build_aud_old()',globals(), locals())


aud = gram.build_aud()
# Cortico-gram : 2D complex transform of y5
# we need to define y = gram.y5, para1= vector pf parameters, rv = rate vector, sv = scale vector
y = np.array(gram.y5)

#rec_from_aud = Signal(gram.invert(nb_iter=50, init_vec=sig.data), sig.fs, normalize=True)

cort = cochleo_tools.Corticogram(gram, **params)
cort.build_cor()

cortaud = np.array(cort.coch.y5)


from scipy.io import loadmat
D = loadmat('/home/manu/workspace/test_cortico_done.mat')
t_cor = D['cor']
t_aud = D['aud']


## checking the cochlear filtering
#data = sig.data
#coeffs = gram.coeffs


#np.linalg.norm(t_aud - aud, 'fro')
#np.linalg.norm(t_aud, 'fro')

#plt.figure()
#plt.plot(t_aud[:,0])
#plt.plot(aud[:,0],'r')
##plt.plot(cortaud[:,30],'k')
#plt.show()
#
#
#plt.figure()
#plt.subplot(121)
#plt.imshow(t_aud)
#plt.subplot(122)
#plt.imshow(aud)
#plt.show()

#from scipy import interpolate
#rs_view = np.abs(np.squeeze(np.mean(np.mean(cort.cor,axis=3),axis=2)))
#(K2,K1) = rs_view.shape
#
#x, y = np.meshgrid(cort.params['rv'], cort.params['sv'])
#xnew,ynew = np.meshgrid(range(cort.params['rv'][-1]),range(cort.params['sv'][-1]))
#tck = interpolate.bisplrep(np.log2(x),np.log2(y),rs_view[:,:K1/2],s=10)
#rs_view_interp = interpolate.bisplev(xnew[:,0],ynew[0,:],tck)


        
#plt.figure()
##plt.subplot(121)
##plt.pcolor(np.fliplr(-X), Y, rs_view[:,:K1/2].T,                   
##           cmap=cm.bone_r)
##plt.subplot(122)
#plt.pcolor(xnew,ynew,rs_view_interp,
#           cmap=cm.bone_r)
#plt.show()

rec_aud = cort.invert()

plt.figure()
plt.subplot(121)
plt.imshow(np.log(np.abs(rec_aud.T)), origin='lower')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log(np.abs(y)), origin='lower')
plt.colorbar()
#plt.show()
#rec_aud *= np.max(y)/np.max(rec_aud)
#print np.linalg.norm(np.abs(y) - np.abs(rec_aud),'fro')
#np.linalg.norm(np.abs(y))
#
#rec_sig = Signal(gram.invert(v5 = np.abs(rec_aud.T), init_vec=sig.data, nb_iter=20, display=False), sig.fs, normalize=True)
##rec_sig_rnd = Signal(gram.invert(v5 = np.abs(rec_aud.T), nb_iter=10), sig.fs, normalize=True)
## NOT really satisfying but ... ok
#rec_sig.play()

L = 1000
A_flat = cort.cor
A_flat =  A_flat.flatten()
idx_order = np.abs(A_flat).argsort()
A = np.zeros(A_flat.shape, complex)
A[idx_order[-L:]] = A_flat[idx_order[-L:]]

A = A.reshape(cort.cor.shape)

sp_rec_aud = cochleo_tools._cor2aud(A, **cort.params)


sp_rec_sig = Signal(gram.invert(v5 = np.abs(sp_rec_aud.T), init_vec=sig.data, nb_iter=20, display=False), sig.fs, normalize=True)

plt.figure()
plt.subplot(121)
plt.imshow(np.log(np.abs(rec_aud.T)), origin='lower')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log(np.abs(sp_rec_aud.T)), origin='lower')
plt.colorbar()
plt.show()

# reconstruct

