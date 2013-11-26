'''
sketch_scripts.sandbox_corticograms  -  Created on Jul 18, 2013
@author: M. Moussallam
'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PyMP import Signal
import cProfile
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
from src.settingup import *
from tools.cochleo_tools import _cor2aud, _build_cor

audio_test_file = '/sons/sqam/voicemale.wav'

skaud = CochleoSketch(**{'downsample':8000})
sk = CorticoPeaksSketch(**{'downsample':8000})

sk.recompute(audio_test_file)
skaud.recompute(audio_test_file)
cort = sk.cort
rec = sk.cort.invert(sk.cort.cor, 1)

#plt.figure()
#plt.subplot(121)
#skaud.represent(fig=plt.gcf())
#plt.subplot(122)
#plt.imshow(np.abs(rec.T),
#           origin='lower',cmap=cm.bone_r)
#plt.show()

##### Stability of the Corticogram inversion ####
# can we invert a single point in the corticogram ?
sp_cort = np.zeros_like(sk.cort.cor)


cor = sk.cort.cor
plt.figure()
for i in range(cor.shape[0]):
    for j in range(cor.shape[1]):                
        plt.subplot( cor.shape[0], cor.shape[1], (i* cor.shape[1]) + j+1)
        plt.imshow(np.abs(cor[i,j,:,:]).T, origin='lower',cmap=cm.bone_r)        
        plt.xticks([])
        plt.yticks([])        
plt.show()        


plt.figure()
for i in range(cor.shape[0]):
    for j in range(cor.shape[1]):                
        plt.subplot( cor.shape[0], cor.shape[1], (i* cor.shape[1]) + j+1)
        vals, bins = np.histogram(np.abs(cor[i,j,:,:].flatten()), 1000)
        plt.plot(np.log(vals))
plt.show()


# inversion
sparsified1 = np.zeros_like(cor)
sparsified2 = np.zeros_like(cor)

sidx = 2
ridx = 1
sparsified1[sidx,ridx,:,:] = cor[sidx,ridx,:,:]
sparsified2[sidx,ridx,:,:] = cor[sidx,ridx,:,:]
sparsified2[sidx,ridx+6,:,:] = cor[sidx,ridx+6,:,:]
aud1 = _cor2aud(sparsified1,**sk.params)
aud2 = _cor2aud(sparsified2,**sk.params)

plt.figure()
plt.subplot(121)
plt.imshow(np.abs(aud1))
plt.subplot(122)
plt.imshow(np.abs(aud2))
#plt.show()

sk.sp_rep = sparsified1
rec1 = sk.synthesize(sparse=True)
rec1.normalize()
sk.sp_rep = sparsified2
rec2 = sk.synthesize(sparse=True)
rec2.normalize()

plt.show()


################ what does a scale/fix point mean 

# let us put a 1 steadily across scales at one TF position and see how it translates
tidx = art_cor.shape[2]/4
fidx = art_cor.shape[3]/4


plt.figure()
for i in range(cor.shape[0]):
    for j in range(cor.shape[1]/2):  
        art_cor1 = np.zeros_like(cor)
        art_cor1[:i+1,:j+1,tidx,fidx] = 1            
        plt.subplot( cor.shape[0], cor.shape[1]/2, (i* cor.shape[1]/2) + j+1)
        audart1 = _cor2aud(art_cor1,**sk.params)   
        plt.imshow(np.abs(audart1).T)
plt.show()

        
audart2 = _cor2aud(art_cor2,**sk.params)   
plt.figure()     
plt.subplot(311)
plt.imshow(np.real(audart1).T)
plt.subplot(312)
plt.imshow(np.real(audart2).T)
plt.subplot(313)
plt.imshow(np.real(audart2+audart1).T)
plt.show()

########################
plt.figure()
plt.imshow(np.abs(sk.cort.cor[:,:,tidx, fidx]))
plt.show()

#for i in range(cor.shape[0]):
#    for j in range(cor.shape[1]/2): 
#        art_cor[i,j,tidx, fidx] = 1
#        
art_cor1 = np.zeros_like(cor)
art_cor2 = np.zeros_like(cor)
art_cor1[0,6,tidx,fidx] = 1
art_cor2[0,0,tidx,fidx] = 2

audart1 = _cor2aud(art_cor1,**sk.params)   
audart2 = _cor2aud(art_cor2,**sk.params)   
plt.figure()     
plt.subplot(311)
plt.imshow(np.real(audart1).T)
plt.subplot(312)
plt.imshow(np.real(audart2).T)
plt.subplot(313)
plt.imshow(np.real(audart2+audart1).T)
plt.show()



#sk.recompute(audio_test_file)
##skiht.recompute(audio_test_file)
#
##cProfile.runctx('sk.sparsify(1000)', globals(), locals())
#sk.sparsify(1000)
##skiht.sparsify(1000)
#
##ihtnnz = np.nonzero(skiht.sp_rep)
#np.count_nonzero(sk.sp_rep)
#
#
## with the original in mind
##synth_sig = sk.synthesize(sparse=True)
##synth_sig.normalize()
##
#rec_inv = np.abs(sk.cort.invert().T)
#
#rec_auditory = np.abs(_cor2aud(sk.rep, **sk.params))
#rec_auditory_sp = np.abs(_cor2aud(sk.sp_rep, **sk.params))
#
#
#plt.figure()
#plt.plot(np.real(sk.sp_rep.flatten()[np.flatnonzero(sk.sp_rep)]))
#plt.plot(np.real(sk.rep.flatten()[np.flatnonzero(sk.sp_rep)]), 'r')
#plt.show()
#
#plt.figure()
#plt.subplot(211)
#plt.imshow(np.abs(rec_auditory.T))
#plt.subplot(212)
#plt.imshow(np.abs(rec_auditory_sp.T))
#plt.show()
#
#sp_vec = sk.sp_rep
#
#sk.represent( sparse = True)
#
## binary vector
#bin_nnz = np.flatnonzero(sk.sp_rep)
#
#plt.figure()
#plt.stem(bin_nnz,[1]*len(bin_nnz))
#plt.show()

## Ok so let us load a previously computed cortiocogram
#save_path = '/media/manu/TOURO/corticos/rwc-g-m01_1.wav_seg_0.npy'
#loaded_cort = np.load(save_path)
#sk.rep = loaded_cort

#sig = Signal(sk.coch.invert(rec_auditory, sk.orig_signal.data, 
#                              nb_iter=sk.params['n_inv_iter'], display=True),
#            sk.orig_signal.fs, normalize=True)
#
#init_vec = sk.coch.init_inverse(rec_auditory)
#init_sig = Signal(init_vec, 8000, normalize=True)

#
#sig_init = Signal(sk.coch.invert(rec_auditory, init_vec, 
#                              nb_iter=10, display=True),
#            sk.orig_signal.fs, normalize=True)
#
#
#sig_rand = Signal(sk.coch.invert(rec_auditory, np.random.randn(len(init_vec)), 
#                              nb_iter=2, display=True),
#            sk.orig_signal.fs, normalize=True)
# without the original in mind
#sk.orig_signal = None
#
#
##plt.figure()
##plt.imshow(np.abs(sk.rec_aud.T))
##plt.show()
#
#sk.params['n_inv_iter'] = 10
#synth_sig_w = sk.synthesize(sparse=False)
#
#synth_sig_w.normalize()

#synth_sig_iht = skiht.synthesize(sparse=True)


#cProfile.runctx('sk.recompute(audio_test_file)', globals(), locals())
