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

from classes import sketch 
from tools.cochleo_tools import _cor2aud

audio_test_file = '/sons/jingles/panzani.wav'

sk = sketch.CorticoPeaksSketch(**{'downsample':8000})
skiht = sketch.CorticoIHTSketch(**{'downsample':8000})

sk.recompute(audio_test_file)
#skiht.recompute(audio_test_file)

#cProfile.runctx('sk.sparsify(1000)', globals(), locals())
sk.sparsify(1000)
#skiht.sparsify(1000)

#ihtnnz = np.nonzero(skiht.sp_rep)
np.count_nonzero(sk.sp_rep)


# with the original in mind
#synth_sig = sk.synthesize(sparse=True)
#synth_sig.normalize()
#
rec_inv = np.abs(sk.cort.invert().T)

rec_auditory = np.abs(_cor2aud(sk.rep, **sk.params))
rec_auditory_sp = np.abs(_cor2aud(sk.sp_rep, **sk.params))


plt.figure()
plt.plot(np.real(sk.sp_rep.flatten()[np.flatnonzero(sk.sp_rep)]))
plt.plot(np.real(sk.rep.flatten()[np.flatnonzero(sk.sp_rep)]), 'r')
plt.show()

plt.figure()
plt.subplot(211)
plt.imshow(np.abs(rec_auditory.T))
plt.subplot(212)
plt.imshow(np.abs(rec_auditory_sp.T))
plt.show()

sp_vec = sk.sp_rep

sk.represent( sparse = True)

# binary vector
bin_nnz = np.flatnonzero(sk.sp_rep)

plt.figure()
plt.stem(bin_nnz,[1]*len(bin_nnz))
plt.show()

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
sk.orig_signal = None


#plt.figure()
#plt.imshow(np.abs(sk.rec_aud.T))
#plt.show()

sk.params['n_inv_iter'] = 10
synth_sig_w = sk.synthesize(sparse=False)

synth_sig_w.normalize()

#synth_sig_iht = skiht.synthesize(sparse=True)


#cProfile.runctx('sk.recompute(audio_test_file)', globals(), locals())
