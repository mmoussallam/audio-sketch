'''
sketch_scripts.sandbox_corticograms  -  Created on Jul 18, 2013
@author: M. Moussallam
'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cProfile
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

from classes import sketch 


audio_test_file = '/sons/jingles/panzani.wav'

sk = sketch.CorticoPeaksSketch(**{'downsample':8000})
skiht = sketch.CorticoIHTSketch(**{'downsample':8000})

sk.recompute(audio_test_file)
#skiht.recompute(audio_test_file)

#cProfile.runctx('sk.sparsify(1000)', globals(), locals())
sk.sparsify(10000)
#skiht.sparsify(1000)

#ihtnnz = np.nonzero(skiht.sp_rep)
np.count_nonzero(sk.sp_rep)


# with the original in mind
synth_sig = sk.synthesize(sparse=True)
synth_sig.normalize()

# without the original in mind
sk.orig_signal = None
sk.params['n_inv_iter'] = 10
synth_sig_w = sk.synthesize(sparse=False)
synth_sig_w.normalize()

#synth_sig_iht = skiht.synthesize(sparse=True)


#cProfile.runctx('sk.recompute(audio_test_file)', globals(), locals())
