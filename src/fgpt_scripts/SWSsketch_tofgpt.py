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

sk.sparsify(30)   
sk.represent(sparse=True)
synth30 = sk.synthesize(sparse=True)

#sk.params['n_formants']=5
#sk.sparsify(30000)   
#sk.represent(sparse=True)
#synth30000 = sk.synthesize(sparse=True)

plt.show()
#x_vec = np.arange(0.0,sk.orig_signal.get_duration(),sk.orig_signal.get_duration()/float(A.shape[0]))
#y_vec = sk.formants[1]
# 
#deg = 30
#p = np.polyfit(x_vec, y_vec, deg)
#rec = np.zeros_like(y_vec)
#for i in range(deg+1):
#    rec += p[-i-1]*(x_vec ** i)
#
#plt.figure()
#plt.plot(x_vec, y_vec)
#plt.plot(x_vec, rec)
#plt.show()
