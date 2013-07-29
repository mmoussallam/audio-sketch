'''
fgpt_scripts.visu_peak_pairs  -  Created on Jul 29, 2013
@author: M. Moussallam
'''
import os
import sys
import numpy as np
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

#from classes import pydb, sketch
from classes.pydb import *
from classes.sketches.cortico import CorticoSubPeaksSketch
from classes.sketches.cochleo import CochleoPeaksSketch
from PyMP.signals import LongSignal, Signal
import os.path as op
import matplotlib.pyplot as plt

single_test_file1 = '/sons/sqam/voicemale.wav'

fgpthand = CochleoPeaksBDB('CorticoSub_0_0Peaks.db', **{'wall':False})
sk1 = CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})
sk = CorticoSubPeaksSketch(**{'fs':8000,'step':128,'downsample':8000,'sub_slice':(4,11)})

sk1.recompute(single_test_file1)
sk.recompute(single_test_file1)
sk1.sparsify(300)    
sk.sparsify(300)    
test_fgpt_cochleo = sk1.fgpt(sparse=True)
test_fgpt_cortico = sk.fgpt(sparse=True)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.spy(test_fgpt)
#ax.arrow(0, 50, 5, 5, head_width=0.05, head_length=0.1, fc='k', ec='k')
#plt.show()
#import matplotlib.cm as cm
#plt.figure()
#plt.subplot(221)
#plt.imshow(np.abs(sk1.rep), cmap=cm.bone_r)
#plt.subplot(223)
#plt.imshow(np.abs(sk.rep[4,11,:,:].T), cmap=cm.bone_r)
#plt.subplot(222)
#plt.imshow(test_fgpt_cochleo, cmap=cm.bone_r)
#plt.subplot(224)
#plt.imshow(test_fgpt_cortico, cmap=cm.bone_r)
#plt.show()


#fgpthand.params['min_bin_dist']=0
#fgpthand.params['min_fr_dist']=0
#fgpthand._build_pairs(test_fgpt_cochleo, sk.params, display=True)
#fgpthand._build_pairs(test_fgpt_cortico, sk.params, display=True)
#plt.show()
