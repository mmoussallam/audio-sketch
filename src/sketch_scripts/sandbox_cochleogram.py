'''
sketch_scripts.sandbox_cochleogram  -  Created on Jul 31, 2013
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

from classes.sketches.cochleo import *
from classes.pydb import CochleoPeaksBDB
fs = 8000
sk = CochleoPeaksSketch(**{'fs':fs,'step':512,'downsample':fs}) 

target_file = '/sons/sqam/voicemale.wav'

sk.recompute(target_file)

fgpthandle = CochleoPeaksBDB(None,
                             load=False,
                             persistent=False, **{'wall':False})

sk.sparsify(5)

fgpthandle._build_pairs(sk.fgpt(),sk.params, display=True)

plt.show()