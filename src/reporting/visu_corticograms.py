'''
reporting.visu_corticograms  -  Created on Aug 22, 2013
@author: M. Moussallam
'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyMP import Signal
import cProfile
import bsddb.db as db
sys.path.append(os.path.abspath('../../'))
#sys.path.append('/home/manu/workspace/PyMP')
#sys.path.append('/home/manu/workspace/meeg_denoise')

from classes.sketches.cortico import * 
from tools.cochleo_tools import _cor2aud, Corticogram
from classes import pydb

audio_test_file = os.path.abspath('./audio/original_surprise.wav')
fs = 8000

sig = Signal(audio_test_file, mono=True, normalize=True)
#sig.crop(0, 2*sig.fs)
sk = CorticoIndepSubPeaksSketch(**{'fs':fs,'downsample':fs,'frmlen':8,
                                   'shift':0,'fac':-2,'BP':1})
sk.recompute(sig)

# saving the resynthesized sounds


#sk.sparsify(10)
#sk.represent()


#env = db.DBEnv()
#env.open("/home/manu/workspace/audio-sketch/src/reporting", db.DB_INIT_MPOOL|db.DB_CREATE )
#fgpthandle = pydb.CorticoIndepSubPeaksBDB("temp.db",dbenv=None,                                         
#                                               **{'wall':True,'max_pairs':500})
#
#
##fgpthandle.dbObj[n][m]._build_pairs(sk.fgpt()[0,M/2+0,:,:], sk.params, display=True, ax=None)
#
#(N,M) = sk.cort.cor.shape[:2]
#plt.figure()
#for n in range(N):
#    for m in range(M/2):
#        ax = plt.subplot(N,M/2, n*(M/2) + m +1)
#        fgpthandle.dbObj[n][m]._build_pairs(sk.fgpt()[n,M/2+m,:,:], sk.params, display=True, ax=ax)
#        plt.xticks([])
#        plt.yticks([])
#    
#plt.show()