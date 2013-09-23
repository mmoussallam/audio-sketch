'''
reporting.visu_fingerprints  -  Created on Aug 30, 2013
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
import sys
import os.path as op
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
from tools import cochleo_tools
#from classes import sketch
import matplotlib.pyplot as plt
from PyMP import Signal
import stft
from scipy.signal import lfilter, hann
#audio_test_file = '/home/manu/workspace/recup_angelique/Sketches/NLS Toolbox/Hand-made Toolbox/forAngelique/61_sadness.wav'
audio_test_file = '/sons/jingles/carglass.wav'
audio_name ='carglass'
from classes.sketches.bench import *
from classes.sketches.misc import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes.pydb import *
figure_output_path = '/home/manu/workspace/audio-sketch/src/reporting/figures/'
audio_output_path = '/home/manu/workspace/audio-sketch/src/reporting/audio/'
single_test_file = '/sons/jingles/carglass.wav'
learned_base_dir = '/home/manu/workspace/audio-sketch/matlab/'    

fgpt_sketches = [
#                 (SWSBDB(None, **{'wall':False,'n_deltas':2}),                  
#                 SWSSketch(**{'n_formants_max':7,'time_step':0.02})), 
                (STFTPeaksBDB(None, **{'wall':False}),
                 STFTPeaksSketch(**{'scale':2048, 'step':512})), 
                (CochleoPeaksBDB(None, **{'wall':False}),
                 CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),
                 (XMDCTBDB(None, load=False,**{'wall':False}),
                  XMDCTSparseSketch(**{'scales':[ 4096],'n_atoms':150,
                                              'nature':'LOMDCT'})),         
#                 (CochleoPeaksBDB(None, **{'wall':False}),
#                  CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),                  
                    (CorticoIndepSubPeaksBDB('tempcort.db', **{'wall':False}),
                     CorticoIndepSubPeaksSketch(**{'fs':8000,'frmlen':8,'downsample':8000}))                                             
                    ]

for fgpthand, sk in fgpt_sketches:
    sk.recompute(single_test_file)
    sk.sparsify(20)
    # convert it to a fingeprint compatible with associated handler
    fgpt = sk.fgpt(sparse=True)
    params = sk.params
#            print fgpt
    # check that the handler is able to process the fingerprint            
    print "Here the params: ",sk.params
    fgpthand.draw_fgpt(fgpt, sk.params)
    plt.savefig(op.join(figure_output_path, 'fingerprint_%s_%s.pdf'%(audio_name,
                                                                     sk.__class__.__name__)))
plt.show()    

