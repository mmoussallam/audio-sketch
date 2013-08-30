'''
reporting.Visu_sketchification  -  Created on Feb 7, 2013
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
audio_test_file = '/sons/jingles/panzani.wav'
from classes.sketches.bench import *
from classes.sketches.misc import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
figure_output_path = '/home/manu/workspace/audio-sketch/src/reporting/figures/'
audio_output_path = '/home/manu/workspace/audio-sketch/src/reporting/audio/'

learned_base_dir = '/home/manu/workspace/audio-sketch/matlab/'    

sketches_to_test = [SWSSketch(),
                    KNNSketch(**{'location':learned_base_dir,
                                                'shuffle':13,
                                                'n_frames':100000,
                                                'n_neighbs':1}),
                    STFTPeaksSketch(**{'scale':2048, 'step':256}),
                    CochleoPeaksSketch(),
                    CorticoIndepSubPeaksSketch(),
                    XMDCTSparseSketch(**{'scales':[64,512,4096], 'n_atoms':100}),                    
                            ]

for sk in sketches_to_test:
    audio_name = op.split(audio_test_file)[1][:-4]
    # also save the complete STFT
    sig = Signal(audio_test_file, normalize=True, mono=True)
    sig.resample(11025)
    plt.figure(figsize=(10,5))    
    sig.spectrogram(1024,64, order=0.5, log=False,
                          cmap=cm.hot, cbar=False)
        
    plt.savefig(op.join(figure_output_path, 'original_%s.pdf'%(audio_name)))    
    sig.write(op.join(audio_output_path, 'original_%s.wav'%(audio_name)))
    
    print " compute full representation"
    sk.recompute(audio_test_file)
    
#    print " plot the computed full representation" 
#    sk.represent()
    
    print " Now sparsify with 1000 elements" 
    sk.sparsify(1000)
    
#    print " plot the sparsified representation"    
#    sk.represent(sparse=True)
        
    synth_sig = sk.synthesize(sparse=True)              
    
#    print synth_sig.data.shape
    
    snr = 10*np.log10(synth_sig.energy/np.sum((synth_sig.data- sk.orig_signal.data[:synth_sig.length])**2))
    
    synth_sig.resample(11025)
    plt.figure(figsize=(10,5))    
    synth_sig.spectrogram(1024,64, order=0.5, log=False,
                          cmap=cm.hot, cbar=False)
    
    plt.title('SNR of %2.2f dB'%snr)
    plt.savefig(op.join(figure_output_path, 'sketchified_%s_%s.pdf'%(audio_name,
                                                                     sk.__class__.__name__)))    

    synth_sig.write(op.join(audio_output_path, 'sketchified_%s_%s.wav'%(audio_name,
                                                                        sk.__class__.__name__)))

plt.show()