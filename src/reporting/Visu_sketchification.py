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
from classes import sketch
figure_output_path = '/home/manu/workspace/audio-sketch/src/reporting/figures/'
audio_output_path = '/home/manu/workspace/audio-sketch/src/reporting/audio/'

learned_base_dir = '/home/manu/workspace/audio-sketch/matlab/'    

sketches_to_test = [sketch.SWSSketch(),
                    sketch.KNNSketch(**{'location':learned_base_dir,
                                                'shuffle':13,
                                                'n_frames':100000,
                                                'n_neighbs':1}),
                    sketch.STFTPeaksSketch(**{'scale':2048, 'step':256}),
                    sketch.CochleoPeaksSketch(),
                    sketch.XMDCTSparseSketch(**{'scales':[64,512,4096], 'n_atoms':100}),
                    sketch.STFTDumbPeaksSketch(**{'scale':2048, 'step':256}),
                            ]

for sk in sketches_to_test:
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
    plt.savefig(op.join(figure_output_path, 'sketchified_%s.pdf'%sk.__class__.__name__))    

    synth_sig.write(op.join(audio_output_path, 'sketchified_%s.wav'%sk.__class__.__name__))

plt.show()