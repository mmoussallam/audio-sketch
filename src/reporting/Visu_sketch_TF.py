'''
reporting.Visu_sketch_TF  -  Created on Apr 22, 2013
@author: M. Moussallam
'''
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
audio_test_file = '/sons/sqam/glocs.wav'
from classes import sketch
figure_output_path = '/home/manu/workspace/audio-sketch/src/reporting/figures/'
audio_output_path = '/home/manu/workspace/audio-sketch/src/reporting/audio/'


########## plot spectrogram for glockenspiel ###########
synth_sig = Signal(audio_test_file,normalize=True, mono=True)
synth_sig.crop(0.1*synth_sig.fs, 3.5*synth_sig.fs)

#synth_sig.resample(32000)
plt.figure(figsize=(10,5))
plt.subplot(211)
plt.plot(np.arange(.0,synth_sig.length)/float(synth_sig.fs), synth_sig.data)
plt.xticks([])
plt.ylim([-1,1])
plt.grid()
plt.subplot(212)
synth_sig.spectrogram(1024,64, order=0.25, log=False,
                      cmap=cm.hot, cbar=False)

plt.savefig(op.join(figure_output_path, 'glocs_spectro.pdf'))    
plt.show()

