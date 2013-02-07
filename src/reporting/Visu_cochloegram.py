'''
reporting.Visu_cochloegram  -  Created on Feb 7, 2013
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
plt.switch_backend('Agg')
audio_test_file = '/home/manu/workspace/recup_angelique/Sketches/NLS Toolbox/Hand-made Toolbox/forAngelique/61_sadness.wav'

figure_output_path = '/home/manu/workspace/audio-sketch/src/reporting/figures/'

sig = Signal(audio_test_file, mono=True, normalize=True)
sig.downsample(16000)
scale = 512
step=128

def plot_spectrogram(sig_stft, scale=512, step=128):
    plt.figure()
    plt.imshow(20*np.log10(np.abs(sig_stft[0,:,:])),
               aspect='auto',
               origin='lower',
               interpolation='nearest',
               cmap = cm.copper_r)
    
    x_tick_vec = (np.linspace(0, sig_stft.shape[2], 10)).astype(int)
    x_label_vec = (x_tick_vec*float(step))/float(sig.fs)
    
    y_tick_vec = (np.linspace(0, sig_stft.shape[1], 6)).astype(int)
    y_label_vec = (y_tick_vec/float(scale))*float(sig.fs)
    
    plt.xlabel('Time (s)')
    plt.xticks(x_tick_vec, ["%1.1f"%a for a in x_label_vec])
    plt.ylabel('Frequency')
    plt.yticks(y_tick_vec, ["%d"%int(a) for a in y_label_vec])

# do the stft of the signal
sig_stft = stft.stft(sig.data, scale, step);
plot_spectrogram(sig_stft)
plt.savefig(op.join(figure_output_path, 'exemple_stft.pdf'))


# do the auditory spectrogram of the signal
gram = cochleo_tools.cochleogram(sig.data, load_coch_filt=True)
gram.build_aud()

gram.plot_aud(duration = float(sig.length)/float(sig.fs))

plt.savefig(op.join(figure_output_path, 'exemple_cochleogram.pdf'))
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')


init_vec = gram.init_inverse()
inv_data = gram.invert(init_vec=init_vec, nb_iter=15);

rec_stft = stft.stft(inv_data, scale, step);
plot_spectrogram(rec_stft)
plt.savefig(op.join(figure_output_path, 'exemple_stft_inv_cochleogram.pdf'))

plt.show()

# invert it and plot the spectrogram
