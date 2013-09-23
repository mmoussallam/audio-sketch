import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
import sys
import os.path as op
from tools import cochleo_tools
#from classes import sketch
import matplotlib.pyplot as plt
from PyMP import Signal

from scipy.signal import lfilter, hann
#audio_test_file = '/home/manu/workspace/recup_angelique/Sketches/NLS Toolbox/Hand-made Toolbox/forAngelique/61_sadness.wav'
audio_test_file = op.abspath('./audio/original_surprise.wav')
audio_name ='surprise'
from classes.sketches.bench import *
from classes.sketches.misc import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes.pydb import *

fgpthandle = STFTPeaksBDB(None, **{'wall':False})                          
sk = STFTPeaksSketch(**{'scale':2048, 'step':512})

orig_sig = Signal(audio_test_file, normalize=True, mono=True)
noisy_sig = Signal(orig_sig.data + 0.2*np.random.randn(orig_sig.length), orig_sig.fs, normalize=True, mono=True)
sk.recompute(orig_sig)
sk.sparsify(20)

plt.figure(figsize=(10,6))
plt.subplot(221)
orig_sig.spectrogram(512, 128, order=2, log=True, ax=plt.gca(), cmap=cm.bone_r, cbar=False)
plt.subplot(222)
noisy_sig.spectrogram(512, 128,order=2, log=True, ax=plt.gca(), cmap=cm.bone_r, cbar=False)
plt.subplot(223)
fgpthandle._build_pairs(sk.fgpt(), sk.params, display=True, ax=plt.gca())
plt.gca().invert_yaxis()
plt.yticks([])
plt.xticks([])
sk.recompute(noisy_sig)
sk.sparsify(20)
plt.subplot(224)
fgpthandle._build_pairs(sk.fgpt(), sk.params, display=True, ax=plt.gca())
plt.gca().invert_yaxis()
plt.yticks([])
plt.xticks([])
plt.subplots_adjust(left=0.09, bottom=0.05, right=0.96, top=0.97)

fig_path = os.path.abspath('../reporting/figures/')
plt.savefig(os.path.join(fig_path,'noisy_keys.pdf'))
plt.show()