'''
tools.sinewave_tools  -  Created on Apr 18, 2013
@author: M. Moussallam
'''

import sys,os
#sys.path.append('/home/manu/workspace/toolboxes/extractFormants/')
#sys.path.append('/home/manu/workspace/toolboxes/extractFormants/bin/')
sys.path.append(os.environ['SKETCH_ROOT'])
here_path = os.environ['SKETCH_ROOT']+'/src/tools'
import os.path as op
#from extractFormants import *


#def extract_formants(audio_file, n_formants=3):
#    """ extract formants from the audio file """

audio_file = os.environ['SND_DB_PATH'] +'/jingles/panzani.wav'
n_formants = 5
maxFormant = 5
time_step = 0.1
windowSize = 0.025
preEmphasis = 50
import os
os.system('praat %s/getsinewavespeech.praat %s %1.3f %d %d %1.3f %d'%(here_path, audio_file,
                                                                time_step,
                                                             n_formants, maxFormant,
                                                             windowSize, preEmphasis))

# now read the audio output and the formant data
formant1_file = audio_file[:-4] + '_formant1.mtxt'
fid = open(formant1_file, 'rU')
vals = fid.readlines()
fid.close()
# remove first 3 lines and convert to numpy
import numpy as np
val_array = np.array(vals[3:],dtype='float')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(val_array)
plt.show()

from PyMP import Signal
sig = Signal(audio_file, normalize=True)
