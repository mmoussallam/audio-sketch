'''
fgpt_scripts.visu_dataset_stats  -  Created on Jul 1, 2013
@author: M. Moussallam
'''
import os
import os.path as op
from classes import pydb, sketch
from tools.fgpt_tools import db_creation, db_test
from PyMP.signals import LongSignal
import numpy as np
# The RWC subset path
audio_path = '/sons/rwc/Learn'
db_path = '/home/manu/workspace/audio-sketch/fgpt_db' 

file_names = [f for f in os.listdir(audio_path) if '.wav' in f]
nb_files = len(file_names)


# dataset length in seconds
dur = []
for fileIndex in range(nb_files):
    l_sig = LongSignal(op.join(audio_path, file_names[fileIndex]), 
                               frame_duration=5.0, 
        mono=True, 
        Noverlap=0)
    dur.append(l_sig.n_seg * 5.0)

tot = np.sum(dur)
hours = int(np.floor(tot / 3600))
minutes =  int(np.floor((tot - (hours* 3600))/60))
print "Dataset last %d hours and %d minutes"%(hours, minutes)
