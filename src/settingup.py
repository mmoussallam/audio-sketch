'''
manu_sandbox.settingup  -  Created on Oct 4, 2013
@author: M. Moussallam
'''
import os
import os.path as op
import time
from scipy.io import savemat, loadmat
import sys

from joblib import Memory
from src.classes.sketches.base import AudioSketch
from src.classes.sketches.bench import *
from src.classes.sketches.cortico import *
from src.classes.sketches.cochleo import *
from src.tools import stft, cqt


from src.classes.fingerprints import *
from src.classes.fingerprints.bench import *
from src.classes.fingerprints.cortico import *
from src.classes.fingerprints.cochleo import *
from src.classes.fingerprints.CQT import *
from tools.fgpt_tools import db_creation, db_test
from tools.fgpt_tools import get_filepaths
from tools.stft import stft, istft
from PyMP.signals import Signal, LongSignal

SKETCH_ROOT = os.environ['SKETCH_ROOT']
SND_DB_PATH = os.environ['SND_DB_PATH']

sys.path.append(SKETCH_ROOT)

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(0,512*1024*1024,0)
env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
env.log_set_config(db.DB_LOG_IN_MEMORY, 1)
env.open(None, env_flags)

bases = {'RWCLearn':(op.join(SND_DB_PATH,'rwc/Learn/'),'.wav'),
         'voxforge':(op.join(SND_DB_PATH,'voxforge/main/Learn/'),'wav'),
         'GTZAN':(op.join(SND_DB_PATH,'genres/'),'.au')}


#from sklearn.neighbors import NearestNeighbors
from cProfile import runctx