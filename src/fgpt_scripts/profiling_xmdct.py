# what is faster ? reading from disk or computing corticograms?
import os
import os.path as op
import time
from scipy.io import savemat
from classes.pydb import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes.sketches.bench import *
from tools.fgpt_tools import db_creation, db_test
from tools.fgpt_tools import get_filepaths
import cProfile
import bsddb.db as db
env = db.DBEnv()
env.set_cachesize(0,256*1024*1024,0)
print env.get_cachesize()
# The RWC subset path
audio_path = '/home/manu/workspace/databases/genres/'
db_path = '/home/manu/workspace/audio-sketch/fgpt_db'
score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'

file_names = get_filepaths(audio_path, 0,  ext='au')
nb_files = len(file_names)
# define experimental conditions
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
sparsity = 150
seg_dur = 5.0
fs = 8000
downfs = fs


#sk = CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs, 'sub_slice':(2,9)})

# sk = CorticoSubPeaksSketch(**{'fs':fs,'step':128,'downsample':fs, 'sub_slice':(2,9),
#                               'pre_comp':'/media/manu/TOURO/corticos'})
sk = XMDCTSparseSketch(**{'scales':[ 2048, 4096],'n_atoms':30,'nature':'LOMDCT'})

sk_id = sk.__class__.__name__[:-6]
# construct a nice name for the DB object to be saved on disk
db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                        int(seg_dur), int(fs))
fgpthandle = XMDCTBDB(op.join(db_path, db_name),
                                   load=False,
                                   persistent=True, **{'wall':False})

# PROFILING Without pre-compute
cProfile.runctx("db_creation(fgpthandle, sk, sparsity, \
                file_names[:2], \
                force_recompute = True,\
                seg_duration = seg_dur, resample = fs,\
                files_path = audio_path, debug=True, n_jobs=2)", globals(), locals())
