'''
fgpt_scripts.profiling_db  -  Created on Aug 27, 2013
@author: M. Moussallam
'''
'''
fgpt_scripts.many_sparsities  -  Created on Jul 30, 2013
@author: M. Moussallam

Now let us see how performances evolve with the sparsity
'''
import os
import os.path as op
import time
import cProfile
from scipy.io import savemat
from classes.sketches.bench import *
from classes.sketches.cochleo import *
from classes.sketches.cortico import *
from classes import pydb
from tools.fgpt_tools import db_creation, db_test
from tools.fgpt_tools import get_filepaths
db_path = '/home/manu/workspace/temp.db'
import bsddb3.db as db

env = db.DBEnv()
env.set_cachesize(2,512*1024*1024,0)
#if op.exists(op.join(db_path, '__db.001')):    
##    env.remove(db_path)
#    os.remove(op.join(db_path, '__db.001'))
#if op.exists(op.join(db_path, '__db.002')):    
#    os.remove(op.join(db_path, '__db.002'))
#if op.exists(op.join(db_path, '__db.003')):    
#    os.remove(op.join(db_path, '__db.003'))

env.set_thread_count(4)    
env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
env.log_set_config(db.DB_LOG_IN_MEMORY, 1)
env.open(None, env_flags)


## create base
#dbObj = db.DB(env)
#db_flags = db.DB_DUP | db.DB_DUPSORT
#
#
## open
#if op.exists(op.join(db_path, 'profiling.db')):        
#    os.remove(op.join(db_path, 'profiling.db'))
#
#db_name = op.join(db_path,'profiling.db')
#dbObj.set_flags(db_flags)
#dbObj.open(db_name, dbname=db_name, dbtype=db.DB_HASH, flags=db.DB_CREATE)
#
#
#Bin_value = 10001
#Bin_key = 128
#import struct
#Kbin = struct.pack('<I4', Bin_key)
#
#nb_values = 1000000
#
#t = time.time()
#for valueidx in range(nb_values):
#    val = Bin_value + valueidx
#    Bbin = struct.pack('<I4', val)
#    dbObj.put(Kbin, Bbin)
#print "Duplicate writing Took %2.2f sec"%(time.time() - t)
#
#t = time.time()
#for valueidx in range(nb_values):
#    bin = Bin_key + valueidx + 1
#    val = Bin_value + valueidx
#    Bbin = struct.pack('<I4', val)
#    Kbin = struct.pack('<I4', bin)
#    dbObj.put(Kbin, Bbin)
#print "Non-duplicate writing Took %2.2f sec"%(time.time() - t)
#
#
##dbObj.rename(op.join(db_path, 'profiling.db'), None, op.join(db_path, 'profiling.db'))
#
#dbObj.close()
#env.close()
#print env.get_cachesize()
# define a pair FgptHandle/Sketch 

bases = {'RWCLearn':'/sons/rwc/Learn/',
         'voxforge':'/sons/voxforge/main/Learn/',
         'GTZAN':'/home/manu/workspace/databases/genres/'}

# The RWC subset path
#audio_path = '/sons/rwc/Learn'
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path = bases[set_id]
score_path = '/home/manu/workspace/audio-sketch/fgpt_scores'

file_names = get_filepaths(audio_path, 0,  ext='.au')

nb_files = len(file_names)
# define experimental conditions

sparsities = [500,]
seg_dur = 5.0
fs = 8000
step = 3.0
nfiles = 10
## Initialize the sketchifier
#sk = STFTPeaksSketch(**{'scale':2048, 'step':512})
#sk = CorticoIndepSubPeaksSketch(**{'fs':fs,'downsample':fs,'frmlen':8,
#                                   'shift':0,'fac':-2,'BP':1})
sk = CochleoPeaksSketch(**{'fs':fs,'step':512,'downsample':fs,'frmlen':8})
sk_id = sk.__class__.__name__[:-6]

for sparsity in sparsities:    
    # construct a nice name for the DB object to be saved on disk
    db_name = "%s_%s_k%d_%s_%dsec_%dfs.db"%(set_id, sk_id, sparsity, sk.get_sig(),
                                            int(seg_dur), int(fs))
        
    # initialize the fingerprint Handler object
#    fgpthandle = pydb.CorticoIndepSubPeaksBDB(op.join(db_path, db_name),
#                                              load=True,persistent=True,dbenv=env,
#                                               **{'wall':False,'max_pairs':500})
#    fgpthandle = pydb.STFTPeaksBDB(op.join(db_path, db_name),
#                                   load=not learn,
#                                   persistent=True, **{'wall':False})
    fgpthandle = pydb.CochleoPeaksBDB(op.join(db_path, db_name),dbenv=None,
                                    load=False,
                                    persistent=True, **{'wall':False})
    ################# This is a complete experimental run given the setup ############## 
    # create the base:
    commandStr = "db_creation(fgpthandle, sk, sparsity,\
                file_names[:nfiles], \
                force_recompute = True,\
                seg_duration = seg_dur, resample = fs,\
                files_path = audio_path, debug=True, n_jobs=1)"
                         
    cProfile.runctx(commandStr, globals(), locals())   
#    
    
    
#    # run a fingerprinting experiment
#    test_proportion = 1.0 # proportion of segments in each file that will be tested
#    print fgpthandle.dbObj.stat_print()
#    if test:
#        tstart = time.time()
#        scores, failures = db_test(fgpthandle, sk, sparsity,
#                         file_names, 
#                         files_path = audio_path,
#                         test_seg_prop = test_proportion,
#                         seg_duration = seg_dur, resample =fs,
#                         step = step, tolerance = 7.5, shuffle=True, debug=False,n_jobs=1)
#        ttest = time.time() - tstart
#        ################### End of the complete run #####################################
#        # saving the results
#        score_name = "%s_%s_k%d_%s_%dsec_%dfs_test%d_step%d.mat"%(set_id, sk_id, sparsity, sk.get_sig(),
#                                                int(seg_dur), int(fs), int(100.0*test_proportion),int(step))
#        
#        stats =  os.stat(op.join(db_path, db_name))
#        savemat(op.join(score_path,score_name), {'score':scores, 'time':ttest,
#                                                 'size':stats.st_size,'failures': failures})
#        
