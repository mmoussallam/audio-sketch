'''
ghost_eval.expe_sws_xmdct  -  Created on Dec 10, 2013
@author: M. Moussallam
'''

import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
from src.ghost_eval.expe_tools import *

set_id = 'voxforge' 
nbfiles = 10
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)[:nbfiles]

def testratio_specific(sketchifier,fgptsystem, file_names, testratio, tstep, refsparsity):
    (sparsifier, fgpthandle) = fgptsystem
    rndindexes = np.random.random_integers(0,len(file_names)-1,testratio*len(file_names))
    metric = []
    for rndidx in rndindexes:
        print "Working on ",file_names[rndidx]
        sketchifier.recompute(file_names[rndidx])
        sparsity = tstep/float(sketchifier.orig_signal.get_duration())
        print "Sparsity is : ",sparsity
        sketchifier.sparsify(sparsity)
        print np.count_nonzero(sketchifier.sp_rep)
        resynth = sketchifier.synthesize(sparse=True)    
        # now test the KOR 
        # reference ? 
        refhist = test((sparsifier,fgpthandle),file_names[rndidx], refsparsity, len(file_names))
        testhist = test((sparsifier,fgpthandle),resynth, refsparsity, len(file_names))
        scores = np.sum(testhist, axis=0)/np.sum(refhist[:,rndidx])
        print scores
        # the masked array will use all elements EXCEPT the one where the mask is TRUE
        masked_scores = np.ma.array(scores, mask=False)
        masked_scores.mask[rndidx] = True        
        score = scores[rndidx]
        if score>0:            
            metric.append((score - np.max(masked_scores))/score)
        else:
            metric.append(-100)
        print "Score of ",metric[-1]
    return metric


###### SCRIPT SPECIFIC WE ARE HERE INTERESTED IN THIS TYPE OF FINGERPRINT
# FGPT PARAMETERS
scales = [64,256,1024]
nature = 'LOMDCT'
fgptsparsity = 0.005 # ratio of sparse elements
sparsifier = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,'fs':8000,
                                 'nature':nature,'pad':False,'downsample':8000})

fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                  'f1_n_bits':6,
                                                              'nb_neighbors_max':3,
                                                              'delta_f_bits':6,
                                                              'delta_f_min':250,
                                                              'delta_f_max':2000,
                                                              'delta_t_min':0.5, 
                                                              'delta_t_max':2.0})


##### FIRST PHASE LEARN THE BASE
learn((sparsifier,fgpthandle),file_names, fgptsparsity)


##### SECOND PHASE TEST WITH VARIOUS SKETCHES 
test_ratio = 0.5

sketches_to_test = [
             SWSSketch(**{'n_formants': 1,'n_formants_max': 7}),
#              SWSSketch(**{'n_formants': 2,'n_formants_max': 7}),
              SWSSketch(**{'n_formants': 3,'n_formants_max': 7}),
#              SWSSketch(**{'n_formants': 4,'n_formants_max': 7}),
            SWSSketch(**{'n_formants': 5,'n_formants_max': 7}),
            
             ]

time_steps_to_test = [0.1, 0.05, 0.01]
dist2second = np.zeros((len(sketches_to_test), len(time_steps_to_test), test_ratio*len(file_names)))
sklegends = []
for skidx, sketchifier in enumerate(sketches_to_test):
    sklegends.append("%d Formants"%sketchifier.params['n_formants'])
    print sklegends[-1]
    for spidx, time_step in enumerate(time_steps_to_test):
                        
        dist2second[skidx, spidx,:] = testratio_specific(sketchifier,
                                                      (sparsifier,fgpthandle),
                                                      file_names, test_ratio,
                                                      time_step, fgptsparsity)
        print sketchifier.params

plt.figure()
plt.plot(time_steps_to_test,np.mean(dist2second, axis=2).T)
plt.legend(sklegends, loc='lower right')
plt.grid()
plt.show()