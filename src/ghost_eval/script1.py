'''
ghost_eval.script1  -  Created on Dec 3, 2013

Ok so we want to evaluate How a sketchification affects the recognition
This is an example script to illustrate the evaluation
@author: M. Moussallam
'''

# All the imports in one
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
from expe_tools import *

set_id = 'voxforge' 
nbfiles = 20
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)[:nbfiles]


# define a sketch
sketchifier = CochleoIHTSketch(**{'downsample':8000,
                                               'frmlen':8,'shift':-1,
                                               'max_iter':2,
                                               'n_inv_iter':2})

###################### Phase 1: define a fingerprint system and parameters
#sparsifier = CochleoPeaksSketch(**{'downsample':8000,'fs':8000,
#                               'frmlen':8,'shift':-1,
#                               'max_iter':5,
#                               'n_inv_iter':2})
#fgpthandle = CochleoPeaksBDB('CochleoPeaks.db', **{'wall':False,'time_max':20.0})


sparsifier = XMDCTSparsePairsSketch(**{'scales':[64,256,1024],'n_atoms':1,'fs':8000,
                                 'nature':'MDCT','pad':False,'downsample':8000})

fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                  'f1_n_bits':6,
                                                                  'time_max':20.0,
                                                              'nb_neighbors_max':5,
                                                              'delta_f_bits':6,
                                                              'delta_f_min':250,
                                                              'delta_f_max':2000,
                                                              'delta_t_min':0.5, 
                                                              'delta_t_max':2.0})

sparsity = 0.001

###################### Phase 2: Train on unaltered samples
learn((sparsifier,fgpthandle),file_names, sparsity)



###################### Phase 3: Test on "sketchified" sampels
rndidx = 3
sketchifier.recompute(file_names[rndidx])
sketchifier.sparsify(0.05)
print np.count_nonzero(sketchifier.sp_rep)
resynth = sketchifier.synthesize(sparse=True)

# now test the KOR 
# reference ? 
refhist = test((sparsifier,fgpthandle),file_names[rndidx], sparsity)
testhist = test((sparsifier,fgpthandle),resynth, sparsity)

############ debug mode 
#sparsifier.recompute(file_names[rndidx])
#sparsifier.sparsify(sparsity)
#keysref, valuesref = fgpthandle._build_pairs(sparsifier.fgpt(),sparsifier.params)
#sparsifier.recompute(resynth)
#sparsifier.sparsify(sparsity)
#keys, values = fgpthandle._build_pairs(sparsifier.fgpt(),sparsifier.params)
plt.figure()
comparekeys(sparsifier, fgpthandle, file_names[rndidx], sparsity, resynth, sparsity)
plt.show()

# what is the metric ? contrast-to-noise ratio?
# let us use two metrics: the distance to noise and the distance to second
scores = np.sum(testhist, axis=0)/np.sum(refhist[:,rndidx])
# the masked array will use all elements EXCEPT the one where the mask is TRUE
masked_scores = np.ma.array(scores, mask=False)
masked_scores.mask[rndidx] = True

score = scores[rndidx]

metric1 = score - np.mean(masked_scores)
metric2 = (score - np.max(masked_scores))/score

plt.figure()
#plt.plot(np.sum(refhist, axis=0))
plt.plot(scores,'o')
plt.show()
