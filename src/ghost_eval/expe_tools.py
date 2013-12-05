'''
ghost_eval.expe_tools  -  Created on Dec 5, 2013
@author: M. Moussallam
'''

import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *

def learn(fgptsystem, filenames, sparsity, debug=0):
    """ Given a fingerprinting system (sparsifier + landmark builder)
    analyse all files in the list and populate the berkeley db object"""
    (sparsifier, dbhandler) = fgptsystem
    for fileIndex, filename in enumerate(filenames):
        print fileIndex
        # sparsify
        sparsifier.recompute(filename,**{'sig_name':filename})
        sparsifier.sparsify(sparsity)
        
        # and build the landmarks        
        dbhandler.populate(sparsifier.fgpt(), sparsifier.params,
                           fileIndex, offset=0, debug=debug)        

def test(fgptsystem, target, sparsity, nbfiles):
    (sparsifier, dbhandler) = fgptsystem
    sparsifier.recompute(target)
    sparsifier.sparsify(sparsity)
    return dbhandler.retrieve(sparsifier.fgpt(), sparsifier.params, 0, nbfiles)


def testratio(sketchifier,fgptsystem, file_names, testratio, sparsity):
    (sparsifier, fgpthandle) = fgptsystem
    rndindexes = np.random.random_integers(0,len(file_names)-1,testratio*len(file_names))
    metric = []
    for rndidx in rndindexes:
        print "Working on ",file_names[rndidx]
        sketchifier.recompute(file_names[rndidx])
        sketchifier.sparsify(sparsity)
        print np.count_nonzero(sketchifier.sp_rep)
        resynth = sketchifier.synthesize(sparse=True)    
        # now test the KOR 
        # reference ? 
        refhist = test((sparsifier,fgpthandle),file_names[rndidx], sparsity, len(file_names))
        testhist = test((sparsifier,fgpthandle),resynth, sparsity, len(file_names))
        scores = np.sum(testhist, axis=0)/np.sum(refhist[:,rndidx])
        # the masked array will use all elements EXCEPT the one where the mask is TRUE
        masked_scores = np.ma.array(scores, mask=False)
        masked_scores.mask[rndidx] = True        
        score = scores[rndidx]            
        metric.append((score - np.max(masked_scores))/score)
        print "Score of ",metric[-1]
    return metric

def comparekeys(sketchifier, fgpthandle, ref, sparsityref, target, sparsitytarget):
    """ Show the overlapping keys """
    fs = target.fs
    target.spectrogram(2048,256,ax=plt.gca(),order=0.5,log=False,cbar=False,
                         cmap=cm.bone_r, extent=[0,target.get_duration(),0, fs/2])
    sparsifier.recompute(ref)
    sparsifier.sparsify(sparsityref)
    fgpthandle._build_pairs(sparsifier.fgpt(), sparsifier.params,display=True,
                            ax=plt.gca(), color='m')
    sparsifier.recompute(target)
    sparsifier.sparsify(sparsitytarget)
    fgpthandle._build_pairs(sparsifier.fgpt(), sparsifier.params,display=True,
                            ax=plt.gca(), color='y')

