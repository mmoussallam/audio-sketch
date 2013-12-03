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

set_id = 'voxforge' 
nbfiles = 20
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)[:nbfiles]

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

def test(fgptsystem, target, sparsity):
    (sparsifier, dbhandler) = fgptsystem
    sparsifier.recompute(target)
    sparsifier.sparsify(sparsity)
    return dbhandler.retrieve(sparsifier.fgpt(), sparsifier.params, 0, nbfiles)


def comparekeys(sketchifier, fgpthandle, ref, sparsityref, target, sparsitytarget):
    """ Show the overlapping keys """
    fs = target.fs
    target.spectrogram(2048,256,ax=plt.gca(),order=0.5,log=False,cbar=False,
                         cmap=cm.bone_r, extent=[0,target.get_duration(),0, fs/2])
    sparsifier.recompute(ref)
    sparsifier.sparsify(sparsityref)
    fgpthandle._build_pairs(sparsifier.fgpt(), sparsifier.params,display=True,
                            ax=plt.gca(), color='b')
    sparsifier.recompute(target)
    sparsifier.sparsify(sparsitytarget)
    fgpthandle._build_pairs(sparsifier.fgpt(), sparsifier.params,display=True,
                            ax=plt.gca(), color='r')


# define a sketch
sketchifier = CochleoIHTSketch(**{'downsample':8000,
                                               'frmlen':8,'shift':-1,
                                               'max_iter':2,
                                               'n_inv_iter':2})

###################### Phase 1: define a fingerprint system and parameters
sparsifier = STFTPeaksSketch(**{'downsample':8000,'fs':8000,
                               'frmlen':8,'shift':-1,
                               'max_iter':5,
                               'n_inv_iter':2})
fgpthandle = STFTPeaksBDB('STFTPeaks.db', **{'wall':False,'time_max':20.0})


sparsity = 0.01

###################### Phase 2: Train on unaltered samples
learn((sparsifier,fgpthandle),file_names, sparsity)



###################### Phase 3: Test on "sketchified" sampels
rndidx = 3
sketchifier.recompute(file_names[rndidx])
sketchifier.sparsify(0.1)
print np.count_nonzero(sketchifier.sp_rep)
resynth = sketchifier.synthesize(sparse=True)

# now test the KOR 
# reference ? 
refhist = test((sparsifier,fgpthandle),file_names[rndidx], sparsity)
testhist = test((sparsifier,fgpthandle),resynth, sparsity)

#plt.figure()
#comparekeys(sketchifier, fgpthandle, file_names[rndidx], sparsity, resynth, sparsity)
#plt.show()

# what is the metric ? contrast-to-noise ratio?
# let us use two metrics: the distance to noise and the distance to second
scores = np.sum(testhist, axis=0)/np.sum(refhist[:,rndidx])
# the masked array will use all elements EXCEPT the one where the mask is TRUE
masked_scores = np.ma.array(scores, mask=False)
masked_scores.mask[rndidx] = True

score = scores[rndidx]

metric1 = score - np.mean(masked_scores)
metric2 = score - np.max(masked_scores)

plt.figure()
#plt.plot(np.sum(refhist, axis=0))
plt.plot(scores,'o')
plt.show()
