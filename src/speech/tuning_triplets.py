'''
speech.tuning_triplets  -  Created on Nov 7, 2013
@author: M. Moussallam
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']

learn_path = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
learn_filenames = get_filepaths(learn_path, 0,  ext='wav')

test_path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_ksp_arctic')
test_filenames = get_filepaths(test_path, 0,  ext='wav')


fgpthandle = CQTPeaksTripletsBDB(None, **{'wall':False,'f1_n_bits':5,'dt_n_bits':4})
skhandle = CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000})                                        
                    
print fgpthandle.delta

sp_per_secs = 50
fs = 8000

def _populate(skhandle, fgpthandle, filenames, nbFiles):
    "Inner method that will populate 8 files in the db using the sketch handler provided"           
    for fileIndex in range(nbFiles):
        
        RandomAudioFilePath = filenames[fileIndex]
        if not (RandomAudioFilePath[-3:] == 'wav'):
            continue
        print " Populating %s "%filenames[fileIndex]       
        sig = Signal(RandomAudioFilePath, mono=True, normalize=True)
        sig.resample(fs)
        # run the decomposition                        
        skhandle.recompute(sig)
        skhandle.sparsify(int(sp_per_secs* sig.get_duration()))
        # populate
        fgpt = skhandle.fgpt()
        fgpthandle.populate(fgpt, skhandle.params, fileIndex, offset=0)


def _test_retrieve(skhandle, fgpthandle,filenames, fileIndex, nbFiles):
    " test whether the retrieved index is the good one"               
    RandomAudioFilePath = filenames[fileIndex]        
    sig = Signal(RandomAudioFilePath, mono=True, normalize=True)
    sig.resample(fs)
    # run the decomposition                        
    skhandle.recompute(sig)
    skhandle.sparsify(int(sp_per_secs* sig.get_duration()))
    # populate
    fgpt = skhandle.fgpt()
    keys, values = fgpthandle._build_pairs(fgpt, skhandle.params,0)
    histograms = fgpthandle.retrieve(fgpt,  skhandle.params, nbCandidates= nbFiles)
    scores = np.sum(histograms, axis=0)
    print scores, np.max(scores), len(keys)
    return fgpthandle.get_candidate(fgpt, skhandle.params, nbCandidates=nbFiles, smooth=1)[0]


_populate(skhandle, fgpthandle,learn_filenames, 1)
skhandle.represent(sparse=True)
plt.show()
