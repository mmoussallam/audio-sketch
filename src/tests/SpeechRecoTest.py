'''
tests.SpeechRecoTest  -  Created on Nov 4, 2013
@author: M. Moussallam

Test if a given fingerprinting setup is able to recognize speech 
uttered by same and/or different speaker

'''

# setup
import unittest
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']

learn_path = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
learn_filenames = get_filepaths(learn_path, 0,  ext='wav')

test_path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_ksp_arctic')
test_filenames = get_filepaths(test_path, 0,  ext='wav')

fgpt_sketches = [
     (CQTPeaksTripletsBDB(None, **{'wall':False,'f1_n_bits':4,
                                   'dt_n_bits':8,'t_targ_width':100,'f_targ_width':12}),
     CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000}))   
#    (SparseFramePairsBDB(None, **{'wall':False,'nb_neighbors_max':10,
#                                             'delta_t_max':0.2, 'f1_n_bits':8,'dt_n_bits':5}),
#     XMDCTSparsePairsSketch(**{'scales':[64,512, 4096],'n_atoms':1,
#                                 'nature':'LOMDCT','pad':False}))                                       
                    ]

sp_per_secs = 50
fs = 8000

def _get_samephrase(phraseindex, remove='jmk'):
    """ get all instances of the given file among all 7 speakers """
    l_path = ['/sons/voxforge/main/Learn/cmu_us_jmk_arctic',
              '/sons/voxforge/main/Learn/cmu_us_ksp_arctic',
              '/sons/voxforge/main/Learn/cmu_us_rms_arctic',
              '/sons/voxforge/main/Learn/cmu_us_slt_arctic',
              '/sons/voxforge/main/Test/cmu_us_awb_arctic',
              '/sons/voxforge/main/Test/cmu_us_bdl_arctic',
              '/sons/voxforge/main/Test/cmu_us_clb_arctic']
    filepaths = []
    for path in l_path:
        if remove in path:
            continue
        filenames = get_filepaths(path, 0,  ext='wav')
        filepaths.append(filenames[phraseindex])
    return filepaths

def _populate(skhandle, fgpthandle, filenames, nbFiles):
    "Inner method that will populate 8 files in the db using the sketch handler provided"           
    for fileIndex in range(nbFiles):
        
        RandomAudioFilePath = filenames[fileIndex]
        if not (RandomAudioFilePath[-3:] == 'wav'):
            continue
        print " Populating %s "%filenames[fileIndex]       
        sig = Signal(RandomAudioFilePath, mono=True, normalize=True)
        sig.resample(fs)
        print sig.get_duration()
        # run the decomposition                        
        skhandle.recompute(sig)
        skhandle.sparsify(int(sp_per_secs* sig.get_duration()))
        # populate
        fgpt = skhandle.fgpt()
        fgpthandle.populate(fgpt, skhandle.params, fileIndex, offset=0)

def _get_keys(skhandle, fgpthandle, reffile):
    sig = Signal(reffile, mono=True, normalize=True)
    sig.resample(fs)
    # run the decomposition                        
    skhandle.recompute(sig)
    skhandle.sparsify(int(sp_per_secs* sig.get_duration()))
    # populate
    fgpt = skhandle.fgpt()
    keys, values = fgpthandle._build_pairs(fgpt, skhandle.params,0)
    return keys

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
    return fgpthandle.get_candidate(fgpt, skhandle.params, nbCandidates=nbFiles, smooth=3)[0]

def _test_prec_recall(skhandle, fgpthandle, test_samefiles, test_difffiles, nb_files, trueidx):
    """ Get the precision/recall -like curve based on the number of similar keys         
    """
    nb_keys_same = []
    nb_keys_diff = []
    for samefile in test_samefiles:
        sig = Signal(samefile, mono=True, normalize=True)
        sig.resample(fs)
        # run the decomposition                        
        skhandle.recompute(sig)
        skhandle.sparsify(int(sp_per_secs* sig.get_duration()))
        # populate
        fgpt = skhandle.fgpt()
        keys, values = fgpthandle._build_pairs(fgpt, skhandle.params,0)
        histograms = fgpthandle.retrieve(fgpt, skhandle.params, nbCandidates= nb_files)
        print samefile[-31:-27],np.sum(histograms[:,trueidx])
        nb_keys_same.append(np.sum(histograms[:,trueidx]))
    
    for difffile in test_difffiles:
        sig = Signal(difffile, mono=True, normalize=True)
        sig.resample(fs)
        # run the decomposition                        
        skhandle.recompute(sig)
        skhandle.sparsify(int(sp_per_secs* sig.get_duration()))
        # populate
        fgpt = skhandle.fgpt()
        keys, values = fgpthandle._build_pairs(fgpt, skhandle.params,0)
        histograms = fgpthandle.retrieve(fgpt,  skhandle.params, nbCandidates= nb_files)
        print difffile[-31:-27],np.sum(histograms[:,trueidx])
        nb_keys_diff.append(np.sum(histograms[:,trueidx]))
    return nb_keys_same, nb_keys_diff
#    print scores, np.max(scores), len(keys)
#    return fgpthandle.get_candidate(fgpt, skhandle.params, nbCandidates=nb_files, smooth=3)[0]


class BuildPrecRecCurveTest(unittest.TestCase):
    """ try the sketches on similar and dissimilar audio files
    get the key overlap ratios for both and hopefully build a Prec-Recall curve
    """
    def __init__(self, nbFiles=10):
        super(BuildPrecRecCurveTest, self).__init__()                     
        self.nbFiles = nbFiles
    
    def runTest(self):
        ref = 'jmk'
        
        for (fgpthandle, skhandle) in fgpt_sketches:
            print "************************************"
            print fgpthandle
            print skhandle                            
            
            _populate(skhandle, fgpthandle,learn_filenames, self.nbFiles)
            
            for fileIndex in range(self.nbFiles):
                
                ref_keys = _get_keys(skhandle, fgpthandle, learn_filenames[fileIndex])
                
                samephrases = _get_samephrase(fileIndex, remove='jmk')
                diffphrases = _get_samephrase(fileIndex+50, remove='jmk')
                
                
                nb_same_keys, nb_diff_keys = _test_prec_recall(skhandle, fgpthandle,
                                                               samephrases, diffphrases,
                                                               self.nbFiles,fileIndex )
                print "File %s"%learn_filenames[fileIndex]
                print "Same ----- ",len(ref_keys)
                print nb_same_keys
                print "Diff -----"
                print nb_diff_keys
                


class SelfRecoTest(unittest.TestCase):
    """ Check the fgpt setup can recognize itself :
    populate a base """
    
    def __init__(self, nbFiles=10):
        super(SelfRecoTest, self).__init__()                     
        self.nbFiles = nbFiles    
                
    def runTest(self):
        for (fgpthandle, skhandle) in fgpt_sketches:
            print "************************************"
            print fgpthandle
            print skhandle            
            tstart = time.time()    
            
            _populate(skhandle, fgpthandle,learn_filenames, self.nbFiles)
            
            for fileIndex in range(self.nbFiles):
                est_idx = _test_retrieve(skhandle, fgpthandle,
                                         learn_filenames, fileIndex, self.nbFiles)
                print "File %d estimated %d"%(fileIndex,est_idx)
                assert fileIndex == est_idx


class DiffRecoTest(unittest.TestCase):
    """ Check the fgpt setup can recognize itself :
    populate a base """
    
    def __init__(self, nbFiles=10):
        super(DiffRecoTest, self).__init__()                     
        self.nbFiles = nbFiles    
                
    def runTest(self):
        for (fgpthandle, skhandle) in fgpt_sketches:
            print "************************************"
            print fgpthandle
            print skhandle            
            tstart = time.time()    
            
            _populate(skhandle, fgpthandle, learn_filenames, self.nbFiles)
            
            for fileIndex in range(self.nbFiles):
                est_idx = _test_retrieve(skhandle, fgpthandle, test_filenames, fileIndex, self.nbFiles)
                print "File %d estimated %d"%(fileIndex,est_idx)
#                assert fileIndex == est_idx
    
if __name__ == "__main__":
    
    suite = unittest.TestSuite()

#    suite.addTest(SelfRecoTest( nbFiles=100))
#    suite.addTest(DiffRecoTest( nbFiles=5))
    suite.addTest(BuildPrecRecCurveTest(nbFiles=5))

    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()