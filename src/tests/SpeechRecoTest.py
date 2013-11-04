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
    (STFTPeaksBDB('STFTPeaks.db', **{'wall':False}),
     STFTPeaksSketch(**{'scale':2048, 'step':512})),                                             
                    ]

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
    return fgpthandle.get_candidate(fgpt, skhandle.params, nbCandidates=nbFiles, smooth=1)[0]



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
                assert fileIndex == est_idx
    
if __name__ == "__main__":
    
    suite = unittest.TestSuite()

    suite.addTest(SelfRecoTest( nbFiles=5))
    suite.addTest(DiffRecoTest( nbFiles=5))

    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()