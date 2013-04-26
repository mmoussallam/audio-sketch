'''
tests.DBTests  -  Created on Apr 24, 2013

A simple set of tests verifying that all FGPThandles 
are correctly implemented

@author: M. Moussallam
'''
import unittest
import sys
import numpy as np
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

from classes import pydb, sketch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

learn_dir = '/sons/rwc/Learn/'
test_dir = '/sons/rwc/Test/'
single_test_file1 = '/sons/sqam/voicemale.wav'
single_test_file2 = '/sons/sqam/voicefemale.wav'

class FgptTest(unittest.TestCase):
    """ testing the fingerprinting """ 
    
    def runTest(self):
        
        abstractFGPT = pydb.FgptHandle('abstract.db')
        print abstractFGPT
        self.assertRaises(NotImplementedError,abstractFGPT.add, None,0)
        self.assertRaises(NotImplementedError,abstractFGPT.retrieve, None, None)
        self.assertRaises(NotImplementedError,abstractFGPT.populate, None, None,0)
        self.assertRaises(NotImplementedError,abstractFGPT.get, None)
        
        fgpt_sketches = [(pydb.STFTPeaksBDB('STFTPeaks.db'),
                          sketch.STFTPeaksSketch(**{'scale':2048, 'step':512})), 
                         (pydb.XMDCTBDB('xMdct.db'),
                          sketch.XMDCTSparseSketch(**{'scales':[64,512,2048],'n_atoms':100})),                                     
                            ]
        
        # for all sketches, we performe the same testing
        for (fgpthand, sk) in fgpt_sketches:
            print fgpthand
            print sk
            
            # Initialize the sketch
            sk.recompute(single_test_file1)
            sk.sparsify(300)
            # convert it to a fingeprint compatible with associated handler
            fgpt = sk.fgpt(sparse=True)
            params = sk.params
#            print fgpt
            # check that the handler is able to process the fingerprint            
            print "Here the params: ",sk.params
            fgpthand.populate(fgpt, sk.params, 0)
            
            # Do the same with the second file
            sk.recompute(single_test_file2)
            sk.sparsify(300)
            print sk.params
            fgpthand.populate(sk.fgpt(sparse=True), sk.params, 1)
            
            # check that the handler can recover the first one
            # does it build a coherent histogram matrix
            self.assertNotEqual(fgpt, sk.fgpt(sparse=True))
            hist = fgpthand.retrieve(fgpt, params, nbCandidates=2)
            self.assertIsNotNone(hist)
            print "Score for first is %d Score for second is %d"%(np.max(hist[:,0]),
                                                                  np.max(hist[:,1]))
            
            plt.figure()
            from scipy.ndimage.filters import median_filter            
            plt.plot(median_filter(hist, (3, 1)))
            # is the best candidate the good one
            estimated_index, estimated_offset  = fgpthand.get_candidate(fgpt,sk.params,
                                                                        nbCandidates=2, smooth=3)
            print "Guessed %d with offset %1.1f s"%(estimated_index, estimated_offset)
            self.assertEqual(0, estimated_index)
            self.assertGreater(5.0, estimated_offset)
            

if __name__ == "__main__":
    
    suite = unittest.TestSuite()

    suite.addTest(FgptTest())

    unittest.TextTestRunner(verbosity=2).run(suite)
#    plt.show()