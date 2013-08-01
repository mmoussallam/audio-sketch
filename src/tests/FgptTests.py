'''
tests.DBTests  -  Created on Apr 24, 2013

A simple set of tests verifying that all FGPThandles 
are correctly implemented

@author: M. Moussallam
'''
import unittest
import os
import sys
import numpy as np
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

#from classes import pydb, sketch
from classes.pydb import *
from classes.sketches.cortico import *
from classes.sketches.cochleo import *
from PyMP.signals import LongSignal, Signal
import os.path as op
import matplotlib.pyplot as plt
from joblib import Memory
mem = Memory(cachedir='/tmp/fgpt')

plt.switch_backend('Agg')

learn_dir = '/sons/rwc/Learn/'
test_dir = '/sons/rwc/Test/'
single_test_file1 = '/sons/sqam/voicemale.wav'
single_test_file2 = '/sons/sqam/voicefemale.wav'

audio_files_path = '/sons/rwc/rwc-p-m07'
file_names = os.listdir(audio_files_path)

#class FgptTest(unittest.TestCase):
#    """ testing the fingerprinting """ 
#    
#    def runTest(self):
        
#abstractFGPT = pydb.FgptHandle('abstract.db')

#self.assertRaises(NotImplementedError,abstractFGPT.add, None,0)
#self.assertRaises(NotImplementedError,abstractFGPT.retrieve, None, None)
#self.assertRaises(NotImplementedError,abstractFGPT.populate, None, None,0)
#self.assertRaises(NotImplementedError,abstractFGPT.get, None)

fgpt_sketches = [
#                 (pydb.STFTPeaksBDB('STFTPeaks.db', **{'wall':False}),
#                  sketch.STFTPeaksSketch(**{'scale':2048, 'step':512})), 
#                 (pydb.CochleoPeaksBDB('CochleoPeaks.db', **{'wall':False}),
#                  sketch.CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),
#                 (pydb.XMDCTBDB('xMdct.db', load=False,**{'wall':False}),
#                  sketch.XMDCTSparseSketch(**{'scales':[ 4096],'n_atoms':150,
#                                              'nature':'LOMDCT'})),         
#                 (CochleoPeaksBDB('CorticoSub_0_0Peaks.db', **{'wall':False}),
#                  CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),
#                  CorticoSubPeaksSketch(**{'fs':8000,'step':128,'downsample':8000,'sub_slice':(4,11)})),
                    (CorticoIndepSubPeaksBDB('Cortico_subs', **{'wall':False}),
                     CorticoIndepSubPeaksSketch(**{'fs':8000,'frmlen':8,'downsample':8000}))                                             
                    ]

#@mem.cache
def populate(sk, fgpthand):
    segDuration = 5        
    sig = LongSignal(op.join(audio_files_path, file_names[0]),
                             frame_duration=segDuration, mono=True, Noverlap=0)

    segmentLength = sig.segment_size
    max_seg_num = 5
#        " run sketchifier on a number of files"
    nbFiles = 8
    keycount = 0
    for fileIndex in range(nbFiles):
        RandomAudioFilePath = file_names[fileIndex]        
        if not (RandomAudioFilePath[-3:] == 'wav'):
            continue

        pySig = LongSignal(op.join(audio_files_path, RandomAudioFilePath),
            frame_size=segmentLength, mono=True, Noverlap=0)
        
        nbSeg = int(pySig.n_seg)
        print 'Working on ' + str(RandomAudioFilePath) + ' with ' + str(nbSeg) + ' segments'
        for segIdx in range(min(nbSeg, max_seg_num)):
            pySigLocal = pySig.get_sub_signal(
                segIdx, 1, True, True, channel=0, pad=0, fast_create=True)
            print "sketchify the segment %d" % segIdx
            # run the decomposition                        
            sk.recompute(pySigLocal)
            sk.sparsify(300)
            fgpt = sk.fgpt()
            print "Populating database with offset " + str(segIdx * segmentLength / sig.fs)
            fgpthand.populate(fgpt, sk.params, fileIndex, offset=segIdx*segDuration)

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
    fgpthand.populate(sk.fgpt(sparse=True), sk.params, 1)
    
    # check that the handler can recover the first one
    # does it build a coherent histogram matrix
#    self.assertNotEqual(fgpt, sk.fgpt(sparse=True))
    assert not fgpt==sk.fgpt(sparse=True)
    hist = fgpthand.retrieve(fgpt, params, nbCandidates=2)
#    self.assertIsNotNone(hist)
    assert hist is not None
    print hist.shape
    print "Score for first is %d Score for second is %d"%(np.max(hist[:,0]),
                                                          np.max(hist[:,1]))
    
    plt.figure()
    from scipy.ndimage.filters import median_filter            
    plt.plot(median_filter(hist, (3, 1)))
    plt.title(fgpthand.__class__)
    # is the best candidate the good one
    estimated_index, estimated_offset  = fgpthand.get_candidate(fgpt,sk.params,
                                                                nbCandidates=2, smooth=3)
    print "Guessed %d with offset %1.1f s"%(estimated_index, estimated_offset)
#    self.assertEqual(0, estimated_index)
#    self.assertGreater(5.0, estimated_offset)
    assert estimated_index == 0
    assert estimated_offset < 6.0
    
    # Now the last of the test: populate a base of a few dozens musical samples
    populate(sk, fgpthand)
    
    # and retrieve a segment in the base
    true_file_index = 3
    true_offset = 11.5
    
    
    
    # get the fingerprint 
    true_file_path = op.join(audio_files_path, file_names[true_file_index])
    true_l_sig = LongSignal(true_file_path,  frame_duration=true_offset)
    true_sig = true_l_sig.get_sub_signal(1, 1, mono=True, normalize=True)
    true_sig.crop(0, 5.0*true_sig.fs)
    sk.recompute(true_sig)
    sk.sparsify(300)    
    test_fgpt = sk.fgpt(sparse=True)

    hist =  fgpthand.retrieve(test_fgpt, sk.params, nbCandidates=8)
    estimated_index, estimated_offset  = fgpthand.get_candidate(test_fgpt,sk.params,
                                                                nbCandidates=8, smooth=1)

    assert estimated_index == true_file_index
    assert np.abs(estimated_offset - true_offset) <= 5

    plt.plot(hist.T)
    plt.show()

#if __name__ == "__main__":
#    
#    suite = unittest.TestSuite()
#
#    suite.addTest(FgptTest())
#
#    unittest.TextTestRunner(verbosity=2).run(suite)
#    plt.show()