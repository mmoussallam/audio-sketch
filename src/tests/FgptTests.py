'''
tests.DBTests  -  Created on Apr 24, 2013

A simple set of tests verifying that all FGPThandles 
are correctly implemented

@author: M. Moussallam
'''
import unittest
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *

#from joblib import Memory
#mem = Memory(cachedir='/tmp/fgpt')
#plt.switch_backend('Agg')

SND_DB_PATH = os.environ['SND_DB_PATH']

single_test_file1 = op.join(SND_DB_PATH,'sqam/voicefemale.wav')
single_test_file2 = op.join(SND_DB_PATH,'sqam/voicemale.wav')

audio_files_path =  op.join(SND_DB_PATH,'rwc/rwc-p-m07')
file_names = os.listdir(audio_files_path)

fgpt_sketches = [
#     (SWSBDB('SWSdeltas.db', **{'wall':False,'n_deltas':2}),                  
#     SWSSketch(**{'n_formants_max':7,'time_step':0.02})), 
#    (STFTPeaksBDB('STFTPeaks.db', **{'wall':False}),
#     STFTPeaksSketch(**{'scale':2048, 'step':512})), 
#   (CochleoPeaksBDB('CochleoPeaks.db', **{'wall':False}),
#    CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),
##     (XMDCTBDB('xMdct.db', load=False,**{'wall':False}),
##      XMDCTSparseSketch(**{'scales':[ 4096],'n_atoms':150,
##                                 'nature':'LOMDCT'})),
#                     (CochleoPeaksBDB(None, **{'wall':False}),
#                     CochleoPeaksSketch(**{'fs':8000.0,'step':128,'downsample':8000.0,'frmlen':6})),
#    (CQTPeaksBDB('CQTPeaks.db', **{'wall':False}),
#     CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000})),  
# (CQTPeaksTripletsBDB(None, **{'wall':False}),
#     CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000}))                        
        (CorticoIndepSubPeaksBDB('Cortico_subs', **{'wall':False}),
         CorticoIndepSubPeaksSketch(**{'fs':8000,'frmlen':8,'downsample':8000})) 
                                            
                    ]


class FgptTest(unittest.TestCase):
    """ testing the fingerprinting """ 
    
    def populate_test(self, skhandle, fgpthandle):
        "Inner test that will populate 8 files in the db using the sketch handler provided"
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
#            print RandomAudioFilePath
            if not (RandomAudioFilePath[-3:] == 'wav'):
                continue
    
            pySig = LongSignal(op.join(audio_files_path, RandomAudioFilePath),
                frame_size=segmentLength, mono=True, Noverlap=0)
            
            nbSeg = int(pySig.n_seg)
            print 'Working on ' + str(RandomAudioFilePath) + ' with ' + str(nbSeg) + ' segments'
            for segIdx in range(min(nbSeg, max_seg_num)):
                pySigLocal = pySig.get_sub_signal(
                    segIdx, 1, True, True, channel=0, pad=0, fast_create=False)
                print ".",                           
#                print "sketchify the segment %d" % segIdx 
                # run the decomposition                        
                skhandle.recompute(pySigLocal)
                skhandle.sparsify(150)
                fgpt = skhandle.fgpt()
#                print "Populating database with offset " + str(segIdx * segmentLength / sig.fs)
                fgpthandle.populate(fgpt, skhandle.params, fileIndex, offset=segIdx*segDuration)
            print 'done, max offset of %d seconds'%(segIdx*segDuration)
        
    
    def add_get_test(self, skhandle, fgpthandle, display=False):
        # Initialize the sketch handle
        skhandle.recompute(single_test_file1)
        skhandle.sparsify(20)
        print "Calling sketch handle fgpt method"
        fgpt = skhandle.fgpt(sparse=True)
        params = skhandle.params
        if display:
            print fgpt
        print "Checking that the fgpt handle is able to process the result - populate..",                    
        fgpthandle.populate(fgpt, skhandle.params, 0,display=True)
#        plt.show()
        print " retrieve"
        anchor = np.sum(fgpthandle.retrieve(fgpt, skhandle.params, nbCandidates=1))
        print "fgpt handle has built and retrieved %d keys "%anchor        
        
        print " Do the same with the second file"
        skhandle.recompute(single_test_file2)
        skhandle.sparsify(150)    
        fgpthandle.populate(skhandle.fgpt(sparse=True), skhandle.params, 1)
        print " check that the handler can recover the first one "
        # does it build a coherent histogram matrix
        assert not fgpt==skhandle.fgpt(sparse=True)
        
#        plt.figure()
#        fgpthandle.draw_fgpt(fgpt, skhandle.params)
#        plt.show()
        hist = fgpthandle.retrieve(fgpt, skhandle.params, nbCandidates=2)
        
        assert hist is not None
#        print hist.shape
        print "Score for first is %d Score for second is %d"%(np.max(hist[:,0]),
                                                              np.max(hist[:,1]))
        
        if display:
            plt.figure()
            from scipy.ndimage.filters import median_filter            
            plt.plot(median_filter(hist, (3, 1)))
            plt.title(fgpthandle.__class__)
            plt.show()
            
        print "Is the best candidate the good one?"
        estimated_index, estimated_offset  = fgpthandle.get_candidate(fgpt,skhandle.params,
                                                                    nbCandidates=2, smooth=3)
        print "Guessed %d with offset %1.1f s"%(estimated_index, estimated_offset)
        assert estimated_index == 0
        assert estimated_offset < 6.0
        print "OK"
    
    def runTest(self):        
        abstractFGPT = FgptHandle('abstract.db')
        
        self.assertRaises(NotImplementedError,abstractFGPT.add, None,0)
        self.assertRaises(NotImplementedError,abstractFGPT.retrieve, None, None)
        self.assertRaises(NotImplementedError,abstractFGPT.populate, None, None,0)
        self.assertRaises(NotImplementedError,abstractFGPT.get, None)

        # for all sketches, we performe the same testing
        Full = False
        display=False
        import time
        
        for (fgpthandle, skhandle) in fgpt_sketches:
            print "************************************"
            print fgpthandle
            print skhandle            
            tstart = time.time()    
            if Full:
                self.add_get_test(skhandle, fgpthandle, display=display)
            
            # Now the last of the test: populate a base of a few dozens musical samples
            self.populate_test(skhandle, fgpthandle)
            
            # and retrieve a segment in the base
            true_file_index = 3
            # This is on eof the hardest offset, its right in the middle of two learning frames
            true_offset = 7.5   
            
            # get the fingerprint for the test
            true_file_path = op.join(audio_files_path, file_names[true_file_index])
            true_l_sig = LongSignal(true_file_path,  frame_duration=true_offset)
            true_sig = true_l_sig.get_sub_signal(1, 1, mono=True, normalize=True)
            true_sig.crop(0, 5.0*true_sig.fs)
            skhandle.recompute(true_sig)
            skhandle.sparsify(150)    
            test_fgpt = skhandle.fgpt(sparse=False)
        
            hist =  fgpthandle.retrieve(test_fgpt, skhandle.params, nbCandidates=8)
            
            estimated_index, estimated_offset  = fgpthandle.get_candidate(test_fgpt,skhandle.params,
                                                                        nbCandidates=8, smooth=1)
        
            print "The System %s retrieved index %d (%d) at position %d (%d)"%(fgpthandle.__class__.__name__,
                                                                               estimated_index,
                                                                            true_file_index,
                                                                            estimated_offset,
                                                                            true_offset)
            
            print " It took %d seconds, DB contains %d keys and %d key/data pairs"%(
                                                                    time.time()-tstart,
                                                                     fgpthandle.get_stats()['nkeys'],
                                                                     fgpthandle.get_stats()['ndata'])
            if display:
                plt.figure()
                plt.plot(hist)
                plt.show()
            assert estimated_index == true_file_index
            assert np.abs(estimated_offset - true_offset) <= 5


if __name__ == "__main__":
    
    suite = unittest.TestSuite()

    suite.addTest(FgptTest())

    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()