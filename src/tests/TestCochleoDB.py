'''
tests.TestCochleoDB  -  Created on Jul 16, 2013
@author: M. Moussallam
'''
import sys
#sys.path.append('/home/manu/workspace/audio-sketch')
#sys.path.append('/home/manu/workspace/PyMP')

import unittest
import numpy as np
import bsddb.db as db
import matplotlib.pyplot as plt
from math import floor, ceil, log
import struct
import os
import os.path as op
from classes.pydb import STFTPeaksBDB , CochleoPeaksBDB
from PyMP import Signal 
from PyMP.signals import LongSignal
from classes import sketch
audio_files_path = '/sons/rwc/rwc-p-m07'
file_names = os.listdir(audio_files_path)


class DatabaseConstructionTest(unittest.TestCase):

    def runTest(self):
        ''' time to test the fingerprinting scheme, create a base with 200 atoms for 8 songs, then
            Construct the histograms and retrieve the fileIndex and time offset that is the most
            plausible '''
        print "------------------ Test5  DB construction ---------"
#        # create the base : persistent
        ppdb = CochleoPeaksBDB('LargeCochleoPeaksdb.db', load=False, time_res=0.2,wall=False)
        print ppdb
        
        segDuration = 5        
        
        sig = LongSignal(op.join(audio_files_path, file_names[0]),
                                 frame_duration=segDuration, mono=False, Noverlap=0)

        segmentLength = sig.segment_size
        max_seg_num = 5
#        " run sketchifier on a number of files"
        nbFiles = 8
        keycount = 0
        for fileIndex in range(nbFiles):
            RandomAudioFilePath = file_names[fileIndex]
            print fileIndex, RandomAudioFilePath
            if not (RandomAudioFilePath[-3:] == 'wav'):
                continue

            pySig = LongSignal(
                op.join(audio_files_path, RandomAudioFilePath),
                frame_size=segmentLength, mono=False, Noverlap=0)
            sk = sketch.CochleoPeaksSketch(**{'fs':pySig.fs,'step':128,'wall':False})
            nbSeg = int(pySig.n_seg)
            print 'Working on ' + str(RandomAudioFilePath) + ' with ' + str(nbSeg) + ' segments'
            for segIdx in range(min(nbSeg, max_seg_num)):
                pySigLocal = pySig.get_sub_signal(
                    segIdx, 1, True, True, channel=0, pad=0)
                print "sketchify the segment %d" % segIdx
                # run the decomposition
                
                
                sk.recompute(pySigLocal)
                sk.sparsify(200)
                fgpt = sk.fgpt()
                print "Populating database with offset " + str(segIdx * segmentLength / sig.fs)
                ppdb.populate(fgpt, sk.params, fileIndex)

#                keycount += approx.atom_number

        print ppdb.get_stats()

class OffsetDetectionTest(unittest.TestCase):
    ''' create a database from a single file, then try to recover the correct offset '''
    
    def runTest(self):
        ppdb = CochleoPeaksBDB('tempdb.db', load=False, persistent=True,
                               time_max=500.0,t_width=12,f_width=10,wall=False)        
        

        pySig = LongSignal(
                op.join(audio_files_path, file_names[0]),
                frame_duration=5, mono=False, Noverlap=0)

        self.assertEqual(pySig.segment_size, 5.0*pySig.fs)
        
        max_nb_seg = 15
        nb_atoms = 100
        sk = sketch.CochleoPeaksSketch(**{'fs':pySig.fs,'step':128,'downsample':8000})
        for segIdx in range(min(max_nb_seg,pySig.n_seg)):
            pySigLocal = pySig.get_sub_signal(
                segIdx, 1, mono=True, normalize=False, channel=0, pad=0)
            print "sketchify segment %d" % segIdx
            # run the decomposition
            
            
            sk.recompute(pySigLocal)
            sk.sparsify(nb_atoms)                          
            fgpt = sk.fgpt()
            
#            plt.figure()
#            plt.spy(fgpt[:,100:300])
#            plt.show()
            print "Populating database with offset " + str(segIdx * pySig.segment_size / pySig.fs)
            ppdb.populate(fgpt, sk.params, 0,
                      offset = segIdx * pySig.segment_size / pySig.fs)
    

        # ok we have a DB with only 1 file and different segments, now 
        nb_test_seg = 15
        long_sig_test = LongSignal(
                op.join(audio_files_path, file_names[0]),
                frame_duration=5, mono=False, Noverlap=0.5)
        count = 0
        for segIdx in range(min(nb_test_seg,long_sig_test.n_seg)):
            pySigLocal = long_sig_test.get_sub_signal(
                segIdx, 1, mono=True, normalize=False, channel=0, pad=0)
#            print "MP on segment %d" % segIdx
            # run the decomposition
            sk.recompute(pySigLocal)
            sk.sparsify(nb_atoms)
            fgpt = sk.fgpt()
            
            histograms = ppdb.retrieve(fgpt, sk.params,
                      offset=0, nbCandidates=1)
            
            
            maxI = np.argmax(histograms[:])
            OffsetI = maxI / 1
            estFileI = maxI % 1
            
            oracle_value = segIdx * long_sig_test.segment_size * (1 - long_sig_test.overlap) / long_sig_test.fs 
            print "Seg %d Oracle: %1.1f - found %1.1f"%(segIdx, oracle_value, OffsetI )
            if abs(OffsetI - oracle_value) < 5:
                count +=1 

                
        glob = float(count)/float(min(nb_test_seg,long_sig_test.n_seg))
        print "Global Score of %1.3f"%glob
        self.assertGreater(glob, 0.8)
#            
#class FileRecognitionTest(unittest.TestCase):
#
#    def runTest(self):
#        ''' take the base previously constructed and retrieve the song index based on 200 atoms/seconds
#        '''
#        print "------------------ Test6  recognition ---------"
#
#        nbCandidates = 8
#        ppdb = STFTPeaksBDB('LargeCochleodb.db', load=True)
#
#        print 'Large Db of ' + str(ppdb.get_stats()['nkeys']) + ' and ' + str(ppdb.get_stats()['ndata'])
#        # Now take a song, decompose it and try to retrieve it
#        fileIndex = 6
#        RandomAudioFilePath = file_names[fileIndex]
#        print 'Working on ' + str(RandomAudioFilePath)
#        pySig = Signal(op.join(audio_files_path, RandomAudioFilePath),
#                               mono=True)
#
#        pyDico = LODico(sizes)
#        segDuration = 5
#        offsetDuration = 7
#        offset = offsetDuration * pySig.fs
#        nbAtom = 50
#        segmentLength = ((segDuration * pySig.fs) / sizes[-1]) * sizes[-1]
#        pySig.crop(offset, offset + segmentLength)
#
#        approx, decay = mp.mp(pySig, pyDico, 40, nbAtom, pad=True)
#
##        plt.figure()
##        approx.plotTF()
##        plt.show()
#        res = map(ppdb.get , map(ppdb.kform, approx.atoms), [(a.time_position - pyDico.get_pad())  / approx.fs for a in approx.atoms])
##
#        #res = map(bdb.get, map(bdb.kform, approx.atoms))
#        
#        histogram = np.zeros((600, nbCandidates))
#        
#        for i in range(approx.atom_number):
#            print res[i]
#            histogram[res[i]] +=1
#        
#        max1 = np.argmax(histogram[:])
#        Offset1 = max1 / nbCandidates
#        estFile1 = max1 % nbCandidates
##        candidates , offsets = ppdb.retrieve(approx);
##        print approx.atom_number
#        histograms = ppdb.retrieve(approx, offset=0, nbCandidates=8)
## print histograms , np.max(histograms) , np.argmax(histograms, axis=0) ,
## np.argmax(histograms, axis=1)
#
##        plt.figure()
##        plt.imshow(histograms[0:20,:],interpolation='nearest')
##        plt.show()
#
#        maxI = np.argmax(histograms[:])
#        OffsetI = maxI / nbCandidates
#        estFileI = maxI % nbCandidates
#
#        print fileIndex, offsetDuration, estFileI,  OffsetI, estFile1,  Offset1, max1, maxI
#        import matplotlib.pyplot as plt
##        plt.figure(figsize=(12,6))
##        plt.subplot(121)
##        plt.imshow(histograms,aspect='auto',interpolation='nearest')
##        plt.subplot(122)
##        plt.imshow(histogram,aspect='auto',interpolation='nearest')
###        plt.imshow(histograms,aspect='auto',interpolation='nearest')
###        plt.colorbar()
##        plt.show()
#
#        print maxI, OffsetI, estFileI
#        self.assertEqual(histograms[OffsetI, estFileI], np.max(histograms))
#        
#        
#        self.assertEqual(fileIndex, estFileI)
#        self.assertTrue(abs(offsetDuration - OffsetI) <= 2.5)
#        estOffset = OffsetI;
#        print estOffset , estFileI
      
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCreateAndDestroyBase']

    suite = unittest.TestSuite()

#    suite.addTest(CreateAndDestroyBaseTest())
#    suite.addTest(HandlingMultipleKeyTest())
#    suite.addTest(DifferentFormattingTest())
#    suite.addTest(PPBSDHandlerTest())
#    suite.addTest(PopulatePeakPairTest())
#    suite.addTest(PersistentBaseCreationTest())
#    suite.addTest(DatabaseConstructionTest())
#    suite.addTest(FileRecognitionTest())
    suite.addTest(OffsetDetectionTest())

    unittest.TextTestRunner(verbosity=2).run(suite)
