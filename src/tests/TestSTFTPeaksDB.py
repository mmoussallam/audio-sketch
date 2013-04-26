'''
tests.TestSTFTPeaksDB  -  Created on Apr 24, 2013
@author: M. Moussallam
'''
'''
Created on Sep 7, 2011

@author: moussall
'''
import sys
#sys.path.append('/home/manu/workspace/audio-sketch')
#sys.path.append('/home/manu/workspace/PyMP')

import unittest
import numpy as np
import bsddb.db as db
from math import floor, ceil, log
import struct
import os
import os.path as op
from classes.pydb import STFTPeaksBDB 
from PyMP import Signal 
from PyMP.signals import LongSignal
from classes import sketch
audio_files_path = '/sons/rwc/rwc-p-m07'
file_names = os.listdir(audio_files_path)
sizes = [128, 1024, 8192]


class CreateAndDestroyBaseTest(unittest.TestCase):

    def runTest(self):
        print "------------------ Test1  DB Creation and destruction ---------"
        print 'Creating a hash table using Berkeley DB'
        dbName = 'dummy.db'
        if os.path.exists(dbName):
            os.remove(dbName)
        
        Db = db.DB()
        Db.open(dbName, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        
        print 'Closing and reopening the database'
        Db.close()
        
        Db = db.DB()
        Db.open(dbName, dbtype=db.DB_HASH)
        
        print 'populating with a key from a pair of TF pics'
        
        # anchor in time/frequency
        (t1, f1) = (155.68, 442.1)
        (t2, f2) = (157.12, 512.4)
        delta_t_max = 3.0 # in seconds
                
        fmax = 8000.0 # in hertz
        FileIdx = 104
        
        # Decide the total size of a key is 16 bits and value is 32 bits
        key_total_nbits = 32
        f1_n_bits = 10
        f2_n_bits = 10
        dt_n_bits = 10
        
        value_total_bits = 32
        file_index_n_bits = 20
        time_n_bits = value_total_bits - file_index_n_bits
        time_max = 60.0* 20.0
        
        Bin_value = floor(((t1 / time_max)*(2**time_n_bits -1))+ FileIdx * (2**file_index_n_bits))
             
        # Formatting the key - FORMAT 1 : Absolute
        alpha = ceil((fmax)/(2**f1_n_bits-1))
        beta = ceil((fmax)/(2**f2_n_bits-1))
        gamma = ceil(delta_t_max/(2**dt_n_bits-1))
        
        Bin_key = floor((f1/alpha)*2**(f2_n_bits+dt_n_bits)) + \
                floor( (f2/beta)*2**(dt_n_bits)) + \
                floor((t2 - t1)/gamma)
        
        # To retrieve each element
        Bbin = struct.pack('<I4', Bin_value)
        Kbin = struct.pack('<I4', Bin_key)
        Db.put(Kbin, Bbin)
        
        result = Db.get(Kbin)
        Tres = struct.unpack('<I4', result)
        self.assertEqual(Tres[0], Bin_value)
        
        print 'retrieving the file idex and time position'
        # to retrieve the file_index and time :
        songID = floor(Tres[0]/2**file_index_n_bits)   
        self.assertEqual(songID, FileIdx)
        
        # and quantized time
        timeofocc = Tres[0]-songID*(2**file_index_n_bits)        
        timeofocc = float(timeofocc)/(2**time_n_bits-1)*time_max 
        
        self.assertEqual(floor(t1), floor(timeofocc))   


        Db.close()


class PPBSDHandlerTest(unittest.TestCase):
    def runTest(self):
        print "------------------ Test2  DB Handle ---------"
        print 'Creating a hash table using Berkeley DB'
        dbName = 'dummy.db'

        ppbdb = STFTPeaksBDB(dbName, load=False, persistent=False)
        ppbdb.keyformat = None
        print 'populating with a key as a pair of peaks'
        (t1, f1) = (155.68, 442.1)
        (t2, f2) = (157.12, 512.4)
        FileIdx = 104

        key = (f1,f2,t2-t1)
        value = t1
        ppbdb.add(zip( (key,), (value,) ), FileIdx)
        estT, estFileI = ppbdb.get(key)

        # refactoring, renvoi une liste de candidats

        self.assertEqual(estFileI[0], FileIdx)
        print estT[0], t1
        self.assertTrue(abs(estT[0] - t1) < 1)

        print ppbdb.get_stats()
        print ppbdb
        del ppbdb
#


class PopulatePeakPairTest(unittest.TestCase):

    def runTest(self):
        print "------------------ Test3  Populate from a true pair of peaks ---------"
        fileIndex = 2
        RandomAudioFilePath = file_names[fileIndex]
        print 'Working on %s' % RandomAudioFilePath
        sizes = [2 ** j for j in range(7, 15)]
        segDuration = 5
        nbAtom = 20
        
        pySig = Signal(op.join(audio_files_path, RandomAudioFilePath),
                               mono=True, normalize=True)
        
        segmentLength = ((segDuration * pySig.fs) / sizes[-1]) * sizes[-1]
        nbSeg = floor(pySig.length / segmentLength)
        # cropping
        pySig.crop(0, segmentLength)
        
        # create the sparsified matrix of peaks
        # the easiest is to use the existing PeakPicking in sketch
        from classes import sketch
        sk = sketch.STFTPeaksSketch()
        sk.recompute(pySig)
        sk.sparsify(100)
        fgpt = sk.fgpt(sparse=True)
        ppdb = STFTPeaksBDB('STFTPeaksdb.db', load=False)
#        ppdb.keyformat = None

        # compute the pairs of peaks
        peak_indexes = np.nonzero(fgpt[0, : , :])
        # Take one peak
        peak_ind = (peak_indexes[0][2],peak_indexes[1][2]) 
        f_target_width = 2*sk.params['f_width']
        t_target_width = 2*sk.params['t_width']
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.log(np.abs(fgpt[0,
                peak_ind[0]: peak_ind[0]+f_target_width,
                peak_ind[1]: peak_ind[1]+t_target_width])))
        
        
        target_points_i, target_points_j = np.nonzero(fgpt[0,
                                                        peak_ind[0]: peak_ind[0]+f_target_width,
                                                        peak_ind[1]: peak_ind[1]+t_target_width])
        # now we can build a pair of peaks , and thus a key
        f1 = (float(peak_ind[0]) / sk.params['scale'])*pySig.fs
        f2 = (float(peak_ind[0]+target_points_i[1]) / sk.params['scale'])*pySig.fs
        delta_t = float(target_points_j[1] * sk.params['step'])/float(pySig.fs)
        t1 = float(peak_ind[1] * sk.params['step'])/float(pySig.fs)
        key = (f1, f2, delta_t)
        print key, t1
        ppdb.populate(sk.fgpt(), sk.params, fileIndex)

        nKeys = ppdb.get_stats()['ndata']
        # compare the number of keys in the base to the number of atoms

#        print ppdb.get_stats()
        self.assertEqual(nKeys, 116)

        # now try to recover the fileIndex knowing one key
        T, fileI = ppdb.get(key)
        
        
        self.assertEqual(fileI[0], fileIndex)
        Tpy = np.array(T)
        print Tpy
        self.assertTrue((np.abs(Tpy - t1)).min() < 0.5)

        # last check: what does a request for non-existing atom in base return?
        T, fileI = ppdb.get((11, 120.0, 0.87))
        self.assertEqual(T, [])
        self.assertEqual(fileI, [])

        # now let's just retrieve the atoms from the base and see if they are
        # the same
        histograms = ppdb.retrieve(fgpt, sk.params)
#        plt.figure()
#        plt.imshow(histograms[0:10,:])
#        plt.show()
        del ppdb
#

#
#
class DatabaseConstructionTest(unittest.TestCase):

    def runTest(self):
        ''' time to test the fingerprinting scheme, create a base with 200 atoms for 8 songs, then
            Construct the histograms and retrieve the fileIndex and time offset that is the most
            plausible '''
        print "------------------ Test5  DB construction ---------"
#        # create the base : persistent
        ppdb = STFTPeaksBDB('LargeSTFTPeaksdb.db', load=False, time_res=0.2)
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
            sk = sketch.STFTPeaksSketch(**{'scale':512,'step':128})
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
#        self.assertEqual(keycount, ppdb.get_stats()['ndata'])
#
##
#
class OffsetDetectionTest(unittest.TestCase):
    ''' create a database from a single file, then try to recover the correct offset '''
    
    def runTest(self):
        ppdb = STFTPeaksBDB('tempdb.db', load=False, persistent=True, time_max=500.0)        
        

        pySig = LongSignal(
                op.join(audio_files_path, file_names[0]),
                frame_duration=5, mono=False, Noverlap=0)

        self.assertEqual(pySig.segment_size, 5.0*pySig.fs)
        
        max_nb_seg = 10
        nb_atoms = 400
        sk = sketch.STFTPeaksSketch(**{'scale':512,'step':128})
        for segIdx in range(min(max_nb_seg,pySig.n_seg)):
            pySigLocal = pySig.get_sub_signal(
                segIdx, 1, mono=True, normalize=False, channel=0, pad=0)
            print "sketchify segment %d" % segIdx
            # run the decomposition
            
            
            sk.recompute(pySigLocal)
            sk.sparsify(nb_atoms)
            fgpt = sk.fgpt()

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
class FileRecognitionTest(unittest.TestCase):

    def runTest(self):
        ''' take the base previously constructed and retrieve the song index based on 200 atoms/seconds
        '''
        print "------------------ Test6  recognition ---------"

        nbCandidates = 8
        ppdb = STFTPeaksBDB('LargeSTFTdb.db', load=True)

        print 'Large Db of ' + str(ppdb.get_stats()['nkeys']) + ' and ' + str(ppdb.get_stats()['ndata'])
        # Now take a song, decompose it and try to retrieve it
        fileIndex = 6
        RandomAudioFilePath = file_names[fileIndex]
        print 'Working on ' + str(RandomAudioFilePath)
        pySig = Signal(op.join(audio_files_path, RandomAudioFilePath),
                               mono=True)

        pyDico = LODico(sizes)
        segDuration = 5
        offsetDuration = 7
        offset = offsetDuration * pySig.fs
        nbAtom = 50
        segmentLength = ((segDuration * pySig.fs) / sizes[-1]) * sizes[-1]
        pySig.crop(offset, offset + segmentLength)

        approx, decay = mp.mp(pySig, pyDico, 40, nbAtom, pad=True)

#        plt.figure()
#        approx.plotTF()
#        plt.show()
        res = map(ppdb.get , map(ppdb.kform, approx.atoms), [(a.time_position - pyDico.get_pad())  / approx.fs for a in approx.atoms])
#
        #res = map(bdb.get, map(bdb.kform, approx.atoms))
        
        histogram = np.zeros((600, nbCandidates))
        
        for i in range(approx.atom_number):
            print res[i]
            histogram[res[i]] +=1
        
        max1 = np.argmax(histogram[:])
        Offset1 = max1 / nbCandidates
        estFile1 = max1 % nbCandidates
#        candidates , offsets = ppdb.retrieve(approx);
#        print approx.atom_number
        histograms = ppdb.retrieve(approx, offset=0, nbCandidates=8)
# print histograms , np.max(histograms) , np.argmax(histograms, axis=0) ,
# np.argmax(histograms, axis=1)

#        plt.figure()
#        plt.imshow(histograms[0:20,:],interpolation='nearest')
#        plt.show()

        maxI = np.argmax(histograms[:])
        OffsetI = maxI / nbCandidates
        estFileI = maxI % nbCandidates

        print fileIndex, offsetDuration, estFileI,  OffsetI, estFile1,  Offset1, max1, maxI
        import matplotlib.pyplot as plt
#        plt.figure(figsize=(12,6))
#        plt.subplot(121)
#        plt.imshow(histograms,aspect='auto',interpolation='nearest')
#        plt.subplot(122)
#        plt.imshow(histogram,aspect='auto',interpolation='nearest')
##        plt.imshow(histograms,aspect='auto',interpolation='nearest')
##        plt.colorbar()
#        plt.show()

        print maxI, OffsetI, estFileI
        self.assertEqual(histograms[OffsetI, estFileI], np.max(histograms))
        
        
        self.assertEqual(fileIndex, estFileI)
        self.assertTrue(abs(offsetDuration - OffsetI) <= 2.5)
#        estOffset = OffsetI;
#        print estOffset , estFileI
        
#
#
#
#class HandlingMultipleKeyTest(unittest.TestCase):
#
#    def runTest(self):
#        ''' what happens when the same key is used with a different value? we should be able to retrieve
#        a list of possible songs
#        '''
#        print "------------------ Test 7  handling uniqueness ---------"
#        dummyDb = XMDCTBDB('dummyDb.db', load=False, persistent=False)
#        dummyDb.keyformat = None
#        dummyDb.add(zip((440, 128, 440), (12.0, 16.0, 45.0)), 1)
#        dummyDb.add(zip((512, 320, 440), (54.0, 16.0, 11.0)), 2)
#
##        print dummyDb.get_stats()
#        self.assertEqual(6, dummyDb.get_stats()['ndata'])
#
#        T, fileIndex = dummyDb.get(440)
#        print zip(T, fileIndex)
##
#
#
#class DifferentFormattingTest(unittest.TestCase):
#
#    def runTest(self):
#        print "------------------ Test 8  different formattings ---------"
#        dbFormat1 = XMDCTBDB('format1.db', load=False, persistent=False)
#        dbFormat0 = XMDCTBDB('format2.db', load=False, persistent=False)
#        dbFormat1.keyformat = None
#        key1 = [0, 12]
#        data = 13.0
#
#        dbFormat1.add(zip((key1[1],), (data,)), 2)
#        dbFormat0.add(zip((key1,), (data,)), 2)
#
#        T, fi = dbFormat0.get(key1)
#        print fi


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCreateAndDestroyBase']

    suite = unittest.TestSuite()

    suite.addTest(CreateAndDestroyBaseTest())
#    suite.addTest(HandlingMultipleKeyTest())
#    suite.addTest(DifferentFormattingTest())
    suite.addTest(PPBSDHandlerTest())
    suite.addTest(PopulatePeakPairTest())
#    suite.addTest(PersistentBaseCreationTest())
    suite.addTest(DatabaseConstructionTest())
#    suite.addTest(FileRecognitionTest())
    suite.addTest(OffsetDetectionTest())

    unittest.TextTestRunner(verbosity=2).run(suite)
