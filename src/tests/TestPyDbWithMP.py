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
from classes.pydb import ppBDB

from PyMP import signals, mp
from PyMP.mdct import Dico, LODico
from PyMP.mdct.dico import SpreadDico

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

        print 'populating with a key from MP'
        F = 128
        fmax = 5500.0
        # max frequency: plain quantization
        F_N = 14
        # for quantization
        T = 125.0
        FileIdx = 104

        Nbits = 2 ** 16

        Tbin = floor(T / (10.0 * 60.0) * (Nbits - 1))
        # Coding over 16bits, max time offset = 10min

        B = int(FileIdx * Nbits + Tbin)

        # preparing the key/values
        beta = ceil((fmax) / (2.0 ** F_N - 1.0))

        K = int(floor(F) * 2 ** (F_N) + floor(float(F) / float(beta)))

        print B, beta, K, T, Tbin
        Bbin = struct.pack('<I4', B)
        Kbin = struct.pack('<I4', K)
        Db.put(Kbin, Bbin)

        result = Db.get(Kbin)
        Tres = struct.unpack('<I4', result)
        self.assertEqual(Tres[0], B)
        print 'retrieving the file idex and time position'
        estTimeI = B % Nbits
        estFileI = B / Nbits
        estTime = (float(estTimeI) / (Nbits - 1)) * (600)

        self.assertEqual(estTimeI, Tbin)
        self.assertEqual(estFileI, FileIdx)
        self.assertTrue(abs(estTime - T) < 0.01)

        Db.close()


class PPBSDHandlerTest(unittest.TestCase):
    def runTest(self):
        print "------------------ Test2  DB Handle ---------"
        print 'Creating a hash table using Berkeley DB'
        dbName = 'dummy.db'

        ppbdb = ppBDB(dbName, load=False, persistent=False)
        ppbdb.keyformat = None
        print 'populating with a key from MP'
        F = 128
        T = 125.0
        FileIdx = 104

        ppbdb.add(zip((F,), (T,)), FileIdx)
        estT, estFileI = ppbdb.get(F)

        # refactoring, renvoi une liste de candidats

        self.assertEqual(estFileI[0], FileIdx)
        self.assertTrue(abs(estT[0] - T) < 0.01)

        print ppbdb.get_stats()
        print ppbdb
        del ppbdb
#


class PopulateMPAtomsTest(unittest.TestCase):

    def runTest(self):
        print "------------------ Test3  Populate from MP coeffs ---------"
        fileIndex = 2
        RandomAudioFilePath = file_names[fileIndex]
        print 'Working on %s' % RandomAudioFilePath
        sizes = [2 ** j for j in range(7, 15)]
        segDuration = 5
        nbAtom = 20

        pySig = signals.Signal(op.join(audio_files_path, RandomAudioFilePath),
                               mono=True, normalize=True)

        segmentLength = ((segDuration * pySig.fs) / sizes[-1]) * sizes[-1]
        nbSeg = floor(pySig.length / segmentLength)
        # cropping
        pySig.crop(0, segmentLength)

        # create dictionary
        pyDico = Dico(sizes)

        approx, decay = mp.mp(pySig, pyDico, 20, nbAtom, pad=True, debug=0)

        ppdb = ppBDB('MPdb.db', load=False)
#        ppdb.keyformat = None
        ppdb.populate(approx, fileIndex)

        nKeys = ppdb.get_stats()['ndata']
        # compare the number of keys in the base to the number of atoms

        print ppdb.get_stats()
        self.assertEqual(nKeys, approx.atom_number)

        # now try to recover the fileIndex knowing one of the atoms
        Key = [log(approx.atoms[0].length, 2), approx.atoms[0]
               .reduced_frequency * pySig.fs]
        T, fileI = ppdb.get(Key)
        Treal = (float(approx.atoms[0].time_position) / float(pySig.fs))
        print T, Treal
        self.assertEqual(fileI[0], fileIndex)
        Tpy = np.array(T)
        self.assertTrue((np.abs(Tpy - Treal)).min() < 0.1)

        # last check: what does a request for non-existing atom in base return?
        T, fileI = ppdb.get((11, 120.0))
        self.assertEqual(T, [])
        self.assertEqual(fileI, [])

        # now let's just retrieve the atoms from the base and see if they are
        # the same
        histograms = ppdb.retrieve(approx, offset=0)
#        plt.figure()
#        plt.imshow(histograms[0:10,:])
#        plt.show()
        del ppdb
#


class PersistentBaseCreationTest(unittest.TestCase):

    def runTest(self):
        print "------------------ Test4  DB persistence ---------"

        ppdb = ppBDB('NonPersistentMPdb.db', persistent=False)
        self.assertTrue(os.path.exists('./NonPersistentMPdb.db'))
        del ppdb
        self.assertFalse(os.path.exists('./NonPersistentMPdb.db'))

        ppdb = ppBDB('PersistentMPdb.db', persistent=True)
        self.assertTrue(os.path.exists('./PersistentMPdb.db'))
        del ppdb
        self.assertTrue(os.path.exists('./PersistentMPdb.db'))

        # now add something in the base
        ppdb = ppBDB('./PersistentMPdb.db', load=True)
        ppdb.keyformat = None
        self.assertTrue(ppdb.persistent)
        ppdb.add(zip((440,), (1.45,)), 4)

        # delete the base and reload it
        del ppdb

        ppdb = ppBDB('./PersistentMPdb.db', load=True)
        ppdb.keyformat = None
        T, fi = ppdb.get(440)
        self.assertTrue(abs(T[0] - 1.45) < 0.01)
        self.assertEqual(fi[0], 4)
#
#


class DatabaseConstructionTest(unittest.TestCase):

    def runTest(self):
        ''' time to test the fingerprinting scheme, create a base with 10 atoms for 8 songs, then
            Construct the histograms and retrieve the fileIndex and time offset that is the most
            plausible '''
        print "------------------ Test5  DB construction ---------"
#        # create the base : persistent
        ppdb = ppBDB('LargeMPdb.db', load=False, time_res=0.2)
        print ppdb
        padZ = 2 * sizes[-1]
        # BUGFIX: pour le cas MP classique: certains atome reviennent : pas
        # cool car paire key/data existe deja!
        pyDico = LODico(sizes)
        segDuration = 5
        nbAtom = 50
        sig = signals.LongSignal(op.join(audio_files_path, file_names[0]),
                                 frame_size=sizes[-1], mono=False, Noverlap=0)

        segmentLength = ((segDuration * sig.fs) / sizes[-1]) * sizes[-1]
        max_seg_num = 5
#        " run MP on a number of files"
        nbFiles = 8
        keycount = 0
        for fileIndex in range(nbFiles):
            RandomAudioFilePath = file_names[fileIndex]
            print fileIndex, RandomAudioFilePath
            if not (RandomAudioFilePath[-3:] == 'wav'):
                continue

            pySig = signals.LongSignal(
                op.join(audio_files_path, RandomAudioFilePath),
                frame_size=segmentLength, mono=False, Noverlap=0)

            nbSeg = int(pySig.n_seg)
            print 'Working on ' + str(RandomAudioFilePath) + ' with ' + str(nbSeg) + ' segments'
            for segIdx in range(min(nbSeg, max_seg_num)):
                pySigLocal = pySig.get_sub_signal(
                    segIdx, 1, True, True, channel=0, pad=padZ)
                print "MP on segment %d" % segIdx
                # run the decomposition
                approx, decay = mp.mp(
                    pySigLocal, pyDico, 40, nbAtom, pad=False)

                print "Populating database with offset " + str(segIdx * segmentLength / sig.fs)
                ppdb.populate(
                    approx, fileIndex, offset=(segIdx * segmentLength) - padZ)

                keycount += approx.atom_number

        print ppdb.get_stats()
#        self.assertEqual(keycount, ppdb.get_stats()['ndata'])

#

class OffsetDetectionTest(unittest.TestCase):
    ''' create a database from a single file, then try to recover the correct offset '''
    
    def runTest(self):
        ppdb = ppBDB('tempdb.db', load=False, persistent=True, maxOffset=500.0)        
        
        pySig = signals.LongSignal(
                op.join(audio_files_path, file_names[0]),
                frame_duration=5, mono=False, Noverlap=0)

        self.assertEqual(pySig.segment_size, 5.0*pySig.fs)
        
        max_nb_seg = 10;
        nb_atoms = 150;
        
        scales = SpreadDico([8192], penalty=0.1, mask_time=2, mask_freq=20)
        
#        scales = Dico([8192])
        for segIdx in range(min(max_nb_seg,pySig.n_seg)):
            pySigLocal = pySig.get_sub_signal(
                segIdx, 1, mono=True, normalize=False, channel=0, pad=scales.get_pad())
            print "MP on segment %d" % segIdx
            # run the decomposition
            approx, decay = mp.mp(
                pySigLocal, scales, 2, nb_atoms, pad=False)

            print "Populating database with offset " + str(segIdx * pySig.segment_size / pySig.fs)
            ppdb.populate(
                approx, 0, offset=(segIdx * pySig.segment_size) - scales.get_pad())
    

        # ok we have a DB with only 1 file and different segments, now 
        nb_test_seg = 15
        long_sig_test = signals.LongSignal(
                op.join(audio_files_path, file_names[0]),
                frame_duration=5, mono=False, Noverlap=0.5)
        count = 0
        for segIdx in range(min(nb_test_seg,long_sig_test.n_seg)):
            pySigLocal = long_sig_test.get_sub_signal(
                segIdx, 1, mono=True, normalize=False, channel=0, pad=scales.get_pad())
#            print "MP on segment %d" % segIdx
            # run the decomposition
            approx, decay = mp.mp(
                pySigLocal, scales, 2, nb_atoms, pad=False)
            print approx.atom_number
            
            histograms = ppdb.retrieve(approx, nbCandidates=1)
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
            
class FileRecognitionTest(unittest.TestCase):

    def runTest(self):
        ''' take the base previously constructed and retrieve the song index based on 10 atoms
        '''
        print "------------------ Test6  recognition ---------"

        nbCandidates = 8
        ppdb = ppBDB('LargeMPdb.db', load=True)

        print 'Large Db of ' + str(ppdb.get_stats()['nkeys']) + ' and ' + str(ppdb.get_stats()['ndata'])
        # Now take a song, decompose it and try to retrieve it
        fileIndex = 6
        RandomAudioFilePath = file_names[fileIndex]
        print 'Working on ' + str(RandomAudioFilePath)
        pySig = signals.Signal(op.join(audio_files_path, RandomAudioFilePath),
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


class HandlingMultipleKeyTest(unittest.TestCase):

    def runTest(self):
        ''' what happens when the same key is used with a different value? we should be able to retrieve
        a list of possible songs
        '''
        print "------------------ Test 7  handling uniqueness ---------"
        dummyDb = ppBDB('dummyDb.db', load=False, persistent=False)
        dummyDb.keyformat = None
        dummyDb.add(zip((440, 128, 440), (12.0, 16.0, 45.0)), 1)
        dummyDb.add(zip((512, 320, 440), (54.0, 16.0, 11.0)), 2)

#        print dummyDb.get_stats()
        self.assertEqual(6, dummyDb.get_stats()['ndata'])

        T, fileIndex = dummyDb.get(440)
        print zip(T, fileIndex)
#


class DifferentFormattingTest(unittest.TestCase):

    def runTest(self):
        print "------------------ Test 8  different formattings ---------"
        dbFormat1 = ppBDB('format1.db', load=False, persistent=False)
        dbFormat0 = ppBDB('format2.db', load=False, persistent=False)
        dbFormat1.keyformat = None
        key1 = [0, 12]
        data = 13.0

        dbFormat1.add(zip((key1[1],), (data,)), 2)
        dbFormat0.add(zip((key1,), (data,)), 2)

        T, fi = dbFormat0.get(key1)
        print fi


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCreateAndDestroyBase']

    suite = unittest.TestSuite()

#    suite.addTest(CreateAndDestroyBaseTest())
#    suite.addTest(HandlingMultipleKeyTest())
#    suite.addTest(DifferentFormattingTest())
#    suite.addTest(PPBSDHandlerTest())
#    suite.addTest(PopulateMPAtomsTest())
#    suite.addTest(PersistentBaseCreationTest())
#    suite.addTest(DatabaseConstructionTest())
#    suite.addTest(FileRecognitionTest())
    suite.addTest(OffsetDetectionTest())

    unittest.TextTestRunner(verbosity=2).run(suite)
