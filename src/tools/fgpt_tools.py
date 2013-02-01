'''
Created on Sep 14, 2011

Tools for fgpt experiments

@author: moussall
'''
import pydb
import time
import os
from PyMP import signals, mp
import numpy as np
from scipy.io import savemat, loadmat
audio_files_path = '/sons/rwc/rwc-g-m01/'
default_db_path = '/home/manu/workspace/fgpt_sparse/db/'

def fgpt_expe(file_names, n_atoms, dico,
              db_name=None,
              create_base=True,
              test_base=True,
              n_test_files=None,
              n_test_atoms=None,
              test_file_names=None,
              seg_duration=5.0,
              learn_step=5.0,
              test_step=2.5,
              hierarchical=False,
              threshold=None,
              n_atom_step=None):
    ''' Full range of experiments '''
    score = None

    if db_name is None:
        db_name = '%sMPdb_%dfiles_%datoms_%dx%s.db' %(default_db_path,
                                                     len(file_names),
                                                           n_atoms,
                                                           len(dico.sizes),
                                                           dico.nature)

    # load or create the base
    ppdb = pydb.ppBDB(db_name, load=(not create_base), persistent=True)

    if create_base:
        print 'Reconstructing The base'
        db_creation(ppdb, file_names, n_atoms, dico,
                   db_name=db_name, seg_duration=seg_duration, step=learn_step)

    if test_base:
        testNatom = n_atoms
        testFiles = len(file_names)

        if test_file_names is not None:
            file_names = test_file_names
        if n_test_atoms is not None:
            testNatom = n_test_atoms
        if n_test_files is not None:
            testFiles = n_test_files

        t_start = time.time()
        if not hierarchical:
            score = db_test(ppdb, file_names, n_atoms, dico,
                           test_n_atom=testNatom, test_files=testFiles,
                           seg_duration=seg_duration, step=test_step)
        else:
            if threshold is None or n_atom_step is None:
                raise ValueError('Not enough arguments provided for Hierarchical pruning!!')

            score = db_hierarchical_test(
                ppdb, file_names, n_atoms, dico, n_test_files,
                testNatom=testNatom, testFiles=testFiles,
                seg_duration=seg_duration, step=test_step,
                threshold=threshold, nbAtomPerIter=n_atom_step)

    return score, ppdb, time.time() - t_start


def db_creation(ppdb, file_names, n_atoms, dico,
               db_name=None, 
               seg_duration=5.0, 
               padZ=None, 
               step=None,
               files_path=None):
    ''' method to create a DB with the given dictionary and parameters 
        can take a pydb object or a string with the db name to be created'''
    n_files = len(file_names)
    sizes = dico.sizes
    
    if files_path is None:
        files_path = audio_files_path
    
    if not isinstance(ppdb, pydb.ppBDB):
        
        if db_name is None:
            db_name = '../data/MPdb_%dfiles_%datoms_%dxMDCT_%s.db' %( 
                                                           n_files,
                                                           n_atoms,
                                                           len(sizes),
                                                           dico.nature)

        # create the base
        ppdb = pydb.ppBDB(db_name, load=False)
        ppdb.keyformat = 0

    if padZ is None:
        padZ = 2 * sizes[-1]

    t0 = time.time()
    for fileIndex in range(n_files):
        
        l_sig, segPad = get_rnd_file(file_names,
                                       seg_duration,
                                       step,
                                       sizes,
                                       fileIndex)

        
        for segIdx in range(l_sig.n_seg):
            pySigLocal = l_sig.get_sub_signal(segIdx,
                                                 1,
                                                 mono=True,
                                                 normalize=True, 
                                                 pad=2 * sizes[-1])
            # run the decomposition
            try:
                approx = mp.mp(pySigLocal,
                               dico, 20, n_atoms,
                               pad=False,
                               silent_fail=True)[0]
            except ValueError:
                outPath = '%s/../fails/%s_seg_%d.wav'%(default_db_path,
                                                       file_names[fileIndex][:-4],
                                                       segIdx)
                pySigLocal.write(outPath)

            ppdb.populate(approx, fileIndex, offset=segIdx *
                          segPad, largebases=True)
        estTime = (float(
            (time.time() - t0)) / float(fileIndex + 1)) * (n_files - fileIndex)
        print 'Elapsed ' + str(time.time() - t0) + ' sec . Estimated : ' + str(estTime / 60) + ' minutes'



def get_rnd_file(file_names, seg_duration, step, sizes, fileIndex, n_files=None):
    """ returns a LongSignal object allowing fast disk access """
    
    if n_files is None:
        n_files = len(file_names)
    
    RandomAudioFilePath = file_names[fileIndex]
    
    sig = signals.LongSignal(audio_files_path + RandomAudioFilePath,
                             frame_duration = seg_duration, 
                             mono = True,
                             Noverlap = (1.0 - float(step)/float(seg_duration)))
    
#    pySig = signals.Signal(
#        audio_files_path + RandomAudioFilePath, mono=True, normalize=True)
#    
#    segmentLength = ((seg_duration * pySig.fs) / sizes[-1]) * sizes[-1]
    seg_pad = step * sig.fs
#    nbSeg = int(pySig.length / segPad - 1)
##        print pySig.fs , segmentLength , nbSeg
#        
    deb_str = 'Working on %s (%d/%d) with %d segments ' % (RandomAudioFilePath,
                                                               fileIndex+1,
                                                               n_files,
                                                               sig.n_seg )
    print deb_str
    return sig, seg_pad

def db_test(ppdb,
            file_names,
            n_atoms,
            dico,
            test_n_atom=None,
            test_files=None,
            seg_duration=5,
            step=2.5,
            shuffle=True):
    ''' Lets try to identify random segments from the files using the pre-calculated database
    '''
    if test_n_atom is not None:
        nbAtoms = test_n_atom

    n_files = len(file_names)
    sizes = dico.sizes
    " change the order of the files for testing"
    sortedIndexes = range(n_files)
    if shuffle:
        np.random.shuffle(sortedIndexes)

    if test_files is not None:
        sortedIndexes = sortedIndexes[:test_files]

    countok = 0.0
    countall = 0.0
    t0 = time.time()
    i = 0
    for fileIndex in sortedIndexes:
        i +=1
        long_sig, seg_pad = get_rnd_file(file_names,
                               seg_duration,
                               step,
                               sizes,
                               fileIndex,
                               n_files=i)
        
        for segIdx in range(int(long_sig.n_seg)):
            countall += 1.0
#            print segIdx
            pySigLocal = long_sig.get_sub_signal(segIdx,
                                                 1,
                                                 mono=True,
                                                 normalize=True, 
                                                 pad=2 * sizes[-1])

#            print "MP on segment %d"%segIdx
            # run the decomposition
            approx, decay = mp.mp(
                pySigLocal, dico, 20, nbAtoms, pad=False, silent_fail=True)

# print "Populating database with offset "
# +str(segIdx*segmentLength/11025)
            histograms = ppdb.retrieve(approx, nbCandidates=n_files)

            # first retrieve the most credible song
            maxI = np.argmax(histograms[:])
            OffsetI = maxI / n_files
            estFileI = maxI % n_files
#            print estFileI , OffsetI

#            # Rank the songs and calculate a distance
#            maxima = np.amax(histograms,axis=0);
#            time_shifts = np.argmax(histograms,axis=0);
#            estFileI = maxima.argmax()
#
#            ranked = np.sort(maxima);
#            OffsetI = time_shifts[estFileI]
#            distances = ranked[2:]-ranked[1:-1];
#
#
            if (fileIndex == estFileI):
                countok += 1.0
#                print "Correct answer, file " + str(fileIndex) + " with offset " +\
#                         str(OffsetI) + " ref : " + str(segIdx*step) #+ ' distance :' + str(distances[-1])
            else:
                print " Wrong answer " + str(estFileI)+ " with offset " + str(OffsetI) +\
                 " instead of " + str(fileIndex)   + " ref : " + str(segIdx*step)
#                print estFileI , fileIndex
        estTime = (float(
            (time.time() - t0)) / float(fileIndex + 1)) * (n_files - fileIndex)
        print 'Elapsed ' + str(time.time() - t0) + ' sec . Estimated : ' + str(estTime / 60) + ' minutes'

        print "Global Score of " + str(countok / countall)
    return countok / countall

# ppdb, file_names , nbAtoms, dico , test_n_atom=None , test_files = None,
# seg_duration=5, step = 2.5):


def db_hierarchical_test(ppdb,
                         file_names, 
                         n_atoms,
                         dico,
                         nbFiles,
                       testNatom=None, testFiles=None, seg_duration=5,
                       step=2.5, threshold=0.4, nbAtomPerIter=10, debug=True):
    ''' Same as db_test, except that the search is hierarchically pruned if distance between best candidate
    and second best one is above a pre-defined threshold
    '''
    sizes = dico.sizes

    if testNatom is not None:
        n_atoms = testNatom

    " change the order of the files for testing"
    sortedIndexes = range(nbFiles)
    np.random.shuffle(sortedIndexes)

    if testFiles is not None:
        sortedIndexes = sortedIndexes[:testFiles]

    countok = 0.0
    countall = 0.0
#    OkDistances = [];
#    WrongDistances = [];

    for fileIndex in sortedIndexes:
        nbSeg, pySig, segPad, segmentLength = get_rnd_file(file_names,
                                                           seg_duration,
                                                           step,
                                                           sizes,
                                                           fileIndex)
        for segIdx in range(nbSeg):
            countall += 1.0
            pySigLocal = pySig.copy()
            pySigLocal.crop(segIdx * segPad, segIdx * segPad + segmentLength)
            pySigLocal.pad(2 * sizes[-1])
#            print "MP on segment %d"%segIdx
            # run the decomposition
            approx, decay = mp.mp(
                pySigLocal, dico, 20, nbAtomPerIter, pad=False)
            histograms = ppdb.retrieve(approx, nbCandidates=nbFiles)

            condition = True
            bestGuess = -1
            bestScore = 0
            while condition:
                maxima = np.amax(histograms, axis=0)
                time_shifts = np.argmax(histograms, axis=0)
                estFileI = maxima.argmax()

                ranked = np.sort(maxima)
                OffsetI = time_shifts[estFileI]
                distances = ranked[1:] - ranked[:-1]

                if ranked[-1] == .0:
                    raise ValueError('Why the fuck am I here?')

                score = distances[-1] / ranked[-1]
                if score > bestScore:
                    bestScore = score
                    bestGuess = estFileI
                if score > threshold:
                    if debug:
                        print " We are above Threshold %.2f-  Stopping. Answer is " % score, str(estFileI == fileIndex)
                    condition = False
                else:
                    if debug:
                        print " We are under threshold. %.2f-  Let'sizes go further , best Guess is :" % score, bestGuess
                    approx, decay = mp.mp_continue(approx, pySigLocal, dico, 20, nbAtomPerIter, pad=False, debug=0)
                    histograms = ppdb.retrieve(approx, nbCandidates=nbFiles)
                    condition = (approx.atomNumber < n_atoms)

            if (fileIndex == estFileI):
                countok += 1.0
#                print "Correct answer, file " + str(fileIndex) + " with offset " + \
# str(OffsetI) + " ref : " + str(segIdx*step) + ' distance :' +
# str(float(distances[-1])/float(ranked[-1]))
            else:
                if debug:
                    print " Wrong answer " + str(estFileI) + " with offset " + str(OffsetI) +\
                        " instead of " + str(fileIndex) + " ref : " + str(segIdx * step) + ' distance :' + str(float(distances[-1]) / float(ranked[-1]))

#                pySigLocal.write('ConfusedFor-File'+str(fileIndex)+'-seg'+str(segIdx)+'.wav')
#                # Loading the one that has been confusing!"
#                ConfusingFilePath = file_names[estFileI];
#                pySigConfusing = signals.Signal(audio_files_path + ConfusingFilePath[:-1], True, True );
#                pySigConfusing.crop(OffsetI*pySigConfusing.fs, OffsetI*pySigConfusing.fs + segmentLength)
# pySigConfusing.write('ConfusedBy-
# File'+str(fileIndex)+'-seg'+str(segIdx)+'.wav')
        if debug:
            print "Global Score of " + str(countok / countall)
    return countok / countall


def create_sig_list(n_files, path='', filt=None):
    import cPickle
    from random import shuffle
    file_names = os.listdir(audio_files_path)
    if filter is not None:
        file_names = [a for a in file_names if filt in a]
        
#    fileIndexes = range(n_files)
    shuffle(file_names)
    
    sub_list = file_names[0:n_files]
    print sub_list
    savemat('%ssigList%d.mat'%(path, n_files),{'list':sub_list})
    
    output = open('%ssigList%d.list'%(path, n_files), 'wb')
    cPickle.dump(sub_list, output)

def get_sig_list(n_files, path='', filt=None):
    import cPickle
    if not os.path.exists('%ssigList%d.list'%(path, n_files)):
        create_sig_list(n_files, path, filt=filt)
    
    print "Opening %ssigList%d.list"%(path, n_files)
    file_obj = open('%ssigList%d.list'%(path, n_files), 'r')
#    dict = loadmat('%ssigList%d.mat'%(path, n_files))
    rand_list = cPickle.load(file_obj)
    return rand_list
