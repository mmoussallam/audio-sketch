'''
Created on Sep 14, 2011

Tools for fgpt experiments

@author: moussall
'''
from scipy.io import savemat, loadmat
import numpy as np
import time
import os
import os.path as op
from joblib import Parallel, delayed

from PyMP import signals
#from classes import pydb, sketch
from src.classes.sketches.base import AudioSketch
from src.classes.fingerprints.base import FgptHandle

audio_files_path = '/sons/rwc/rwc-g-m01/'
default_db_path = '/home/manu/workspace/audio-sketch/fgpt_db/'


def _process_seg_test(sk, sparsity,  resample, pad, l_sig, segIdx):
    
    sig_local = l_sig.get_sub_signal(segIdx, 1, 
        mono=True, 
        normalize=True, 
        pad=pad)
# run the decomposition
    if resample > 0:
        sig_local.resample(resample)
# computing the local fingerprint
    sk.recompute(sig_local)
    sk.sparsify(sparsity)
    fgpt = sk.fgpt()
# now ask the handle what it estimates
    
    return (fgpt, segIdx)

def db_test(fgpthandle,
            sk,
            sparsity,
            file_names,
            files_path = '',            
            test_seg_prop = 0.5,
            seg_duration=5.0,
            resample = -1,
            step=2.5,
            tolerance = 5.0,
            shuffle=True,
            debug=False,
            n_jobs=3):
    ''' Lets try to identify random segments from the files 
    using the pre-calculated database
    
    Parameters
    ----------
    fgpthandle : FgptHandle
        the object that encapsulate the fingerprints
    sk : AudioSketch
        any Sketch object that is able to compute the fingerprints handled by fgpthandle
    sparsity : int
        dimension of the fingerprint
    file_names :  list
        a list of file names
    files_path : string (opt)
        The static path to where the files are actually located. Use only
        if all the files belong to the same directory
    test_seg_prop :  float (opt)
        value between 0 and 1, the proportion of segments that
        serve for testing
    seg_duration : float (opt)
        The duration of the segments in seconds
    tolerance :  float (opt)
        The tolerance on the segment localization in the file.
    resample : int (opt)
        The desired resampling frequency, default is -1 for no resampling
    step : float (opt)
        The step between segments in seconds. Default is -1 for no overlap
    
    '''
    if test_seg_prop <= 0 or test_seg_prop >1:
        raise ValueError("Unproper test_seg_prop parameter should be between 0 and 1 but got %1.1f"%test_seg_prop)

    n_files = len(file_names)
    
    if fgpthandle.params.has_key('pad'):
        pad = fgpthandle.params['pad']
    else:
        pad = False
    
    # change the order of the files for testing"
    sortedIndexes = range(n_files)
    if shuffle:
        np.random.shuffle(sortedIndexes)

    countokok = 0.0    # number of correctly retrieved segments
    countokbad = 0.0   # number of segments retrieve in correct file but misplaced
    countbadbad = 0.0  # number of segments in the wrong file
    countall = 0.0     # total number of segments
    t0 = time.time()
    i = 0
    failures = []
    for fileIndex in sortedIndexes:
        i +=1
        if seg_duration > 0:
            # get file as a PyMP.LongSignal object
            l_sig = signals.LongSignal(op.join(files_path, file_names[fileIndex]),
                             frame_duration = seg_duration, 
                             mono = True,
                             Noverlap = (1.0 - float(step)/float(seg_duration)))
            n_segs = l_sig.n_seg
            if n_segs < 1:
                continue
            
            if debug: print "Loaded file %s - with %d segments of %1.1f seconds"%(file_names[fileIndex],
                                                                                   n_segs,
                                                                                   seg_duration)
                
            segment_indexes = range(int(l_sig.n_seg))
            if shuffle:
                np.random.shuffle(segment_indexes)
            max_seg = int(test_seg_prop * l_sig.n_seg)
            
            
            # Loop on random segments*
            if n_jobs>1:              
                fgpts = Parallel(n_jobs=n_jobs)(delayed(_process_seg_test)(sk, sparsity,  resample,
                                                                       pad,l_sig, segIdx)
                                        for segIdx in segment_indexes[:max_seg-1])
                # Again ugly hack to counter the effects of joblib recopy of sketch object
                fgpts.append(_process_seg_test(sk, sparsity, resample, pad, l_sig, segment_indexes[-1]))
            else:
                fgpts = []
                for segIdx in segment_indexes[:max_seg-1]:
                    fgpts.append(_process_seg_test(sk, sparsity, resample, pad, l_sig, segIdx))
                     
        else:
            l_sig = signals.Signal(op.join(files_path, file_names[fileIndex]), mono=True)
            n_segs = 1
            if resample > 0:
                l_sig.resample(resample)
                # computing the local fingerprint
            sk.recompute(l_sig)
            sk.sparsify(sparsity)
            fgpts = ((sk.fgpt(),0),)
            
        for fgpt, segIdx in fgpts:
            countall += 1.0
            true_offset = segIdx * step
            estimated_index, estimated_offset = fgpthandle.get_candidate(fgpt, sk.params, nbCandidates=n_files, 
            smooth=1)
            if (fileIndex == estimated_index):
                if debug:
                    print "Correct answer, file %d" % fileIndex
                if np.abs(estimated_offset - true_offset) < tolerance:
                    countokok += 1.0
                    if debug:
                        print "Correct offset %d" % int(estimated_offset)
                else:
                    countokbad += 1.0
                    if debug:
                        print "Wrong offset %d instead of %d" % (int(estimated_offset), int(segIdx * step))
            else:
                countbadbad += 1.0
                failures.append((file_names[fileIndex], true_offset, file_names[estimated_index], estimated_offset))
                if debug:
                    print " Wrong answer File %d offset %d instead of File %d offset %d" % (estimated_index, 
                        int(estimated_offset), 
                        fileIndex, int(segIdx * step))
        
        estTime = (float( (time.time() - t0)) / float(i)) * (n_files - i)
        print 'Elapsed %d  min . Estimated : %d min and %d seconds'%((time.time() - t0)/60.0,(estTime / 60.0), estTime - (int(estTime / 60.0)*60) )

        print "Global Scores of %1.2f - %1.2f - %1.2f" %((countokok / countall,
                                                          countokbad / countall,
                                                          countbadbad / countall))
    print "Final Scores of %1.2f - %1.2f - %1.2f" %((countokok / countall,
                                                          countokbad / countall,
                                                          countbadbad / countall))
    return (countokok / countall, countokbad / countall, countbadbad / countall), failures


def db_test_cortico(fgpthandle,
            sk,
            sparsity,
            file_names,
            files_path = '',            
            test_seg_prop = 0.5,
            seg_duration=5.0,
            resample = -1,
            step=2.5,
            tolerance = 5.0,
            shuffle=True,
            debug=False,
            n_jobs=3,
            n_files=None):
    ''' Same as above but evaluation on each of the sub plots
    
    '''
    if test_seg_prop <= 0 or test_seg_prop >1:
        raise ValueError("Unproper test_seg_prop parameter should be between 0 and 1 but got %1.1f"%test_seg_prop)

    if n_files is None:
        n_files = len(file_names)
    
    if fgpthandle.params.has_key('pad'):
        pad = fgpthandle.params['pad']
    else:
        pad = False
    import time
    # change the order of the files for testing"
    sortedIndexes = range(len(file_names))
    if shuffle:
        np.random.shuffle(sortedIndexes)

    (n_scales, n_rates) = fgpthandle.params['n_sv'], fgpthandle.params['n_rv']
    countokok = np.zeros((n_scales, n_rates))    # number of correctly retrieved segments
    countokbad = np.zeros((n_scales, n_rates))   # number of segments retrieve in correct file but misplaced
    countbadbad = np.zeros((n_scales, n_rates))  # number of segments in the wrong file
    countall = 0.0     # total number of segments
    t0 = time.time()
    i = 0
#    failures = []
    for fileIndex in sortedIndexes:
        i +=1
        # get file as a PyMP.LongSignal object
        l_sig = signals.LongSignal(op.join(files_path, file_names[fileIndex]),
                             frame_duration = seg_duration, 
                             mono = True,
                             Noverlap = (1.0 - float(step)/float(seg_duration)))

        if debug: print "Loaded file %s - with %d segments of %1.1f seconds"%(file_names[fileIndex],
                                                                               l_sig.n_seg,
                                                                               seg_duration)
        segment_indexes = range(int(l_sig.n_seg))
        if shuffle:
            np.random.shuffle(segment_indexes)
        max_seg = int(test_seg_prop * l_sig.n_seg)
        
        # Loop on random segments*              
        fgpts = Parallel(n_jobs=n_jobs)(delayed(_process_seg_test)(sk, sparsity,  resample,
                                                                   pad,l_sig, segIdx)
                                    for segIdx in segment_indexes[:max_seg-1])
        # Again ugly hack to counter the effects of joblib recopy of sketch object
        fgpts.append(_process_seg_test(sk, sparsity, resample, pad, l_sig, segment_indexes[max_seg-1]))
        
        for fgpt, segIdx in fgpts:
            countall += 1.0
            true_offset = segIdx * step
            estimated_index, estimated_offset = fgpthandle.get_candidate(fgpt, sk.params, nbCandidates=n_files, 
            smooth=1)
            for scaleIdx in range(n_scales):
                for rateIdx in range(n_rates):
                    if (fileIndex == estimated_index[scaleIdx,rateIdx]):
                        if debug:
                            print "%d- %d : Correct answer, file %d" % (scaleIdx,rateIdx,fileIndex)
                        if np.abs(estimated_offset[scaleIdx,rateIdx] - true_offset) < tolerance:
                            countokok[scaleIdx,rateIdx] += 1.0
                            if debug:
                                print "%d- %d : Correct offset %d" % (scaleIdx,rateIdx,int(estimated_offset[scaleIdx,rateIdx]))
                        else:
                            countokbad[scaleIdx,rateIdx] += 1.0
                            if debug:
                                print "Wrong offset %d instead of %d" % (int(estimated_offset[scaleIdx,rateIdx]), int(segIdx * step))
                    else:
                        countbadbad[scaleIdx,rateIdx] += 1.0
#                        failures.append((file_names[fileIndex], true_offset, file_names[estimated_index], estimated_offset[scaleIdx,rateIdx]))
                        if debug:
                            print "%d- %d Wrong answer File %d offset %d instead of File %d offset %d" % (scaleIdx,rateIdx,estimated_index[scaleIdx,rateIdx], 
                                int(estimated_offset[scaleIdx,rateIdx]), 
                                fileIndex, int(segIdx * step))
                
        estTime = (float( (time.time() - t0)) / float(i)) * (n_files - i)
        print 'Elapsed %2.2f  min . Estimated : %2.2f min'%((time.time() - t0)/60.0,(estTime / 60.0))

        print "Ok Scores :" ,(countokok / countall)
    print "Final Ok Scores of " , (countokbad / countall)
    
    return (countokok / countall, countokbad / countall, countbadbad / countall)


def _process_seg(sk, sparsity, resample, step, debug, pad, l_sig,  segIdx, filename):        
    
    sig_local = l_sig.get_sub_signal(segIdx, 1, mono=True, 
        normalize=True, 
        pad=pad)
# run the decomposition
    if resample > 0:
        sig_local.resample(resample)
    sk.recompute(sig_local, **{'segIdx':segIdx,'sig_name':filename})
    sk.sparsify(sparsity)
    
    if debug:
        print "Populating database with offset %d " %(segIdx * step)
    return sk.fgpt()

def _process_file(fgpthandle, sk, sparsity, file_names, seg_duration, resample,
                  step, files_path, debug, n_files, pad, t0, fileIndex, n_jobs):
    # get file as a PyMP.LongSignal object
    if seg_duration>0:
        l_sig = signals.LongSignal(op.join(files_path, file_names[fileIndex]), frame_duration=seg_duration, 
                                   mono=True, 
                                   Noverlap=(1.0 - float(step) / float(seg_duration)))
        nbsegs = l_sig.n_seg
    else:
        l_sig = signals.Signal(op.join(files_path, file_names[fileIndex]), mono=True)
        nbsegs = 1
#    if debug:
    print "Loaded file %s - with %d segments of %1.1f seconds" % (file_names[fileIndex], nbsegs, 
            seg_duration)
    
    if n_jobs >1 and nbsegs>1:
        # Loop on segments :  Sparsifying all of them
        fgpt = Parallel(n_jobs=n_jobs)(delayed(_process_seg)(sk, sparsity, resample, step, debug, pad, l_sig, segIdx, file_names[fileIndex])
                                    for segIdx in range(l_sig.n_seg -1))
        
        # ugly hack: do the last segment without parallel so that the sketch object
        # is correctly modified
        fgpt.append(_process_seg(sk, sparsity, resample, step, debug, pad, l_sig, l_sig.n_seg-1,file_names[fileIndex]))

        for segIdx in range(l_sig.n_seg):
            fgpthandle.populate(fgpt[segIdx], sk.params, fileIndex, offset=segIdx * step, debug=debug)
    
    else:
        if nbsegs>1:
#            fgpt = []
            for segIdx in range(nbsegs):
                try:
                    fgpthandle.populate(_process_seg(sk, sparsity, resample, step, debug, pad,
                                                   l_sig,  segIdx, file_names[fileIndex]),
                                      sk.params, fileIndex, offset=segIdx * step, debug=debug)
                except:
                    continue 
        else:
            if resample > 0:
                l_sig.resample(resample)
            sk.recompute(l_sig, **{'segIdx':0,'sig_name':file_names[fileIndex]})
            sk.sparsify(sparsity)
            fgpthandle.populate(sk.fgpt(), sk.params, fileIndex, offset=0, debug=debug)

#            fgpt.append(_process_seg(sk, sparsity, resample, step, debug, pad, l_sig,  segIdx, file_names[fileIndex]))

    # Cannot parallelized this part though ... because of disk access

    estTime = (float((time.time() - t0)) / float(fileIndex + 1)) * (n_files - fileIndex)
    print 'Elapsed %d seconds Estimated : %d minutes and %d seconds' % ((time.time() - t0), (estTime / 60), estTime - (int(estTime / 60)*60))

def db_creation(fgpthandle,
                sk,
                sparsity,
                file_names,
               db_name=None, 
               force_recompute = False,
               seg_duration=5.0,
               resample = -1,               
               step=-1,
               files_path='',
               db_path = None,
               debug = True,
               n_jobs = 3):
    ''' method to create a DataBase from the given FgptHandle 
        a string with the db name to be created
        
    Will either create or load an already existing BerkeleyDB object
    
    Parameters
    ----------
    fgpthandle : FgptHandle
        the object that encapsulate the fingerprints
    sk : AudioSketch
        any Sketch object that is able to compute the fingerprints handled by fgpthandle
    sparsity : int
        dimension of the fingerprint
    file_names :  list
        a list of file names
    files_path : string (opt)
        The static path to where the files are actually located. Use only
        if all the files belong to the same directory
    db_name :  string (opt)
        the unique identifier of the database. If None : the string will be computed 
        from the FgptHandle db_name parameter.
    force_recompute :  bool (opt)
        If the db_name is provided and is found on the disk: the base is not recomputed
        unless this parameter is set to True
    seg_duration : float (opt)
        The duration of the segments in seconds
    resample : int (opt)
        The desired resampling frequency, default is -1 for no resampling
    step : float (opt)
        The step between segments in seconds. Default is -1 for no overlap
        
        '''
    n_files = len(file_names)
    
    if not isinstance(fgpthandle, FgptHandle):
        raise TypeError("First argument should be a pydb.FgptHandle but is a %s"%fgpthandle.__class__)
    
    if not isinstance(sk, AudioSketch):
        raise TypeError("Second argument should be a AudioSketch but is a %s"%sk.__class__)
    
    # Checking the name parameter
    if db_name is None:
        db_name = fgpthandle.db_name
    
    if debug: 
        print "Starting work on %s"%db_name
    import time
    
    # checking the segment overlap
    if step<0:
        step = seg_duration
    
    # checking whether the fingerprint need some signal padding
    if fgpthandle.params.has_key('pad'):
        pad = fgpthandle.params['pad']
    else:
        pad = False
    
    # erase existing .db file if it already exists
    print db_name
    if db_name is not None and op.exists(db_name) and not force_recompute: 
        print "Base already computed"
        return
    
    # Loop on files
    t0 = time.time()        
    for fileIndex in range(n_files):        
        _process_file(fgpthandle, sk, sparsity, file_names,
                      seg_duration, resample, step, files_path,
                      debug, n_files, pad, t0, fileIndex, n_jobs)


def xmdctfgpt_expe(file_names, n_atoms, scales,
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
                                                           len(scales.sizes),
                                                           scales.nature)

    # load or create the base
    ppdb = pydb.XMDCTBDB(db_name, load=(not create_base), persistent=True)

    if create_base:
        print 'Reconstructing The base'
        db_creation(ppdb, file_names, n_atoms, scales,
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
            score = db_test(ppdb, file_names, n_atoms, scales,
                           test_n_atom=testNatom, test_files=testFiles,
                           seg_duration=seg_duration, step=test_step)
        else:
            if threshold is None or n_atom_step is None:
                raise ValueError('Not enough arguments provided for Hierarchical pruning!!')

            score = db_hierarchical_test(
                ppdb, file_names, n_atoms, scales, n_test_files,
                testNatom=testNatom, testFiles=testFiles,
                seg_duration=seg_duration, step=test_step,
                threshold=threshold, nbAtomPerIter=n_atom_step)

    return score, ppdb, time.time() - t_start


def xmdctdb_creation(ppdb, file_names, n_atoms, scales,
               db_name=None, 
               seg_duration=5.0, 
               padZ=None, 
               step=None,
               files_path=None):
    ''' method to create a DB with the given dictionary and parameters 
        can take a pydb object or a string with the db name to be created'''
    n_files = len(file_names)
    sizes = scales.sizes
    
    if files_path is None:
        files_path = audio_files_path
    
    if not isinstance(ppdb, pydb.XMDCTBDB):
        
        if db_name is None:
            db_name = '../data/MPdb_%dfiles_%datoms_%dxMDCT_%s.db' %( 
                                                           n_files,
                                                           n_atoms,
                                                           len(sizes),
                                                           scales.nature)

        # create the base
        ppdb = pydb.XMDCTBDB(db_name, load=False)
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
                               scales, 20, n_atoms,
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





def xmdctdb_test(ppdb,
            file_names,
            n_atoms,
            scales,
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
    sizes = scales.sizes
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
                pySigLocal, scales, 20, nbAtoms, pad=False, silent_fail=True)

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

# ppdb, file_names , nbAtoms, scales , test_n_atom=None , test_files = None,
# seg_duration=5, step = 2.5):


def xmdctdb_hierarchical_test(ppdb,
                         file_names, 
                         n_atoms,
                         scales,
                         nbFiles,
                       testNatom=None, testFiles=None, seg_duration=5,
                       step=2.5, threshold=0.4, nbAtomPerIter=10, debug=True):
    ''' Same as db_test, except that the search is hierarchically pruned if 
    distance between best candidate
    and second best one is above a pre-defined threshold
    '''
    sizes = scales.sizes

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
                pySigLocal, scales, 20, nbAtomPerIter, pad=False)
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
                    approx, decay = mp.mp_continue(approx, pySigLocal, scales, 20, nbAtomPerIter, pad=False, debug=0)
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

def get_filepaths(audio_path, random_seed=None, forbid_list=[],ext='.wav'):
    """function [file_paths] = get_filepaths(audio_path, random_seed)
    % retrieves all the wav file names and relative path given the directory
    % if random_seed is specified: it applies a random suffling of the files
    % paths"""

    import os
    import os.path as op
    file_paths = []
    # root
    dir_list = os.listdir(audio_path)
    
    # recursive search
    for dir_ind in range(len(dir_list)):

        if op.isdir(op.join(audio_path, dir_list[dir_ind])):

            sub_files = get_filepaths(op.join(audio_path,
                                              dir_list[dir_ind]),
                                      forbid_list=forbid_list, ext=ext)
            file_paths.extend(sub_files)
        else:
            if ext in dir_list[dir_ind]:
                if not dir_list[dir_ind] in forbid_list:                                
                    file_paths.append(op.join(audio_path, dir_list[dir_ind]))

    if random_seed is not None:
        # use the random_seed to initialize random state
        np.random.seed(random_seed)
        file_paths = np.random.permutation(file_paths)

    return file_paths

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

def get_rnd_file(file_names, seg_duration, step, fileIndex, n_files=None, debug=True):
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
