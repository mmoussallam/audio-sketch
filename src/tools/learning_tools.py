'''
tools.learning_tools  -  Created on Feb 6, 2013
@author: M. Moussallam
'''

import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.io import savemat, loadmat
import os.path as op
from PyMP import signals, mp
from PyMP.mdct import Dico, LODico
import sys
import os
sys.path.append('/home/manu/workspace/toolboxes/MSongsDB-master/PythonSrc')    
#import hdf5_utils as HDF5
import hdf5_getters    

from tempfile import mkdtemp
cachedir = mkdtemp()
from joblib import Memory
memory = Memory(cachedir='/data/tmp/joblib')


@memory.cache
def _compute_file(files_path, seg_size, seg_num,
                  pad, n_atom, energyTr, maxIndex,
                  dicotype, scales, c, filename , weights,
                  skeep_weak=False):
    ''' routine to avoid recomputing all the time the decompositions '''
    
    if dicotype == 'MDCT':
        pyDico = Dico(scales)
    elif dicotype == 'LoMP':
        pyDico = LODico(scales)
    c += 1.
#    print 100 * c / float(len(filenames)), '%'
    
    tmpsquareMat = csr_matrix((maxIndex, maxIndex))
    pySigLong = signals.LongSignal(op.join(files_path, filename),
                                   frame_size=seg_size, mono=True, Noverlap=0.5)
    if pySigLong.n_seg < seg_num:
        seg_num = pySigLong.n_seg
    print filename, seg_num
    for i in range(0, seg_num):
#            print i
        if (i % 20) == 0:
            print (100 * i) / (seg_num), '%',
        pySig = pySigLong.get_sub_signal(
            i, 1, mono=True, normalize=False, pad=False)
#            pySig.data *= window;
        pySig.pad(pad)
        segEnergy = np.sum(pySig.data ** 2) / float(pySig.length)
#            print segEnergy
        if (segEnergy < energyTr) and skeep_weak:
            print i, ' too weak:', segEnergy
        else:
#            nbUsedSeg += 1
            approx = mp.mp(pySig, pyDico, 100, n_atom, pad=False, silent_fail=True)[0]
            spMat = approx.to_sparse_array()
            maxValue = np.max([[val[0] for val in spMat.values()]])
        #            spVec = np.zeros((maxIndex,1), dtype=np.int8);
            spVec = np.zeros((maxIndex, ))
            if weights:
                spVec[[spMat.keys()]] = [val[0]/maxValue for val in spMat.values()]
            else:
                spVec[[spMat.keys()]] = 1
            spVec = csr_matrix(spVec)
            tmpsquareMat = tmpsquareMat + (kron(spVec, spVec.T)).tocsr()
    
    return [tmpsquareMat, approx.length, approx.fs]

def compute_cooc_mat(files_path, output_root_path, group_name,
                            filenames, seg_size, seg_num, pad, scales,
                            n_atom, energyTr, dicotype='MDCT', weights=False):
    ''' Run MP decomposition on seg_num segments of each of the files 
        with given parameters. Then count co-occurrences in a sparse matrix
        
        seg_size is the segment size
        
        energyTr is a energy threshold under which the segment is skipped
        
    '''

    maxIndex = len(scales) * (seg_size + (2 * pad))
#    squareMat = csr_matrix((maxIndex,maxIndex),dtype=np.int8);
    squareMat = csr_matrix((maxIndex, maxIndex))

    # squareMat = np.zeros((maxIndex,maxIndex),dtype=np.int8)
#    nbUsedSeg = 0
    c = 0.
    length = 0
    sr = 0
    for filename in filenames:
#        try:
        tmpsquareMat, L, fs = _compute_file(files_path,
                                      seg_size, seg_num, pad,
                                      n_atom, energyTr,
                                      maxIndex, 
                                      dicotype, scales, c,
                                      filename, weights)
        length = L
        sr = fs
#        except:
#            print "Something failed for ",filename
#            continue
        squareMat = squareMat + tmpsquareMat
    
    weight_str = ''
    if weights:
        weight_str = "_weights_"
        
    # Saving the matrix to spare the decomposition of it again
    targetPath = '%s/%s_Learned%sCoocMat_%d_M_%d_I_%d_nAtom_%d_%dx%s' % (output_root_path,
                                                                       group_name,
                                                                       weight_str,
                                                                       len(filenames),
                                                                       seg_size,
                                                                       seg_num, n_atom,
                                                                       len(scales),
                                                                       dicotype)
    print targetPath, length, sr
    savemat(targetPath + '.mat', {'squareMat': squareMat,
                                  'L': length,
                                  'fs': sr})

    return squareMat, length, sr

def load_cooc_mat(files_path, output_root_path, group_name,
                    filenames, seg_size, seg_num, pad, scales,
                    n_atoms, energyTr, dicotype='MDCT',weights=False):
    
    ''' Load previously computed coocurrence matrix            
        '''
    if dicotype == 'MDCT':
        pyDico = Dico(scales)
    elif dicotype == 'LoMP':
        pyDico = LODico(scales)
    
    weight_str = ''
    if weights:
        weight_str = "_weights_"

    # Loading the matrix to spare the decomposition of it again
    targetPath = '%s/%s_Learned%sCoocMat_%d_M_%d_I_%d_nAtom_%d_%dx%s' % (output_root_path,
                                                                       group_name,
                                                                       weight_str,
                                                                       len(filenames),
                                                                       seg_size,
                                                                       seg_num, n_atoms,
                                                                       len(scales),
                                                                       dicotype)
    D = loadmat(targetPath + '.mat')
    spmat = D['squareMat']
    l = D['L'][0, 0]
    Fs = D['fs'][0, 0]
#    approx = app.read_from_mat_struct(D['approx'])
    return spmat, pyDico, l, Fs


def find_indexes(startIdx, array, stopvalue):
    """ get the indexes in the (sorted) array such that
    elements are smaller than value """
    idxset =[]
    idx = startIdx
    while idx <= array.shape[0]-1 and array[idx] < stopvalue:
        idxset.append(idx)
        idx +=1
#        print idx, array[idx]
    return idxset


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
                                      forbid_list=forbid_list)
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

def get_track_info(h5file):
    h5 = hdf5_getters.open_h5_file_read(h5file)
    title = hdf5_getters.get_title(h5)
    artist = hdf5_getters.get_artist_name(h5)
    return title, artist
    
def get_ten_features_from_file(feats_all, segments_all, confidence_all, h5file):
    h5 = hdf5_getters.open_h5_file_read(h5file)
    timbre = hdf5_getters.get_segments_timbre(h5)
    loudness_start = hdf5_getters.get_segments_loudness_start(h5)
    loudness_max = hdf5_getters.get_segments_loudness_max(h5)
    loudness_max_time = hdf5_getters.get_segments_loudness_max_time(h5)
    C = hdf5_getters.get_segments_pitches(h5)
    
    confidence_all.append(hdf5_getters.get_segments_confidence(h5))
    
    segments_all.append(np.array([hdf5_getters.get_segments_start(h5), os.path.splitext(os.path.split(h5file)[-1])[0]]))
    
    feats_all.append(np.hstack((timbre, loudness_start.reshape((loudness_start.shape[0], 1)), loudness_max.reshape((loudness_max.shape[0], 1)), loudness_max_time.reshape((loudness_max_time.shape[0], 1)), C)))
    h5.close()

def get_ten_features(h5_dir):
    
    # load the Echo Nest features
    feats_all = []
    segments_all = []
    conf_all = []
    for h5file in get_filepaths(h5_dir, ext='.h5'):
        get_ten_features_from_file(feats_all, segments_all, conf_all, h5file)
        
    feats = np.concatenate(feats_all, axis=0)
    segments = np.vstack(segments_all)    
    confidence = np.concatenate(conf_all)
    return feats, segments, confidence


#def save_audio_full_ref(learntype, test_file, n_feat, rescale_str, sigout, fs, norm_segments=False):
#    """ do not cut the sounds """
#    # first pass for total length
#    max_idx = int(sigout[-1][1] + len(sigout[-1][0])) + 4*fs
#    print "total length of ",max_idx
#    sig_data = np.zeros((max_idx,))
##    seg_energy = np.sum(sigout[-1][0]**2)
#    for (sig, startidx) in sigout:
##        print sig.shape, sig_data[int(startidx):int(startidx)+sig.shape[0]].shape
#        sig_data[int(startidx):int(startidx)+sig.shape[0]] += sig#*seg_energy/np.sum(sig**2)
#    from PyMP import Signal
#    rec_sig = Signal(sig_data, fs, normalize=True)
#    rec_sig.write('%s/%s_with%s_%dfeats_%s%s.wav' % (outputpath,
#                                                   os.path.split(test_file)[-1],
#                                                   learntype, n_feat,
#                                                   rescale_str,
#                                                   'full_ref'))
#    
#    

def _get_seg_slicing(learn_feats, learn_segs):
    (n_seg, n_feats) = learn_feats.shape
    ref_seg_indices = np.zeros(n_seg,int)
    l_seg_duration = np.zeros(n_seg)
    l_seg_start = np.zeros(n_seg)
    c_idx = 0
    for segI in range(learn_segs.shape[0]):
        n_ub_seg = len(learn_segs[segI,0])
        if c_idx + n_ub_seg>n_seg:
            break
        ref_seg_indices[c_idx:c_idx+n_ub_seg] = segI
        l_seg_start[c_idx:c_idx+n_ub_seg] = learn_segs[segI,0]
        l_seg_duration[c_idx:c_idx+n_ub_seg-1] = learn_segs[segI,0][1:] - learn_segs[segI,0][0:-1]
        c_idx += n_ub_seg
    return l_seg_start, l_seg_duration, ref_seg_indices
    

def resynth(ref_indexes, start_times, dur_times, 
            learn_segs, learn_feats, ref_audio_dir, ext,
            dotime_stretch=False, max_synth_idx=None, normalize=False):
    """ Resynthesize using the reference files """

    l_seg_start, l_seg_duration , ref_seg_indices= _get_seg_slicing(learn_feats, learn_segs)

    from feat_invert.transforms import get_audio, time_stretch
    if max_synth_idx is None:
        max_synth_idx = len(ref_indexes)
    total_target_duration = 0
    sigout = []
    for seg_idx in range(max_synth_idx):
        print "----- %d/%d ----"%(seg_idx, max_synth_idx)
        
        target_audio_duration = dur_times[seg_idx]  
        total_target_duration += target_audio_duration 
                
        # Recover info from the reference
        ref_seg_idx = ref_indexes[seg_idx]
        print ref_seg_idx, ref_seg_indices[ref_seg_idx]
        ref_audio_path = learn_segs[ref_seg_indices[ref_seg_idx],1]
        ref_audio_start = l_seg_start[ref_seg_idx]
        ref_audio_duration = l_seg_duration[ref_seg_idx]    
    
        length_ratios = float(ref_audio_duration)/float(target_audio_duration)
        if length_ratios <= 0:
            continue
    
        # Load the reference audio
        filepath = ref_audio_dir + ref_audio_path + ext
        print "Loading %s  "%( filepath)
        signalin, fs = get_audio(filepath, ref_audio_start, ref_audio_duration)
#        if normalize:
#            signalin = signalin.astype(float)
#            signalin /= np.max(np.abs(signalin))
        target_length = target_audio_duration*fs
        print "Loaded %s length of %d "%( filepath, len(signalin))
        if dotime_stretch:
            print "Stretching to %2.2f"%length_ratios
            sigout.append(time_stretch(signalin, length_ratios, wsize=1024, tstep=128)[128:-1024])
        else:
            sigout.append(signalin)

    if normalize:
        for sig in sigout:
#            print np.max(np.abs(sig))
            sig /= np.max(np.abs(sig))
            sig = sig.astype(float)
#            print np.max(np.abs(sig))
        
    return sigout                
#    save_audio_full_ref(learntype,  test_file, n_feat, '_full_ref_', sigout, fs, norm_segments=False)

def resynth_sequence(ref_indexes, start_times, dur_times, 
            learn_segs, learn_feats, ref_audio_dir, ext, fs,
            dotime_stretch=False, max_synth_idx=None, normalize=False, marge=10, verbose=False):
    """ Resynthesize the target object """
    
    l_seg_start, l_seg_duration, ref_seg_indices = _get_seg_slicing(learn_feats, learn_segs)
    
    # initialize array
    if max_synth_idx is None:
        max_synth_idx = len(ref_indexes)
    
    total_target_duration = np.sum(dur_times[:max_synth_idx]) + marge
    if verbose:
        print total_target_duration
    resynth_data = np.zeros(total_target_duration*fs)
    from feat_invert.transforms import get_audio, time_stretch
    
    names = []
    # LOOP on segments
    for seg_idx in range(max_synth_idx):
        if verbose:
            print "----- %d/%d ----"%(seg_idx, max_synth_idx)
        
        target_audio_start = int(start_times[seg_idx]*fs)
        target_audio_duration = dur_times[seg_idx]          
                
        # Recover info from the reference
        ref_seg_idx = ref_indexes[seg_idx]        
        ref_audio_path = learn_segs[ref_seg_indices[ref_seg_idx],1]
        ref_audio_start = l_seg_start[ref_seg_idx]
        ref_audio_duration = l_seg_duration[ref_seg_idx]    
    
        stretch_ratio = float(ref_audio_duration)/float(target_audio_duration)
        if stretch_ratio <= 0:
            continue    
        # Load the reference audio
        filepath = ref_audio_dir + ref_audio_path + ext
        names.append(os.path.split(filepath)[-1])
#        print "Loading %s  "%( os.path.split(filepath)[-1]),
        signalin, fs = get_audio(filepath, ref_audio_start, ref_audio_duration,
                                 targetfs=fs, verbose=verbose)        
            
        # now add it to the signal, with or without time stretching
        if dotime_stretch:
            print "Stretching to %2.2f"%stretch_ratio
            if stretch_ratio < 1.0:
                stretched = time_stretch(signalin, stretch_ratio, wsize=1024, tstep=128)
            else:
                stretched = signalin.astype(float)
            if normalize:
                stretched /= 1.5*float(np.max(np.abs(stretched)))
                stretched = stretched.astype(float)
            
            resynth_data[target_audio_start:target_audio_start+stretched.shape[0]] += stretched                    
        else:            
            if normalize:
                signalin = signalin.astype(float)
                signalin /= 1.5*float(np.max(np.abs(signalin)))
                
            resynth_data[target_audio_start:target_audio_start+len(signalin)] += signalin.astype(float)
    from collections import Counter
    counts =  Counter(names) 
    if np.max(counts.values()) > 0.5*max_synth_idx:
        print "WARNING: probable duplicate ",np.max(counts.values),0.5*max_synth_idx, counts
    return resynth_data


def resynth_single_seg(ref_index, start_time, dur_time, 
            learn_segs, learn_feats, ref_audio_dir, ext, fs,
            dotime_stretch=False, normalize=False, marge=0.1):
    """ Resynthesize the target object """
    
    l_seg_start, l_seg_duration, ref_seg_indices = _get_seg_slicing(learn_feats, learn_segs)
    
    from feat_invert.transforms import get_audio, time_stretch
    
    # LOOP on segments    
    target_audio_start = int(start_time*fs)
    target_audio_duration = dur_time + marge    
    resynth_data = np.zeros(target_audio_duration*fs)
    # Recover info from the reference
    ref_seg_idx = ref_index     
    ref_audio_path = learn_segs[ref_seg_indices[ref_seg_idx],1]
    ref_audio_start = l_seg_start[ref_seg_idx]
    ref_audio_duration = l_seg_duration[ref_seg_idx]    
    
#    print  ref_seg_idx, ref_audio_duration, target_audio_duration
    stretch_ratio = float(ref_audio_duration)/float(target_audio_duration)
    
    if stretch_ratio <= 0:
        return None    
    # Load the reference audio
    filepath = ref_audio_dir + ref_audio_path + ext
#    print "Loading %s  "%( filepath)
    signalin, fs = get_audio(filepath, ref_audio_start, ref_audio_duration, targetfs=fs)        
        
    # now add it to the signal, with or without time stretching
    if dotime_stretch:
#        print "Stretching to %2.2f"%stretch_ratio
        if stretch_ratio < 1.0:
            stretched = time_stretch(signalin, stretch_ratio, wsize=1024, tstep=128)
        else:
            stretched = signalin.astype(float)
        if normalize:
            stretched /= 1.5*float(np.max(np.abs(stretched)))
            stretched = stretched.astype(float)
        
        resynth_data[0:len(stretched)] = stretched                    
    else:            
        if normalize:
            signalin = signalin.astype(float)
            signalin /= 1.5*float(np.max(np.abs(signalin)))
            
        resynth_data[0:min(len(signalin),len(resynth_data))] = signalin[0:min(len(signalin),len(resynth_data))].astype(float)
    
    return resynth_data

def save_audio(outputpath, aud_str, sigout,  fs, norm_segments=False):
    """ saving output vector to an audio wav"""
    norm_str = ''
    if norm_segments:
        norm_str = 'normed'
        mean_energy = np.mean([np.sum(sig**2)/float(len(sig)) for sig in sigout])
        print mean_energy
        for sig in sigout:
            sig /= np.sum(sig**2)/float(len(sig))
            sig *= mean_energy        
    rec_sig = signals.Signal(np.concatenate(sigout), fs, normalize=True)
    rec_sig.write('%s/%s.wav' % (outputpath,aud_str))
    return rec_sig

def Viterbi(neighbs, distance, t_penalty, c_value=5):
    # can we perform viterbi decoding ?
    n_candidates = neighbs.shape[1]
    n_states = neighbs.shape[0]
    transition_cost = np.ones((n_candidates,))
    cum_scores = np.zeros((n_candidates,))
    paths = []
    # initalize the paths and scores
    for candIdx in range(n_candidates):
        paths.append([0,])
        cum_scores = distance[0,:]
    
    for stateIdx in range(1, n_states):
        for candIdx in range(n_candidates):
            trans_penalty = [1 if not abs(neighbs[stateIdx-1,i]-neighbs[stateIdx,candIdx])<c_value else t_penalty for i in range(n_candidates)]
            trans_score = trans_penalty * cum_scores # to be replaced by a penalty of moving far from previous index         
            best_prev_ind = np.argmin(trans_score)        
            paths[candIdx].append(best_prev_ind)
            cum_scores[candIdx] = distance[stateIdx,candIdx] + trans_score[best_prev_ind]
    
    best_score_ind = np.argmin(cum_scores)
    best_path = paths[best_score_ind]
    return best_path


#def SimPenalize(neighbs, distance, sim_mat, l_feats, sim_penalty, c_value=5):
#    """ use also the correlation with previous segments to influence the choice """
#    n_candidates = neighbs.shape[1]
#    n_states = neighbs.shape[0]
#    transition_cost = np.ones((n_candidates,))
#    cum_scores = np.zeros((n_candidates,))
#    paths = []
#    for candIdx in range(n_candidates):
#        paths.append([0,])
#        cum_scores = distance[0,:]
#        
#    # for each state, weight the distance by the 
#    for stateIdx in range(1, n_states):
#        # for each candidate
#        for candIdx in range(n_candidates):
#            # compute the feat distance to all previous frames
#            # weighted by the similarity
#            scores = l_feats[neighbs[stateIdx,candIdx],:] - 
#            
#    best_score_ind = np.argmin(cum_scores)
#    best_path = paths[best_score_ind]
#    return best_path
    