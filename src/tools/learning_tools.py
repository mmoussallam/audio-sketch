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
import hdf5_utils as HDF5
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