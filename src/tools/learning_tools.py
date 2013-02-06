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


def compute_cooc_mat(files_path, output_root_path, group_name,
                            filenames, seg_size, seg_num, pad, scales,
                            n_atom, energyTr, dicotype='MDCT'):
    ''' Run MP decomposition on seg_num segments of each of the files 
        with given parameters. Then count co-occurrences in a sparse matrix
        
        seg_size is the segment size
        
        energyTr is a energy threshold under which the segment is skipped
        
    '''

    maxIndex = len(scales) * (seg_size + (2 * pad))
#    squareMat = csr_matrix((maxIndex,maxIndex),dtype=np.int8);
    squareMat = csr_matrix((maxIndex, maxIndex))
    if dicotype == 'MDCT':
        pyDico = Dico(scales)
    elif dicotype == 'LoMP':
        pyDico = LODico(scales)
    # squareMat = np.zeros((maxIndex,maxIndex),dtype=np.int8)
    nbUsedSeg = 0
    c = 0.
    for filename in filenames:
        c += 1.
        print 100 * c / float(len(filenames)), '%'
        print filename
        pySigLong = signals.LongSignal(op.join(files_path, filename),
                                       frame_size=seg_size, mono=True, Noverlap=0.5)

        for i in range(0, seg_num):
#            print i
            if (i % 20) == 0:
                print float(100 * i) / float(seg_num), '%'
            pySig = pySigLong.get_sub_signal(
                i, 1, mono=True, normalize=False, pad=False)
#            pySig.data *= window;
            pySig.pad(pad)

            segEnergy = np.sum(pySig.data ** 2) / float(pySig.length)
#            print segEnergy
            if segEnergy < energyTr:
                print i, ' too weak:', segEnergy

                continue

            nbUsedSeg += 1

            approx = mp.mp(pySig, pyDico, 100, n_atom,
                                  pad=False, silent_fail=True)[0]

            spMat = approx.to_sparse_array()

#            spVec = np.zeros((maxIndex,1), dtype=np.int8);
            spVec = np.zeros((maxIndex, 1))
            spVec[[spMat.keys()]] = 1
            spVec = csr_matrix(spVec)
            squareMat = squareMat + (kron(spVec, spVec.T)).tocsr()

    # Saving the matrix to spare the decomposition of it again
    targetPath = '%s/%s_LearnedCoocMat_%d_M_%d_I_%d_nAtom_%d_%dx%s' % (output_root_path,
                                                                       group_name,
                                                                       len(filenames),
                                                                       seg_size,
                                                                       seg_num, n_atom,
                                                                       len(scales),
                                                                       dicotype)
    print targetPath
    savemat(targetPath + '.mat', {'squareMat': squareMat,
                                  'L': approx.length,
                                  'fs': approx.fs})

    return squareMat, pyDico, approx.length, approx.fs


def load_cooc_mat(files_path, output_root_path, group_name,
                         filenames, seg_size, seg_num, pad, scales,
                         n_atoms, energyTr, dicotype='MDCT'):
    ''' Load previously computed coocurrence matrix            
        '''
    if dicotype == 'MDCT':
        pyDico = Dico(scales)
    elif dicotype == 'LoMP':
        pyDico = LODico(scales)

    # Loading the matrix to spare the decomposition of it again
    targetPath = '%s/%s_LearnedCoocMat_%d_M_%d_I_%d_nAtom_%d_%dx%s' % (output_root_path,
                                                                       group_name,
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