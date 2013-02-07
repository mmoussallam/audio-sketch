'''
experiments.expe_learn_cooc_mat_rwc  -  Created on Jan 31, 2013
@author: M. Moussallam
'''

#@todo: # TODO  Implement learning of structures on the rwc database
# principle: or each song, for each bit, decompose on union of MDCT 100 first atoms
# then calculate for each atom, what atoms are co-occurent: add +1 in the corresponding
# co-occcurence matrix

# we can do that not only for sparse rep, but also for STFT peaks,
# Auditory peaks etc..
import os
import numpy as np
# import cProfile
import math
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tools.learning_tools import compute_cooc_mat, load_cooc_mat

from PyMP import approx as app
from PyMP.mdct import Dico, LODico, atom

herePath = '/home/manu/workspace/audio-sketch/src/expe_scripts/'
figurePath = op.join(herePath, 'figures')
audioPath = op.join(herePath, 'audio')

#filePath = '/sons/sqam/'
#fileGroup = 'SQAM'
#filenames = (
#    'glocs.wav', 'trumpet.wav', 'orchestra.wav', 'voicemale.wav', 'vega.wav')

fileGroup = 'RWC_genre'
filePath = '/sons/rwc/'
foldernames = [f for f in os.listdir(filePath) if 'rwc-g' in f]

# Loading all the files
filenames = []
for f in foldernames:
    filenames.extend(['%s/%s'%(f,a) for a in os.listdir(filePath + f) if '32' in a])


M = 6 * 8192
I = 150
pad = 8192
NbAtom = 100
dicotype = 'LoMP'
scales = [128, 1024, 8192]

energyTr = M / 2
# HEURISTIQUE!
maxIndex = len(scales) * (M + (2 * pad))
window = np.hanning(M)

args = [filePath,herePath,fileGroup, filenames, M, I, pad, scales, NbAtom, energyTr]
kwargs = {'dicotype': dicotype}

compute = False

if compute:
    squareMat, L, fs = compute_cooc_mat(*args, **kwargs)
    if dicotype == 'MDCT':
        pyDico = Dico(scales)
    elif dicotype == 'LoMP':
        pyDico = LODico(scales)
else:
    squareMat, pyDico, L, fs = load_cooc_mat(*args, **kwargs)

### Now we have a sparse matrix, we convert it in three local matrices

# plt.figure()
# plt.spy(squareMat,marker='.',markersize=0.2)
# plt.show()

# For now we concentrate on one scale: the big one
localApprox = app.Approx(pyDico, [], None,
                         2 * L,
                         fs)

scaleNum = 2
if scaleNum <= len(scales):
    firstIndex = int(math.floor(scaleNum * maxIndex / len(scales)))
    lastIndex = int(math.floor((scaleNum + 1) * maxIndex / len(scales)))
else:
    firstIndex = 0
    lastIndex = maxIndex
# Get all atoms in the last scale
pruningTr = 10
coocTr = 0.2
MaxScore = np.max(squareMat[firstIndex:lastIndex, firstIndex:lastIndex].data)
nbTrials = len(filenames) * I
pruningTr = MaxScore / 3

print " Number of targets:", np.sum(squareMat[firstIndex:lastIndex, firstIndex:lastIndex].data > pruningTr)

# np.sum((squareMat.data*coocTr) > pruningTr)

for atomIndex in range(firstIndex, lastIndex):

    atomWeight = squareMat[atomIndex, atomIndex]
    #.sum()

    if atomWeight < pruningTr:
        continue
    # Frequency and Time Position

    blockInd = int(np.floor(atomIndex / L))
#    if blockInd <2:
#        raise ValueError('Oups!');

    temp = squareMat[atomIndex, :]

    refframeInd = int(
        np.floor(2 * (atomIndex - blockInd * L) / scales[blockInd]))
    refRealTimePos = refframeInd * scales[blockInd] / 2
    refRealFreq = (atomIndex - (
        blockInd * L) - refframeInd * scales[blockInd] / 2) * fs / scales[blockInd]
#    print atomIndex, atomWeight , refframeInd , refRealFreq

    coTriggeredAtomIndexes = temp.nonzero()
    coAtomList = coTriggeredAtomIndexes[1].tolist()

#    print coAtomList
# rebuild atoms
    for atomI in range(len(coAtomList)):

        atomInd = coAtomList[atomI]
        if atomInd < firstIndex or atomInd > lastIndex:
            continue
            # FOR NOW ONLY CONSIDER SAME SCALE

        if atomInd == atomIndex:
            continue

        coeff = (float(temp.data[atomI]))
        # /float(atomWeight))

        if coeff > atomWeight:
            print 'Weird.....'

        if float(coeff) / float(atomWeight) <= coocTr:
            continue

    # atomInd = int(block*self.length +  frame*float(atom.length /2) +
    # atom.frequencyBin)
        blockInd = int(np.floor(atomInd / L))
        atomScale = scales[blockInd]
        # retrieve time position
        frameInd = int(np.floor(2 * (atomInd - blockInd * L) / atomScale))
        # retrieve frequency index
        freq = atomInd - (blockInd * L) - frameInd * atomScale / 2

        NewTimePos = L + (frameInd * atomScale / 2) - refRealTimePos
#        newRealFreq = fs/4 + (float(freq*fs)/float(atomScale))  - refRealFreq;
        newRealFreq = (float(freq * fs) / float(atomScale)) / (refRealFreq)

#        newFreqBin = math.floor(newRealFreq * atomScale/fs);
        newFreqBin = newRealFreq * atomScale / fs
        newAtom = atom.Atom(atomScale, float(coeff) / float(atomWeight), NewTimePos, newFreqBin, fs, float(coeff) / float(atomWeight))
# print atomScale , frameInd , freq , coeff , newFreqBin ,
# float(coeff)/float(atomWeight)
        localApprox.add(newAtom, noWf=True)

target_path = '%s_MolCooc_Scale%d_Prun_%1.1f_%dfiles_M_%d_I_%d_nbAtom_%d_%dx%s' % (fileGroup,
                                                                                   scaleNum, coocTr,
                                                                                   len(filenames),
                                                                                   M, I, NbAtom,
                                                                                   len(scales), dicotype)


plt.figure()
localApprox.plot_tf(multicolor=True,
                    # recenter=(fs/4,float(L+pad/2)/float(fs)),
                    recenter=(0.01, float(L) / float(fs)),
                    keepValues=True,
                    Alpha=False,
                    french=True,
                    ylim=(-14, 14),
                    logF=True)
plt.grid()
plt.yticks(range(-12, 13, 2))
plt.ylabel('Frequence relative (echelle MIDI)')


#localApprox.recomposed_signal.write(op.join(audioPath, target_path + '.wav'))

plt.savefig(op.join(figurePath, target_path + '.png'))
plt.savefig(op.join(figurePath, target_path + '.pdf'))
plt.show()
