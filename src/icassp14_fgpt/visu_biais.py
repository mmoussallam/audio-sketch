'''
manu_sandbox.visu_biais  -  Created on Oct 14, 2013
Take a dataset: compute the decomposition and visualize the biais on all atoms
Then see if it can be separable 
@author: M. Moussallam
'''
import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
#mem = Memory('/tmp/audio-sketch/')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
max_file_num = 20

seg_dur = 4
sparsity = 100
fs = 8000.0
# the density is approximately of sparsity/seg_dur features per second
scales = [2**j for j in range(5,12)]
skhandlename = XMDCTSparseSketch
nature = 'LOMDCT'
params = {'downsample':fs,'scales':scales,'nature':nature,'n_atoms':sparsity}

skhandle = skhandlename(**params)

sp_reps = []
freqs = []
times = []


# calibrate the atom number
M = np.sum([s/2 * seg_dur*fs for s in scales])
from scipy.sparse import dok_matrix
#sp_mat = coo_matrix((M,M))
#sp_mat = np.zeros((M,M), dtype=np.int8)
# Coarse frequency dependency matrix
print M
freq_biais = []
time_biais = []
scale_biais = []
global_biais = []

## Computing all the sparse rep
for fIdx, file_name in enumerate(file_names[:max_file_num]):
    l_sig =  LongSignal(file_name,frame_duration=seg_dur, mono=True)
    # loop on segments
    t = time.time()
    for segIdx in range(l_sig.n_seg):
        sub_sig = l_sig.get_sub_signal(segIdx,1, mono=True, normalize=True)
        sub_sig.resample(fs)
#        t = time.time()
        skhandle.recompute(sub_sig)
        print skhandle.rep
        for atom in skhandle.rep.atoms:
            freq_biais.append(atom.reduced_frequency * atom.fs)
            time_biais.append(float(atom.time_position)/atom.fs)
            scale_biais.append(atom.length)
            global_biais.append(skhandle.rep.dico.get_atom_key(atom, sub_sig.length))
        
    print "elapsed ",time.time()-t

figure_path = op.join(SKETCH_ROOT,'src/manu_sandbox/figures')
    
plt.figure(figsize=(8,10))
plt.subplot(411)
plt.hist(freq_biais, 200, normed=True)
plt.xlabel('Frequency (Hz)')
plt.subplot(412)
plt.hist(time_biais, 1000, normed=True)
plt.xlabel('Time (s)')
plt.subplot(413)
plt.hist(scale_biais, [float(s-2)/atom.fs for s in scales], normed=True,log=True)
plt.xlabel('Scale (s)')
plt.subplot(414)
plt.hist(global_biais, 1000, normed=True)
plt.xlabel('Atom Index')
plt.subplots_adjust(hspace=0.38)
plt.savefig(op.join(figure_path, 'empirical_biais_%s_%dfiles_%datoms.png'%(set_id,max_file_num,sparsity)))
plt.show()