'''
manu_sandbox.sparse_coocurrence_matrix  -  Created on Oct 7, 2013
@author: M. Moussallam
'''
import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
mem = Memory('/tmp/audio-sketch/')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
max_file_num = 20

seg_dur = 4
sparsity = 100
fs = 8000.0
# the density is approximately of sparsity/seg_dur features per second
scales = [128,1024,8192]
skhandlename = XMDCTSparseSketch
nature = 'MDCT'
params = {'downsample':fs,'scales':scales,'nature':nature,'n_atoms':sparsity}

skhandle = skhandlename(**params)

sp_reps = []
freqs = []
times = []


# calibrate the atom number
M = int(len(scales)*seg_dur*fs + 2*scales[-1])
from scipy.sparse import dok_matrix
#sp_mat = coo_matrix((M,M))
#sp_mat = np.zeros((M,M), dtype=np.int8)
# Coarse frequency dependency matrix
F = int(fs)
T = int(seg_dur*fs + 2*scales[-1])
freq_sp_mat = dok_matrix((F/2,F/2))
freq_biais = []
subsampfact = 1
freq_sp_tens = dok_matrix(((F/(2*subsampfact))**2,F/(2*subsampfact)))

time_sp_mat = dok_matrix((T,T))
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
#        print time.time()-t,
        for atomIdx in range(1, skhandle.rep.atom_number):
            cur_f = int(skhandle.rep.atoms[atomIdx].reduced_frequency * fs)
            prec_f = int(skhandle.rep.atoms[atomIdx-1].reduced_frequency * fs)
            
            freq_sp_mat[prec_f,cur_f] += 1
            freq_biais.append(cur_f)
            cur_t = int(skhandle.rep.atoms[atomIdx].time_position)
            prec_t = int(skhandle.rep.atoms[atomIdx-1].time_position)
 
            time_sp_mat[cur_t,prec_t] += 1
        
        for atomIdx in range(2, skhandle.rep.atom_number):
            cur_f = int(skhandle.rep.atoms[atomIdx].reduced_frequency * fs)/subsampfact
            prec_f = int(skhandle.rep.atoms[atomIdx-1].reduced_frequency * fs)/subsampfact
            precprec_f = int(skhandle.rep.atoms[atomIdx-2].reduced_frequency * fs)/subsampfact
#            print precprec_f, prec_f, cur_f, prec_f + precprec_f*(F/(2*subsampfact))
            freq_sp_tens[prec_f + precprec_f*(F/(2*subsampfact)), cur_f] += 1
        
    print "elapsed ",time.time()-t

output_path = op.join(SKETCH_ROOT, 'src/manu_sandbox/outputs')
suffix = '%s_%dlearned_%dscales_%datoms'%(nature,max_file_num,len(scales), sparsity)
np.save(op.join(output_path, 'freq_sp_mat_%s'%suffix), freq_sp_mat.toarray())
for s in scales:
#    bins = np.linspace(1,F,s/2 +1)
    biais, bin_edges = np.histogram(freq_biais, s/2 , normed=True)
    print biais.shape
    np.save(op.join(output_path, 'biais_%d_%s'%(s,suffix)), biais)
    
        
plt.figure()
plt.subplot(121)
plt.spy(freq_sp_mat,marker='o',markersize=1)
plt.subplot(122)
plt.spy(freq_sp_tens,marker='o',markersize=1,aspect='auto')
plt.show()