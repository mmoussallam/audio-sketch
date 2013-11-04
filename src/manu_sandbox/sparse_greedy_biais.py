'''
manu_sandbox.sparse_greedy_biais  -  Created on Oct 7, 2013
@author: M. Moussallam

We want to expose the strong biais in the atom selection process
'''
import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
mem = Memory('/tmp/audio-sketch/')
figure_path = op.join(SKETCH_ROOT,'src/manu_sandbox/figures')

set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
max_file_num = 16

seg_dur = 5
sparsity = 100
fs = 8000.0
# the density is approximately of sparsity/seg_dur features per second
scales = [2**j for j in range(5,12)]
skhandlename = XMDCTSparseSketch
params = {'downsample':fs,'scales':scales,'nature':'LOMDCT','n_atoms':sparsity}

skhandle = skhandlename(**params)

sp_reps = []
freqs = []
times = []

firstfreqs  = []
firsttimes = []
firstscales = []

# Computing all the sparse rep
for fIdx, file_name in enumerate(file_names[:max_file_num]):
    l_sig =  LongSignal(file_name,frame_duration=seg_dur, mono=True)
    # loop on segments
    for segIdx in range(l_sig.n_seg):
        sub_sig = l_sig.get_sub_signal(segIdx,1, mono=True, normalize=True)
        sub_sig.resample(fs)
        skhandle.recompute(sub_sig)            
        firstfreqs.append(skhandle.rep.atoms[0].reduced_frequency)
        firsttimes.append(float(skhandle.rep.atoms[0].time_position)/fs) 
        firstscales.append(np.log2(skhandle.rep.atoms[0].length))
        for atom in skhandle.rep.atoms:
            freqs.append(atom.reduced_frequency)
            times.append(float(atom.time_position)/fs)


#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.add_axes([0.3,0.3,0.6,0.6], projection='3d')
#ax.scatter(firsttimes, firstfreqs, firstscales)
#ax1 = fig.add_axes([0.3,0.07,0.6,0.15], sharex=ax)
#ax1.hist(firsttimes,30, normed=True)
#ax2 = fig.add_axes([0.07,0.3,0.15,0.6], sharey=ax)
#ax2.hist(firstfreqs,30, orientation='horizontal',log=True, normed=True)
#plt.suptitle('Empirical distribution of the first atom')
#plt.savefig(op.join(figure_path, 'FirstAtom3D_%s_%dfiles_%dxMDCT.pdf'%(set_id,max_file_num,len(scales))))
#plt.savefig(op.join(figure_path, 'FirstAtom3D_%s_%dfiles_%dxMDCT.png'%(set_id,max_file_num,len(scales))))
#
#plt.figure()
#plt.hist([(2**int(f))/fs for f in firstscales],[f/fs for f in scales], normed=True)
#plt.title('Marginal empirical distribution over the scales')
#plt.show()

#times = [float(t) for t in times]
freqs = [float(f)*fs for f in freqs]
            
fig = plt.figure()
ax = fig.add_axes([0.3,0.3,0.65,0.65])
ax.scatter(times, freqs)
plt.xticks([])
plt.yticks([])
ax1 = fig.add_axes([0.3,0.10,0.65,0.15])
ax1.hist(times,200)
plt.xlabel('Time (s)',fontsize=16)
plt.xlim([0,5.5])
ax1.set_yticks([])
ax2 = fig.add_axes([0.12,0.3,0.15,0.65])
ax2.hist(freqs,100, orientation='horizontal', normed=True,log=True)
ax2.set_xticks([])
plt.ylabel('Frequency (Hz)',fontsize=16)
ax.set_xlim(ax1.get_xlim())
ax.set_ylim(ax2.get_ylim())
#plt.suptitle('Empirical distribution of all of the %d selected atom'%sparsity)
#plt.savefig(op.join(figure_path, 'All%dAtoms_%s_%dfiles_%dxMDCT.pdf'%(sparsity,set_id,max_file_num,len(scales))))
plt.savefig(op.join(figure_path, 'All%dAtoms_%s_%dfiles_%dxMDCT.png'%(sparsity,set_id,max_file_num,len(scales))))

plt.show()