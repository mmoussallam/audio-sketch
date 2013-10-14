'''
manu_sandbox.test_sketches  -  Created on Oct 14, 2013
@author: M. Moussallam
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']
single_test_file1 = op.join(SND_DB_PATH,'jingles/panzani.wav')
fs = 8000
tempdir = '.'
sparsity = 20
# let's take a signal and build the fingerprint with pairs of atoms or plain atoms

orig_sig = Signal(single_test_file1, normalize=True, mono=True)
orig_sig.downsample(fs)
orig_sig.pad(4096)
orig_sig.write(op.join(tempdir, 'orig.wav'))

plt.figure()
#case 1
plt.subplot(221)
ax1 = plt.gca()
fgpthandle = SparseFramePairsBDB('xMdctPairs.db',
                                 load=False,**{'wall':False,'nb_neighbors_max':5,
                                               'delta_t_max':3.0})
skhandle = XMDCTSparsePairsSketch(**{'scales':[1024, 4096],'n_atoms':1,
                                 'nature':'LOMDCT','pad':False})

# do what you have to do
skhandle.recompute(op.join(tempdir, 'orig.wav'))
skhandle.sparsify(sparsity)
fgpt = skhandle.fgpt(sparse=True)
fgpthandle.populate(fgpt, skhandle.params, 0, display=True,ax=plt.gca())
anchor = np.sum(fgpthandle.retrieve(fgpt, skhandle.params, 0,nbCandidates=1))
# and do the same under noisy conditions
noisy = Signal(orig_sig.data + 0.01*np.random.randn(len(orig_sig.data)), orig_sig.fs, normalize=False)      
skhandle.recompute(noisy)
skhandle.sparsify(sparsity)
noisy_fgpt = skhandle.fgpt(sparse=True)
plt.subplot(222)
ax2 = plt.gca()
target = np.sum(fgpthandle.retrieve(noisy_fgpt, skhandle.params, 0,nbCandidates=1))
fgpthandle.populate(noisy_fgpt, skhandle.params, 0, display=True,ax=plt.gca())                 

keys, values = fgpthandle._build_pairs(fgpt, skhandle.params, 0)
noisy_keys, noisy_values = fgpthandle._build_pairs(noisy_fgpt, skhandle.params, 0)

keys_not_in_noise = [(k,v) for k,v in zip(keys,values) if k not in noisy_keys]
fgpthandle.draw_keys(keys_not_in_noise, ax=ax1, color='r')

keys_not_in_orig = [(k,v) for k,v in zip(noisy_keys,noisy_values) if k not in keys]
#fgpthandle.draw_keys(keys_not_in_orig, ax=ax1, color='g')

# case 2 
plt.subplot(223)
fgpthandle2 = XMDCTBDB('xMdct.db',
                                 load=False,**{'wall':False,
                                               'delta_t_max':3.0})
skhandle2 = XMDCTSparseSketch(**{'scales':[1024, 4096],'n_atoms':1,
                                 'nature':'LOMDCT'})

# do what you have to do
skhandle2.recompute(op.join(tempdir, 'orig.wav'))
skhandle2.sparsify(sparsity)
fgpt = skhandle2.fgpt(sparse=True)
fgpthandle2.populate(fgpt, skhandle2.params, 0, display=True,ax=plt.gca())
anchor2 = np.sum(fgpthandle2.retrieve(fgpt, skhandle2.params, 0,nbCandidates=1))
# and do the same under noisy conditions
      
skhandle2.recompute(noisy)
skhandle2.sparsify(sparsity)
noisy_fgpt = skhandle2.fgpt(sparse=True)
plt.subplot(224)
target2 = np.sum(fgpthandle2.retrieve(noisy_fgpt, skhandle.params, 0,nbCandidates=1))
fgpthandle2.populate(noisy_fgpt, skhandle2.params, 0, display=True,ax=plt.gca())                 


plt.show()