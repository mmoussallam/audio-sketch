'''
ghost_eval.expe_xmdctsparse  -  Created on Dec 5, 2013
@author: M. Moussallam

Let us evaluate the capacity of the Xmdct based fingerprint to 
correctly detect sketches 
'''

import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
from src.ghost_eval.expe_tools import *

set_id = 'voxforge' 
nbfiles = 100
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)[:nbfiles]

###### SCRIPT SPECIFIC WE ARE HERE INTERESTED IN THIS TYPE OF FINGERPRINT
# FGPT PARAMETERS
scales = [64,256,1024]
nature = 'MDCT'
fgptsparsity = 0.01 # ratio of sparse elements
sparsifier = XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,'fs':8000,
                                 'nature':nature,'pad':False,'downsample':8000})

fgpthandle = SparseFramePairsBDB('SparseMPPairs.db',load=False,**{'wall':False,
                                                                  'f1_n_bits':6,
                                                              'nb_neighbors_max':3,
                                                              'delta_f_bits':6,
                                                              'delta_f_min':250,
                                                              'delta_f_max':2000,
                                                              'delta_t_min':0.5, 
                                                              'delta_t_max':2.0})


##### FIRST PHASE LEARN THE BASE
learn((sparsifier,fgpthandle),file_names, fgptsparsity)


##### SECOND PHASE TEST WITH VARIOUS SKETCHES 
test_ratio = 0.1

sketches_to_test = [
             STFTPeaksSketch(**{'scale':2048, 'step':256,'downsample':8000}),
#             STFTDumbPeaksSketch(**{'scale':2048, 'step':256}),  
             CochleoIHTSketch(**{'downsample':8000,'frmlen':8,'shift':-1,'max_iter':4,'n_inv_iter':5}),
             XMDCTSparsePairsSketch(**{'scales':scales,'n_atoms':1,'fs':8000,'nature':nature,'pad':False,'downsample':8000}),
             CQTPeaksSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0}),    
             cqtIHTSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0, 'max_iter':5}),            
             ]

sparsities_to_test = [0.001, 0.005,0.01]
dist2second = np.zeros((len(sketches_to_test), len(sparsities_to_test), test_ratio*len(file_names)))
sklegends = []
for skidx, sketchifier in enumerate(sketches_to_test):
    sklegends.append(sketchifier.__class__.__name__)
    print sklegends[-1]
    for spidx, sparsity in enumerate(sparsities_to_test):
        print sparsity
        dist2second[skidx, spidx,:] = testratio(sketchifier,
                                                      (sparsifier,fgpthandle),
                                                      file_names, test_ratio, sparsity, fgptsparsity)
        

plt.figure()
plt.plot(sparsities_to_test,np.mean(dist2second, axis=2).T)
plt.legend(sklegends, loc='lower right')
plt.grid()
plt.show()