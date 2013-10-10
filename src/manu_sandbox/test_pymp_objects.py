'''
manu_sandbox.test_pymp_objects  -  Created on Oct 10, 2013
@author: M. Moussallam
'''

# primary testing: I want dictionaries and routine that will give me
# sparse approximation where the selection is penalized by a co-occurrence matrix
# that I provide 

import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
set_id = 'GTZAN' # Choose a unique identifier for the dataset considered
audio_path,ext = bases[set_id]
file_names = get_filepaths(audio_path, 0,  ext=ext)
                           
from PyMP.mdct.dico import Dico, SpreadDico
from PyMP.mp import greedy
from src.manu_sandbox.pymp_objects import PenalizedMDCTDico
                        
exemp_sig = Signal(str(file_names[0]), mono=True, normalize=True)
exemp_sig.crop(0, 4*8192)
exemp_sig.pad(4*8192)
# Standard decomposition
scales = [128,1024,8192]
#exemp_sig = Signal(np.random.randn(128), 20, mono=True)
#scales = [16,64] 
nb_atoms = 100
l_lambda = 0.0

# let us put some 1/f biais
biaises = []
Ws = []
for s in scales:
#    biaises.append(np.ones((s,)))
#    biaises.append(1.0/np.arange(1.,float(s)/2))    
#    biais = np.maximum(0.001, np.sqrt(1.0/np.linspace(1.,float(s)/2, s/2)))
    biais = np.maximum(0.001, np.linspace(1, 0.0,s/2))
#    biais = np.zeros((s/2,))
    biaises.append(biais)
    W = (2.0/float(nb_atoms))*np.eye(s/2,s/2)
    W = np.zeros((s/2,s/2))
    Ws.append(W)

std_dico = Dico(scales)
pen_dico = PenalizedMDCTDico(scales, biaises, Ws,
                             len(scales)*[l_lambda])
spread_dico = SpreadDico(scales, penalty=0, maskSize=3)


std_app , std_dec, = greedy(exemp_sig, std_dico, 100, nb_atoms, debug=1, pad=False)
pen_app , pen_dec = greedy(exemp_sig, pen_dico, 100, nb_atoms, debug=1, pad=False)
spr_app , spr_dec = greedy(exemp_sig, spread_dico, 100, nb_atoms, debug=1, pad=False)

# tell us when atoms stops being alike
t = 0
for atomA, atomB in zip(std_app.atoms, pen_app.atoms):
    if atomA == atomB:
        t += 1
    else:
        break

print "%d out of %d atoms are similar"%(t, std_app.atom_number)
print std_app
print pen_app
print spr_app

plt.figure()
plt.plot(np.abs(pen_dico.blocks[1].projs_matrix))
plt.plot(pen_dico.blocks[1].pen_mask,'r')
plt.plot(np.abs(pen_dico.blocks[1].projs_matrix) - l_lambda* pen_dico.blocks[1].pen_mask,'k--')
#plt.show()

plt.figure()
plt.subplot(221)
std_app.plot_tf()
plt.subplot(222)
pen_app.plot_tf()
plt.subplot(223)
spr_app.plot_tf()
plt.subplot(224)
plt.plot(std_dec)
plt.plot(pen_dec,'k--')
plt.plot(spr_dec,'r-.')
plt.show()


