'''
speech.sws_toy  -  Created on Dec 9, 2013
@author: M. Moussallam
'''

######## Attention chemins matlab #######

import sys, os
#from classes.sketches.cochleo import SKETCH_ROOT
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
import scipy.io
SND_DB_PATH = os.environ['SND_DB_PATH']
audio_outputpath = op.join(SKETCH_ROOT,'src/speech/sounds')

index = 50
spk1path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_rms_arctic')
spk1filename = get_filepaths(spk1path, 0,  ext='wav')[index]
spk2path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
spk2filename = get_filepaths(spk2path, 0,  ext='wav')[index]

spk3filename = get_filepaths(spk2path, 0,  ext='wav')[index+50]

files  = [spk1filename, spk2filename,  spk3filename]
L = len(files)

reals = []
recs = []
plt.figure()
for fidx, filename in enumerate(files):
    plt.subplot(3,1,fidx)
    sig = Signal(filename, mono=True)
    sig.normalize()
    reals.append(sig)
#    sig.downsample(16000)
    sk = SWSSketch(**{'n_formants': 3,'time_step':0.01,'windowSize': 0.025,})
    sk.recompute(filename)
    sk.represent(fig=plt.gcf())
    rec = sk.synthesize()
    rec.normalize()
    recs.append(rec)
    
#plt.show()
    
######################### second test: building a data set of 3 
    ##################### sentences ennonciate by 7 speakers  
# construct the database    
locuteurs = ['awb','bdl','clb']
label_names_loc = ['h1','h2','f']
path_loc = os.environ['SND_DB_PATH']+'/voxforge/main/Test/cmu_us_'
path_ph = '_arctic/wav/arctic_a000'
files = []
label=[]
nb_ph = 6
for i in np.arange(len(locuteurs)):    
  for j in np.arange(1,int(nb_ph+1)):
      files.append(path_loc+locuteurs[i]+path_ph+str(j)+'.wav')
      label.append(label_names_loc[i]+'ph'+str(j))
      
# do the sws 
val = []      
mat = []
cqt = []
cqt_sws = []
cqtsk = cqtsk = cqtIHTSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0})
for fidx, filename in enumerate(files):
    sig = Signal(filename, mono=True)
    sig.normalize()
    cqtsk.recompute(sig)
    cqt.append(cqtsk.rep[0,:,:])
    sk = SWSSketch(**{'n_formants': 3,'time_step':0.01,'windowSize': 0.025,})
    sk.recompute(filename)
    sws_sig = sk.synthesize()
    sws_sig.normalize()
    cqtsk.recompute(sws_sig)
    cqt_sws.append(cqtsk.rep[0,:,:])    
    val.append(sk.rep.shape[1])
    mat.append(sk.rep)

#### scaling time #######
minus = min(val)
np_mat = np.zeros((sk.params['n_formants'], minus , int(fidx+1)))
for indx, spl in enumerate(mat):
    tab_ind = np.ceil(linspace(0,val[indx]-1, minus)).astype(int)
    np_mat[:,:,indx] = spl[:,tab_ind]
    
#### save mat for classification matlab use #########
savemat('/Users/loa-guest/Documents/MATLAB/Classif/sws1.mat', mdict = {'np_mat':np_mat, 'label':label, 'mat':mat, 'cqt':cqt, 'cqt_sws': cqt_sws})

######################### increasing the number of formants
#recsbyformants = []
#sig = Signal(spk1filename, mono=True)
#sig.normalize()
#plt.figure()
##reals.append(sig)
#plt.subplot(6,1,1)
#sig.spectrogram(2 ** int(np.log2(sk.params['windowSize'] * sig.fs)),
#                2 ** int(np.log2(
#                         sk.params['time_step'] * sig.fs)),
#                order=1, log=True, ax=plt.gca(), cbar=False)
#for i in range(1,6):
#    plt.subplot(6,1,i+1)
##    sig.downsample(16000)
#    sk = SWSSketch(**{'n_formants': i,'n_formants_max': 7})
#    sk.recompute(spk1filename)
#    sk.represent(fig=plt.gcf())
#    rec = sk.synthesize()
#    rec.normalize()
#    recsbyformants.append(rec)
#plt.show()


############### Let us visualize the corticogram representations of two sws
#corticosk = CorticoPeaksSketch(**{'downsample':8000,'frmlen':8,'shift':0,'fac':-2,'BP':1})
#corticosk.recompute(reals[1])
#corticosk.represent()
#corticosk.recompute(recs[0])
#corticosk.represent()
#corticosk.recompute(recs[1])
#corticosk.represent()


############### Let us visualize the CQT representations of two sws
cqtsk = cqtIHTSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0})
#cqtsk = CQTPeaksSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0})
plt.figure()
plt.subplot(231)
cqtsk.recompute(reals[0])
cqtsk.represent(fig=plt.gcf())
plt.subplot(232)
cqtsk.recompute(reals[1])
cqtsk.represent(fig=plt.gcf())
plt.subplot(233)
cqtsk.recompute(recs[0])
cqtsk.represent(fig=plt.gcf())
plt.subplot(234)
cqtsk.recompute(recs[1])
cqtsk.represent(fig=plt.gcf())
plt.subplot(235)
cqtsk.recompute(recs[0]+real[0])
cqtsk.represent(fig=plt.gcf())
plt.subplot(236)
cqtsk.recompute(recs[1]+real[1])
cqtsk.represent(fig=plt.gcf())
plt.show()


cqtsk.recompute(reals[0])
cqtsk.represent()

cqtsk.recompute(recs[0])
cqtsk.represent()

####################"" OK so what about the fingerprint one could build?
#cqtsk.params['f_width']=12
#cqtsk.params['t_width']=50
cqtsk.recompute(recs[0])
cqtsk.sparsify(1000)
cqtfgpt = CQTPeaksBDB(None)
plt.figure()
plt.subplot(211)
plt.imshow(np.abs(cqtsk.fgpt()[0,:,:])>0)
#k1,v1 = cqtfgpt._build_pairs(cqtsk.fgpt(), cqtsk.params, 0, display=True, ax=plt.gca())
plt.subplot(212)
cqtsk.recompute(recs[1])
cqtsk.sparsify(1000)
plt.imshow(np.abs(cqtsk.fgpt()[0,:,:])>0)
#k2,v2 = cqtfgpt._build_pairs(cqtsk.fgpt(), cqtsk.params, 0, display=True, ax=plt.gca())
plt.show()
