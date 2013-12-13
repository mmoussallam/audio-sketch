'''
speech.sws_toy  -  Created on Dec 9, 2013
@author: M. Moussallam
'''


import sys, os
#from classes.sketches.cochleo import SKETCH_ROOT
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
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
plt.subplot(221)
cqtsk.recompute(reals[0])
cqtsk.represent(fig=plt.gcf())
plt.subplot(222)
cqtsk.recompute(reals[1])
cqtsk.represent(fig=plt.gcf())
plt.subplot(223)
cqtsk.recompute(recs[0])
cqtsk.represent(fig=plt.gcf())
plt.subplot(224)
cqtsk.recompute(recs[1])
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
