'''
speech.compare_same_sentence_cortico  -  Created on Nov 27, 2013
@author: M. Moussallam
'''

import sys, os
from classes.sketches.cochleo import SKETCH_ROOT
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']

fileidx = 10
spk1path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_ksp_arctic')
spk1filenames = get_filepaths(spk1path, 0,  ext='wav')
spk2path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
spk2filenames = get_filepaths(spk2path, 0,  ext='wav')

def peakpick(arr):
    """ 2D peak picking """
    sp = np.ones(arr.shape, bool)
    alldims = range(len(arr.shape))
    for id in alldims:
        # compute the diff in the first axis after swaping
        d = np.diff(np.swapaxes(np.abs(arr), 0, id), axis=0)
        
        sp = np.swapaxes(sp, 0, id)
        sp[:-1,...] &= d < 0
        sp[1:,...] &= d > 0
        # swap back
        sp = np.swapaxes(sp, 0, id)
    
    # now sp contains indexes of local maxima
    sparr = np.zeros_like(arr)
    sparr[sp>0] = arr[sp>0]    
    return sparr

same = True
sameloc = False
sigspk1 = Signal(spk1filenames[fileidx], mono=True, normalize=True)
if not sameloc:
    sigspk2 = Signal(spk2filenames[fileidx - (not same)*1], mono=True, normalize=True)
else:
    sigspk2 = Signal(spk1filenames[fileidx - (not same)*1], mono=True, normalize=True)

# The sketch is a corticogram
skhandle = CorticoIHTSketch(**{'downsample':8000,'fs':8000,'f_width':50,
                               't_width':200,'frmlen':8,'shift':0,'fac':-2,'BP':1})
# The fgpthandle is a PairWise one
fgpthandle = CochleoPeaksBDB('CochleoPeaks.db', **{'wall':False})

sp_per_secs = 10
fs = 8000

scaleidx = 2
rateidx = 9

# Recompute and Sparsify        
sigspk1.resample(fs)
sigspk1.pad(2048)
sigspk2.resample(fs)
sigspk2.pad(2048)
# run the decomposition                        
skhandle.recompute(sigspk1)
skhandle.sparsify(int(sp_per_secs* sigspk1.get_duration()))
# populate

# sparsify even further by selecting only the local maxima
fgptspk1 = skhandle.sp_rep[scaleidx, rateidx, :,:]
#tpat1 = np.abs(fgptspk1).sum(axis=1)
#fpat1 = np.abs(fgptspk1).sum(axis=0)
## It is good to just select the peaks in the histograms and build a simplified 
## model based on that ? let us try
#tsparse = peakpick(tpat1).reshape((len(tpat1),1))
#fsparse = peakpick(fpat1).reshape((1,len(fpat1)))
sp_fgpt1 = peakpick(fgptspk1)
plt.figure()
plt.subplot(211)
keysspk1, valsspk1 = fgpthandle._build_pairs(fgptspk1,skhandle.params, 0,display=True, ax=plt.gca())

# run the decomposition                        
skhandle.recompute(sigspk2)
skhandle.sparsify(int(sp_per_secs* sigspk2.get_duration()))
# populate
fgptspk2 = skhandle.sp_rep[scaleidx, rateidx, :,:]
#tpat2 = np.abs(fgptspk2).sum(axis=1)
#fpat2 = np.abs(fgptspk2).sum(axis=0)
#tsparse = peakpick(tpat2).reshape((len(tpat2),1))
#fsparse = peakpick(fpat2).reshape((1,len(fpat2)))
sp_fgpt2 = peakpick(fgptspk2)
plt.subplot(212)
keysspk2, valsspk2 = fgpthandle._build_pairs(fgptspk2, skhandle.params,0,display=True, ax=plt.gca())
    
plt.show()

#    # QUANTIFIONS
#    if fgpthandle.__class__.__name__ == 'SparseFramePairsBDB':
#        Qf1 = 50
#        Qf = 5 
#        Qt = 0.01 
#        keysspk1Q = []
#        for key in keysspk1:
#            (f1,rf,rt) = key
#            f1 = Qf1*int(f1/Qf1)        
#            rf = Qf*int(rf/Qf)
#            rt = Qt*int(rt/Qt)
#            keysspk1Q.append((f1,rf,rt))
#        keysspk2Q = []
#        for key in keysspk2:
#            (f1,rf,rt) = key
#            f1 = Qf1*int(f1/Qf1)        
#            rf = Qf*int(rf/Qf)
#            rt = Qt*int(rt/Qt)
#            keysspk2Q.append((f1,rf,rt))
#    elif fgpthandle.__class__.__name__ == 'STFTPeaksTripletsBDB':
#        Qf1 = 1
#        Qf = 0.1 
#        Qt = 0.001 
#        keysspk1Q = []
#        for key in keysspk1:
#            (f1,rf,rt) = key
#            f1 = Qf1*int(f1/Qf1)        
#            rf = Qf*int(rf/Qf)
#            rt = Qt*int(rt/Qt)
#            keysspk1Q.append((f1,rf,rt))
#        keysspk2Q = []
#        for key in keysspk2:
#            (f1,rf,rt) = key
#            f1 = Qf1*int(f1/Qf1)        
#            rf = Qf*int(rf/Qf)
#            rt = Qt*int(rt/Qt)
#            keysspk2Q.append((f1,rf,rt))
#    elif fgpthandle.__class__.__name__ == 'CQTPeaksTripletsBDB':
#        Qf1 = 500
#        Qdf = 1 
#        Qt = 0.05
#        keysspk1Q = []
#        for key in keysspk1:
#            (f1,df1,df2,rt) = key
#            f1 = Qf1*int(f1/Qf1) +  Qf1/2       
#            df1 = Qdf*int(df1/Qdf) 
#            df1 = Qdf*int(df1/Qdf) 
#            rt = Qt*int(rt/Qt)
#            keysspk1Q.append((f1,df1,df2,rt))
#        keysspk2Q = []
#        for key in keysspk2:
#            (f1,df1,df2,rt) = key
#            f1 = Qf1*int(f1/Qf1)  +  Qf1/2   
#            df1 = Qdf*int(df1/Qdf) 
#            df1 = Qdf*int(df1/Qdf) 
#            rt = Qt*int(rt/Qt)
#            keysspk2Q.append((f1,df1,df2,rt))
#    else:
#        keysspk1Q = keysspk1
#        keysspk2Q = keysspk2
#    
#    ck_1 = []
#    ck_2 = []
#    # search loop
#    for (k,v,truek1) in zip(keysspk1Q,valsspk1,keysspk1):
#        if k in keysspk2Q:
#            sidx = keysspk2Q.index(k)
#            ck_1.append((k,v))
#            ck_2.append((k,valsspk2[sidx]))
#    
#    
#    plt.figure(figsize=(12,5))
#    plt.subplot(121)
#    sigspk1.spectrogram(2048,512,ax=plt.gca(),order=0.5,log=True,cbar=False,
#                         cmap=cm.pink_r, extent=[0,sigspk1.get_duration(),0, fs/2])
#    fgpthandle.draw_keys(zip(keysspk1,valsspk1), plt.gca(),'y')
#    fgpthandle.draw_keys(ck_1, plt.gca(),'k')
#    plt.subplot(122)
#    sigspk2.spectrogram(2048,512,ax=plt.gca(),order=0.5,log=True,cbar=False,
#                         cmap=cm.pink_r, extent=[0,sigspk2.get_duration(),0, fs/2])
#    
#    fgpthandle.draw_keys( zip(keysspk2,valsspk2), plt.gca(),'y')
#    fgpthandle.draw_keys(ck_2, plt.gca(),'k')
#    plt.suptitle("%s %d/%d keys in common"%(fgpthandle.__class__.__name__, len(ck_1),len(keysspk1Q)))
#    plt.subplots_adjust(left=0.07,right=0.95)
#    plt.savefig(op.join(SKETCH_ROOT,
#                        'src/reporting/figures/Compare_%s_%s_same_%d_%s.pdf'%(''+(sameloc) *'sameloc',
#                                                                              ''+(not same) *'not',
#                                                                                        fileidx,
#                                                                                        fgpthandle.__class__.__name__)))
#    
#plt.show()