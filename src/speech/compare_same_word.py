'''
speech.compare_same_word  -  Created on Nov 13, 2013
@author: M. Moussallam
'''
'''
speech.compare_same_sentence  -  Created on Nov 6, 2013
@author: M. Moussallam
Ok so we want to compare two fingerprints of two speaker uttering a single sentence
'''
import sys, os
#from classes.sketches.cochleo import SKETCH_ROOT
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']

word = 'danger'
fpath = op.join(SKETCH_ROOT,'src/speech/iso_words')

spk1filename = op.join(fpath,'bdl_author.wav')
spk2filename = op.join(fpath,'rms_author.wav')

same = True
sameloc = False
sigspk1 = Signal(spk1filename, mono=True, normalize=True)
sigspk2 = Signal(spk2filename, mono=True, normalize=True)

fgpt_sketches = [
#    (STFTPeaksTripletsBDB(None, **{'wall':False,'TZ_delta_f':-5,'TZ_delta_t':10}),
#     STFTPeaksSketch(**{'scale':2048, 'step':512})),
#    (CochleoPeaksBDB('CochleoPeaks.db', **{'wall':False}),
#    CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),  
#    (CQTPeaksBDB('CQTPeaks.db', **{'wall':False}),
#    CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000})),  
    (SparseFramePairsBDB(None, **{'wall':False,'nb_neighbors_max':10,
                                             'delta_t_max':0.2, 'f1_n_bits':8,'dt_n_bits':5}),
     XMDCTSparsePairsSketch(**{'scales':[64,512, 4096],'n_atoms':1,
                                 'nature':'LOMDCT','pad':False}))   
#     (CQTPeaksTripletsBDB(None, **{'wall':False,'f1_n_bits':5,
#                                   'dt_n_bits':8,
#                                   'TZ_delta_f':5,
#                                   'TZ_delta_t':10,
#                                   't_targ_width':100,
#                                   'f_targ_width':24}),
#     cqtIHTSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000}))                                  
                    ]
n_keypoints = 200
fs = 8000
for (fgpthandle, skhandle) in fgpt_sketches:
    plt.figure(figsize=(12,5))
    print "************************************"
    print fgpthandle
    print skhandle      
        
    sigspk1.resample(fs)
    sigspk1.pad(512)
    sigspk2.resample(fs)
    sigspk2.pad(512)
    # run the decomposition                        
    skhandle.recompute(sigspk1)
    skhandle.sparsify(n_keypoints)
    plt.subplot(121)
    skhandle.represent(sparse=False,fig=plt.gcf())
    skhandle.represent(sparse=True,fig=plt.gcf())
    # populate
    fgptspk1 = skhandle.fgpt()
    keysspk1, valsspk1 = fgpthandle._build_pairs(fgptspk1, skhandle.params,0)
    
    # run the decomposition                        
    skhandle.recompute(sigspk2)
    skhandle.sparsify(n_keypoints)
    plt.subplot(122)
    skhandle.represent(sparse=False,fig=plt.gcf())
    skhandle.represent(sparse=True,fig=plt.gcf())
    # populate
    fgptspk2 = skhandle.fgpt()
    keysspk2, valsspk2 = fgpthandle._build_pairs(fgptspk2, skhandle.params,0)
    

    # QUANTIFIONS
    if fgpthandle.__class__.__name__ == 'SparseFramePairsBDB':
        Qf1 = 50
        Qf = 30 
        Qt = 0.1 
        keysspk1Q = []
        for key in keysspk1:
            (f1,rf,rt) = key
            f1 = Qf1*int(f1/Qf1)        
            rf = Qf*int(rf/Qf)
            rt = Qt*int(rt/Qt)
            keysspk1Q.append((f1,rf,rt))
        keysspk2Q = []
        for key in keysspk2:
            (f1,rf,rt) = key
            f1 = Qf1*int(f1/Qf1)        
            rf = Qf*int(rf/Qf)
            rt = Qt*int(rt/Qt)
            keysspk2Q.append((f1,rf,rt))
    elif fgpthandle.__class__.__name__ == 'STFTPeaksTripletsBDB':
        Qf1 = 1
        Qf = 0.1 
        Qt = 0.001 
        keysspk1Q = []
        for key in keysspk1:
            (f1,rf,rt) = key
            f1 = Qf1*int(f1/Qf1)        
            rf = Qf*int(rf/Qf)
            rt = Qt*int(rt/Qt)
            keysspk1Q.append((f1,rf,rt))
        keysspk2Q = []
        for key in keysspk2:
            (f1,rf,rt) = key
            f1 = Qf1*int(f1/Qf1)        
            rf = Qf*int(rf/Qf)
            rt = Qt*int(rt/Qt)
            keysspk2Q.append((f1,rf,rt))
    elif fgpthandle.__class__.__name__ == 'CQTPeaksTripletsBDB':
        Qf1 = 1
        Qdf = 1 
        Qt = 0.05
        keysspk1Q = []
        for key in keysspk1:
            (f1,df1,df2,rt) = key
            f1 = Qf1*int(f1/Qf1) +  Qf1/2       
            df1 = Qdf*int(df1/Qdf) 
            df1 = Qdf*int(df1/Qdf) 
            rt = Qt*int(rt/Qt)
            keysspk1Q.append((f1,df1,df2,rt))
        keysspk2Q = []
        for key in keysspk2:
            (f1,df1,df2,rt) = key
            f1 = Qf1*int(f1/Qf1)  +  Qf1/2   
            df1 = Qdf*int(df1/Qdf) 
            df1 = Qdf*int(df1/Qdf) 
            rt = Qt*int(rt/Qt)
            keysspk2Q.append((f1,df1,df2,rt))
    else:
        keysspk1Q = keysspk1
        keysspk2Q = keysspk2
    
    ck_1 = []
    ck_2 = []
    # search loop
    for (k,v,truek1) in zip(keysspk1Q,valsspk1,keysspk1):
        if k in keysspk2Q:
            sidx = keysspk2Q.index(k)
            ck_1.append((k,v))
            ck_2.append((k,valsspk2[sidx]))
    
    
    
    plt.subplot(121)
#    sigspk1.spectrogram(1024,64,ax=plt.gca(),order=0.5,log=True,cbar=False,
#                         cmap=cm.pink_r, extent=[0,sigspk1.get_duration(),0, fs/2])
    fgpthandle.draw_keys(zip(keysspk1,valsspk1), plt.gca(),'y')
    fgpthandle.draw_keys(ck_1, plt.gca(),'k')
    plt.subplot(122)
#    synthsig2 = skhandle.synthesize(sparse=True)
#    sigspk2.spectrogram(1024,64,ax=plt.gca(),order=0.5,log=True,cbar=False,
#                         cmap=cm.pink_r, extent=[0,sigspk2.get_duration(),0, fs/2])
#    skhandle.represent(sparse=False,fig=plt.gcf())
#    skhandle.represent(sparse=True,fig=plt.gcf())
#    plt.yscale('log')
    fgpthandle.draw_keys( zip(keysspk2,valsspk2), plt.gca(),'y')
    fgpthandle.draw_keys(ck_2, plt.gca(),'k')
    plt.suptitle("%s %d/%d keys in common"%(fgpthandle.__class__.__name__, len(ck_1),len(keysspk1Q)))
    plt.subplots_adjust(left=0.07,right=0.95)
#    plt.savefig(op.join(SKETCH_ROOT,
#                        'src/reporting/figures/Compare_%s_%s_same_%d_%s.pdf'%(''+(sameloc) *'sameloc',
#                                                                              ''+(not same) *'not',
#                                                                                        fileidx,
#                                                                                        fgpthandle.__class__.__name__)))
    
plt.show()