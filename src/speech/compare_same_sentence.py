'''
speech.compare_same_sentence  -  Created on Nov 6, 2013
@author: M. Moussallam
Ok so we want to compare two fingerprints of two speaker uttering a single sentence
'''
import sys, os
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']

fileidx = 11
spk1path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_ksp_arctic')
spk1filenames = get_filepaths(spk1path, 0,  ext='wav')
spk2path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
spk2filenames = get_filepaths(spk2path, 0,  ext='wav')

sigspk1 = Signal(spk1filenames[fileidx], mono=True, normalize=True)
sigspk2 = Signal(spk2filenames[fileidx], mono=True, normalize=True)

fgpt_sketches = [
#    (STFTPeaksTripletsBDB(None, **{'wall':False,'TZ_delta_f':-5,'TZ_delta_t':10}),
#     STFTPeaksSketch(**{'scale':2048, 'step':512})),
#    (CochleoPeaksBDB('CochleoPeaks.db', **{'wall':False}),
#    CochleoPeaksSketch(**{'fs':8000,'step':128,'downsample':8000})),  
#    (SparseFramePairsBDB('xMdctPairs.db', **{'wall':False,'nb_neighbors_max':5,
#                                             'delta_t_max':3.0}),
#     XMDCTSparsePairsSketch(**{'scales':[64,512, 4096],'n_atoms':1,
#                                 'nature':'LOMDCT','pad':False}))   
     (CQTPeaksTripletsBDB(None, **{'wall':False}),
     CQTPeaksSketch(**{'n_octave':5,'freq_min':101, 'bins':12.0,'downsample':8000}))                                  
                    ]
sp_per_secs = 15
fs = 8000
for (fgpthandle, skhandle) in fgpt_sketches:
    print "************************************"
    print fgpthandle
    print skhandle      
        
    sigspk1.resample(fs)
    sigspk2.resample(fs)
    # run the decomposition                        
    skhandle.recompute(sigspk1)
    skhandle.sparsify(int(sp_per_secs* sigspk1.get_duration()))
    # populate
    fgptspk1 = skhandle.fgpt()
    keysspk1, valsspk1 = fgpthandle._build_pairs(fgptspk1, skhandle.params,0)
    
    # run the decomposition                        
    skhandle.recompute(sigspk2)
    skhandle.sparsify(int(sp_per_secs* sigspk2.get_duration()))
    # populate
    fgptspk2 = skhandle.fgpt()
    keysspk2, valsspk2 = fgpthandle._build_pairs(fgptspk2, skhandle.params,0)
    

    # QUANTIFIONS
    if fgpthandle.__class__.__name__ == 'STFTPeaksTripletsBDB':
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
        Qf1 = 100
        Qdf = 2 
        Qt = 0.1
        keysspk1Q = []
        for key in keysspk1:
            (f1,df1,df2,rt) = key
            f1 = Qf1*int(f1/Qf1)        
            df1 = Qdf*int(df1/Qdf) 
            df1 = Qdf*int(df1/Qdf) 
            rt = Qt*int(rt/Qt)
            keysspk1Q.append((f1,df1,df2,rt))
        keysspk2Q = []
        for key in keysspk2:
            (f1,df1,df2,rt) = key
            f1 = Qf1*int(f1/Qf1)        
            df1 = Qdf*int(df1/Qdf) 
            df1 = Qdf*int(df1/Qdf) 
            rt = Qt*int(rt/Qt)
            keysspk2Q.append((f1,df1,df2,rt))
    
    
    ck_1 = []
    ck_2 = []
    # search loop
    for (k,v) in zip(keysspk1Q,valsspk1):
        if k in keysspk2Q:
            sidx = keysspk2Q.index(k)
            ck_1.append((k,v))
            ck_2.append((k,valsspk2[sidx]))
    
    
    plt.figure()
    plt.subplot(121)
    sigspk1.spectrogram(2048,512,ax=plt.gca(),order=0.5,log=True,cbar=False,
                         cmap=cm.pink_r, extent=[0,sigspk1.get_duration(),0, fs/2])
    fgpthandle.draw_keys(zip(keysspk1,valsspk1), plt.gca(),'c')
    fgpthandle.draw_keys(ck_1, plt.gca(),'m')
    plt.subplot(122)
    sigspk2.spectrogram(2048,512,ax=plt.gca(),order=0.5,log=True,cbar=False,
                         cmap=cm.pink_r, extent=[0,sigspk2.get_duration(),0, fs/2])
    
    fgpthandle.draw_keys( zip(keysspk2,valsspk2), plt.gca(),'c')
    fgpthandle.draw_keys(ck_2, plt.gca(),'m')
    plt.suptitle("%s %d/%d keys in common"%(skhandle.__class__.__name__, len(ck_1),len(keysspk1Q)))
    
    
plt.show()