'''
expe_scripts.sketchify.sketchify_jingle  -  Created on Apr 18, 2013
@author: M. Moussallam
'''
import os.path as op
from classes import sketch
import matplotlib.pyplot as plt
jingle_path = '/sons/jingles/'
output_path = '/home/manu/workspace/audio-sketch/src/expe_scripts/audio/sketches/'
jingle_list = ['europe1','staracademy','rtl','cirque']
n_sparse = 2000
sketchifiers = [sketch.CochleoPeaksSketch,
            sketch.XMDCTSparseSketch,
            sketch.STFTPeaksSketch]

sketchifiers = [sketch.CochleoPeaksSketch(),
                        sketch.XMDCTSparseSketch(**{'scales':[64,512,2048], 'n_atoms':100}),
                        sketch.STFTPeaksSketch(**{'scale':2048, 'step':256}),
                        sketch.STFTDumbPeaksSketch(**{'scale':2048, 'step':256}),              
                        ]
    
# for all sketches, we performe the same testing
for jingle in jingle_list:
    audio_path = op.join(jingle_path,jingle+'.wav')
    
    for sk in sketchifiers:
        print sk        
        
        print " compute full representation"
        sk.recompute(audio_path)
        
        print " plot the computed full representation" 
        sk.represent()
        
        print " Now sparsify with n_sparse elements" 
        sk.sparsify(n_sparse)
        
        print " plot the sparsified representation"
        sk.represent(sparse=True)
        
        print " and synthesize the sketch"
        synth_sig = sk.synthesize(sparse=True)
        
        plt.figure()
        plt.subplot(211)
        plt.plot(sk.orig_signal.data)
        plt.subplot(212)
        plt.plot(synth_sig.data)
        
    #            synth_sig.play()
        synth_sig.write(op.join(output_path,'%s_%s_%d.wav'%(jingle,
                                                            sk.__class__.__name__,
                                                            n_sparse)))