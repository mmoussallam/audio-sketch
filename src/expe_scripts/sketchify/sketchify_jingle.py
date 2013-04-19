'''
expe_scripts.sketchify.sketchify_jingle  -  Created on Apr 18, 2013
@author: M. Moussallam
'''
import os.path as op
from classes import sketch
import matplotlib.pyplot as plt
jingle_path = '/sons/jingles/'
output_path = '/home/manu/workspace/audio-sketch/src/expe_scripts/audio/sketches/'
jingle_list = ['panzani',]#'staracademy','rtl','cirque']
n_sparse = 1000

learned_base_dir = '/home/manu/workspace/audio-sketch/matlab/'

sketchifiers = [sketch.KNNSketch(**{'location':learned_base_dir,
                                                'shuffle':13,
                                                'n_frames':100000,
                                                'n_neighbs':1}),
                sketch.KNNSketch(**{'location':learned_base_dir,
                                                'shuffle':78,
                                                'n_frames':2000000,
                                                'n_neighbs':1}),
                sketch.KNNSketch(**{'location':learned_base_dir,
                                                'shuffle':87,
                                                'n_frames':100000,
                                                'n_neighbs':1}),
                sketch.SWSSketch(),
                sketch.CochleoPeaksSketch(),
#                        sketch.WaveletSparseSketch(**{'wavelets':[('db8',4),],
#                                                    'n_atoms':100,
#                                                    'nature':'Wavelet'}),
                        sketch.XMDCTSparseSketch(**{'scales':[128,1024,8192],
                                                    'n_atoms':100,
                                                    'nature':'LOMDCT'}),
#                        sketch.XMDCTSparseSketch(**{'scales':[128,1024,8192],
#                                                    'n_atoms':100,
#                                                    'nature':'MDCT'}),
                        sketch.STFTPeaksSketch(**{'scale':2048, 'step':256}),
                        sketch.STFTDumbPeaksSketch(**{'scale':2048, 'step':256}),              
                        ]
    
# for all sketches, we performe the same testing
for jingle in jingle_list:
    audio_path = op.join(jingle_path,jingle+'.wav')
    
    for sk in sketchifiers:
        print sk        
        
        sig_name = '%s_%s_%s_%d.wav'%(jingle,
                                    sk.__class__.__name__,
                                    sk.get_sig(),
                                    n_sparse)
        if op.exists(op.join(output_path,sig_name)):
            continue
        
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
        synth_sig.normalize()
#        plt.figure()
#        plt.subplot(211)
#        plt.plot(sk.orig_signal.data)
#        plt.subplot(212)
#        plt.plot(synth_sig.data)
        
    #            synth_sig.play()
        sig_name = '%s_%s_%s_%d.wav'%(jingle,
                                    sk.__class__.__name__,
                                    sk.get_sig(),
                                    n_sparse)
        print "Saving %s"%sig_name
        synth_sig.write(op.join(output_path,sig_name))
        
        del sk