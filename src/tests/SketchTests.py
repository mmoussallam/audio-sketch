'''
Created on Jan 31, 2013

@author: manu
'''
import unittest
import sys
import os
#sys.path.append('/home/manu/workspace/audio-sketch')
#sys.path.append('/home/manu/workspace/PyMP')
#sys.path.append('/home/manu/workspace/meeg_denoise')

SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)

import src.classes.sketches.base as base
import src.classes.sketches.bench as bench
import src.classes.sketches.misc as misc
import src.classes.sketches.cochleo as cochleo
import src.classes.sketches.cortico as cortico
from src.tools import cochleo_tools 
import matplotlib.pyplot as plt
import time
import os.path as op

#plt.switch_backend('Agg')
SND_DB_PATH = os.environ['SND_DB_PATH']
audio_test_file = os.path.join(SND_DB_PATH,'sqam/voicemale.wav')
#audio_test_file  = '/Users/loa-guest/Documents/Laure/libs/PyMP/data/ClocheB.wav'
#signal = Signal(son, normalize=True, mono=True)
class SketchTest(unittest.TestCase):

    def runTest(self):
                
        abstractSketch = base.AudioSketch()
        print abstractSketch
        self.assertRaises(NotImplementedError,abstractSketch.represent)
        self.assertRaises(NotImplementedError,abstractSketch.recompute)
        self.assertRaises(NotImplementedError,abstractSketch.sparsify, (0))
        self.assertRaises(NotImplementedError,abstractSketch.synthesize, ({}))
        
#        kwargs = {'scale':512, 'step':256}
#        stftpeaksketch = sketch.STFTPeaksSketch(**kwargs)
#        
#        kwargs = {'dico':[64,512,2048], 'n_atoms':100}
#        xmdctmpsketch = sketch.XMDCTSparseSketch()
        #learned_base_dir = '/home/manu/workspace/audio-sketch/matlab/'
        
        sketches_to_test = [
        
#                            misc.KNNSketch(**{'location':learned_base_dir,
#                                                'shuffle':87,
#                                                'n_frames':100000,
#                                                'n_neighbs':1}),
#                            misc.SWSSketch(),
                            #cortico.CorticoIHTSketch(**{'downsample':8000,'frmlen':8,'shift':0,'fac':-2,'BP':1,'max_iter':1,'n_inv_iter':5}),
                             #cochleo.CochleoIHTSketch(**{'downsample':8000,'frmlen':8,'shift':-1,'max_iter':5,'n_inv_iter':2}),
#                             cochleo.CochleoPeaksSketch(**{'fs':8000}),
#                             cortico.CorticoIndepSubPeaksSketch(**{'downsample':8000,'frmlen':8}),
#                             cortico.CorticoIHTSketch(**{'downsample':8000,'frmlen':8})
#                            cortico.CorticoIndepSubPeaksSketch(**{'downsample':8000,'frmlen':8,'shift':0,'fac':-2,'BP':1}),
                             #cortico.CorticoPeaksSketch(**{'downsample':8000,'frmlen':8,'shift':0,'fac':-2,'BP':1}),

                             #cortico.CorticoPeaksSketch(**{'n_octave':6,'freq_min':101.0, 'bins':24.0, 'downsample':8000, 'max_iter':5, 'rep_class': cochleo_tools.Quorticogram}),
                            cortico.CorticoSubPeaksSketch(**{'downsample':8000,
                                                             'sub_slice':(4,11),'n_inv_iter':10}),

                             cortico.CorticoPeaksSketch(**{'n_octave':6,'freq_min':101.0, 'bins':24.0, 'downsample':8000, 'max_iter':5, 'rep_class': cochleo_tools.Quorticogram}),
#                            cortico.CorticoSubPeaksSketch(**{'downsample':8000,
#                                                             'sub_slice':(4,11),'n_inv_iter':10}),

#                            cortico.CorticoSubPeaksSketch(**{'downsample':8000,
#                                                             'sub_slice':(0,11),'n_inv_iter':10}),
#                            cortico.CorticoSubPeaksSketch(**{'downsample':8000,
#                                                             'sub_slice':(4,6),'n_inv_iter':10}),
#                            cortico.CorticoSubPeaksSketch(**{'downsample':8000,
#                                                             'sub_slice':(4,11),'n_inv_iter':10})
#                                                     
#                            bench.XMDCTSparseSketch(**{'scales':[64,512,2048], 'n_atoms':100}),
#                           NOT FINISHED
#                           sketch.WaveletSparseSketch(**{'wavelets':[('db8',6),], 'n_atoms':100}),
                            #bench.STFTPeaksSketch(**{'scale':2048, 'step':256}),
                            #bench.STFTDumbPeaksSketch(**{'scale':2048, 'step':256}),  
#                             bench.CQTPeaksSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0}),    
#                             bench.cqtIHTSketch(**{'n_octave':5,'freq_min':101.0, 'bins':12.0, 'downsample':8000.0, 'max_iter':5})
                            ]
        
        # for all sketches, we performe the same testing
        for sk in sketches_to_test:
            
            print sk
            t = time.time()
            self.assertRaises(ValueError, sk.recompute)
            
            print "%s : compute full representation"%sk.__class__
            sk.recompute(audio_test_file)
            
            print "%s : plot the computed full representation" %sk.__class__
            sk.represent()
            
            print "%s : Now sparsify with 1000 elements"%sk.__class__
            sk.sparsify(1000)                    
#            
#            # Remove the original signal
##            sk.orig_signal = None 
#            
#            print "%s : Synthesize the sketch"%sk.__class__
#            print "temps:",time.time()-t
            
            # Remove the original signal
#            sk.orig_signal = None 
            
            print "%s : Synthesize the sketch"%sk.__class__
            synth_sig = sk.synthesize(sparse=True)
            
            #plt.figure()
#            plt.subplot(211)
#            plt.plot(sk.orig_signal.data)
#            plt.subplot(212)
            #plt.plot(synth_sig.data)
            
            synth_sig.normalize()
#            synth_sig.play()
            synth_sig.write('Test_%s_%s.wav'%(sk.__class__.__name__,sk.get_sig()))

if __name__ == "__main__":
    
    suite = unittest.TestSuite()

    suite.addTest(SketchTest())

    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.savefig(op.join(SKETCH_ROOT,'Quortico/test.pdf'))
    plt.show()
    plt.savefig(op.join(SKETCH_ROOT,'Quortico/test2.pdf'))
    
