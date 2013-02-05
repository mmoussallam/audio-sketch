'''
Created on Feb 1, 2013

@author: manu
'''
import numpy as np
import unittest
import sys
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')

#from classes import sketch
from tools import cochleo_tools
import matplotlib.pyplot as plt
from PyMP import Signal
plt.switch_backend('Agg')
audio_test_file = '/sons/tests/Bach_prelude_4s.wav'
audio_test_file = '/home/manu/workspace/recup_angelique/Sketches/NLS Toolbox/Hand-made Toolbox/forAngelique/61_sadness.wav'
import time
import cProfile
from scipy.io import loadmat
from scipy.signal import hann, lfilter
filter_coeffs_path = '/home/manu/workspace/recup_angelique/Sketches/sketches/nsltools/aud24.mat'

class staticMethodTest(unittest.TestCase):

    def runTest(self):
        
        sig = Signal(audio_test_file, mono=True, normalize=True)                
        
        gram = cochleo_tools.cochleogram(sig.data)
        
        gram._toy2()
        rec_data = gram.invert_y2()
        
#        rec_data = [cochleo_tools.inv_coch_filt(filt_data[ch], coeffs, ch) for ch in rec_channels]),axis=0)
#        
#        print rec_data.shape        
        
#        plt.figure()
#        plt.plot(sig.data)
#        plt.plot(rec_data,'r:')
#        plt.show()

        

class CochleoTest(unittest.TestCase):


    def runTest(self):
        
        # test bad call
        sig = Signal(audio_test_file)
        gram = cochleo_tools.cochleogram(sig.data, load_coch_filt=True)
        self.assertRaises(NotImplementedError, gram.build_aud)
        
        
        sig = Signal(audio_test_file, mono=True, normalize=True)                
        gram = cochleo_tools.cochleogram(sig.data, load_coch_filt=True)
        
        gram.build_aud()
        gram.plot_aud()
        
        
        t = time.clock()
        init_rec_data = gram.init_inverse()
        rec_data = gram.invert(init_vec = init_rec_data,
                               nb_iter=10, display=False)
        print "Elapsed :", time.clock() -t
        
        min_error = min([np.sum((rec_data - sig.data[0:rec_data.shape[0]])**2),
                           np.sum((rec_data + sig.data[0:rec_data.shape[0]])**2)])
                
#        plt.figure()
#        plt.plot(sig.data)
#        plt.plot(rec_data)
#        plt.plot(rec_data - sig.data[0:rec_data.shape[0]])
#        plt.show() 
        
#        self.assertGreater(sig.energy*0.1, min_error)

          
        
#        cProfile.runctx('gram.build_aud()', globals(), locals())
#        t0 = time.time()
#        gram._toy1()
#        print "%1.3f elapsed " % (time.time() - t0)
        
        
#        aud = cochleo_tools.wav2aud(sig.data, sig.fs)
        
#        " Ok this is working"
#        aud, duration = cochleo_tools.auditory_spectrum(audio_test_file)

#        cochleo_tools.plot_auditory(aud, duration)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()