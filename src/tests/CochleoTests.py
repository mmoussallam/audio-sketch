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
import time
import cProfile
from scipy.io import loadmat
from scipy.signal import hann, lfilter
filter_coeffs_path = '/home/manu/workspace/recup_angelique/Sketches/sketches/nsltools/aud24.mat'

class staticMethodTest(unittest.TestCase):

    def runTest(self):
        
        data = hann(1024)*np.random.random(1024,)
        d = loadmat(filter_coeffs_path)
        coeffs = d['COCHBA'] 
        
        p = int(coeffs[0, 10].real)
        b  = coeffs[range(1,p+2), chan_idx].real
        a  = coeffs[range(1,p+2), chan_idx].imag
        
        filt_coeff = lfilter() 
        
        # filtering
        filt_data = np.array([ cochleo_tools.coch_filt(data, coeffs, ch) for ch in range(coeffs.shape[1])])
                
        
        rec_data = cochleo_tools.inv_coch_filt(np.sum(filt_data,axis=0), coeffs, 0)
        
#        rec_data = [cochleo_tools.inv_coch_filt(filt_data[ch], coeffs, ch) for ch in rec_channels]),axis=0)
#        
#        print rec_data.shape
        
        
        plt.figure()
        plt.plot(data)
        plt.plot(rec_data,'r:')
        plt.show()
        
        
        

class CochleoTest(unittest.TestCase):


    def runTest(self):
        sig = Signal(audio_test_file)
        
        
        gram = cochleo_tools.cochleogram(sig.data, load_coch_filt=True)
        
        
        
#        cProfile.runctx('gram._toaud()', globals(), locals())
#        t0 = time.time()
#        gram._toy1()
#        print "%1.3f elapsed " % (time.time() - t0)
        
        
#        aud = cochleo_tools.wav2aud(sig.data, sig.fs)
        
        " Ok this is working"
#        aud, duration = cochleo_tools.auditory_spectrum(audio_test_file)
#
#        cochleo_tools.plot_auditory(aud, duration)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()