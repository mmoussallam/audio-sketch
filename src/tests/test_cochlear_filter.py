
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cProfile
from PyMP import Signal
import sys
sys.path.append('/home/manu/workspace/audio-sketch')
sys.path.append('/home/manu/workspace/PyMP')
sys.path.append('/home/manu/workspace/meeg_denoise')
from src.tools import cochleo_tools
#from classes import sketch
import matplotlib.pyplot as plt
from PyMP import Signal
from scipy.signal import lfilter, hann
plt.switch_backend('Agg')
audio_test_file = '/home/manu/workspace/recup_angelique/Sketches/NLS Toolbox/Hand-made Toolbox/forAngelique/61_sadness.wav'

#audio_test_file = '/sons/tests/Bach_prelude_4s.wav'

sig = Signal(audio_test_file, mono=True, normalize=True)
sig.downsample(8000)

#gram = cochleo_tools.cochleogram(sig.data)
#
#print float(sig.length) /float(sig.fs)
#import time
#t0 = time.time()
#gram._toaud()
#print "%1.3f elapsed " % (time.time() - t0)
#import cProfile
#cProfile.runctx('gram._toaud()',globals(), locals())
from scipy.io import loadmat
from numpy.fft import fft, ifft
filter_coeffs_path = '/home/manu/workspace/recup_angelique/Sketches/sketches/nsltools/aud24.mat'
#L = 8192
data = sig.data
import time
params = {'shift':-1,
          'dec':4,
          'frmlen':4}
gram = cochleo_tools.Cochleogram(sig.data, load_coch_filt=True, **params)
gram.build_aud()

#t = time.clock()
init_rec_data = gram.init_inverse()
rec_data = gram.invert(init_vec = init_rec_data, nb_iter=10, display=False)
#print "Elapsed :", time.clock() -t

cProfile.runctx('gram.invert(init_vec = init_rec_data, nb_iter=10, display=False)',
                globals(), locals())

rec_sig = Signal(rec_data, sig.fs, normalize=True)
#rec_sig.write('/sons/resynth_aud_python_10.wav')
#plt.figure()
#plt.plot(rec_data)
#plt.show()
rec_sig.play()
def debug_invert():
    # initialize
    dec = 8.0
    shift=0.0
    alph = np.exp(-1.0 / (float(dec) * 2.0 ** (4.0 + shift))) 
    x0 = init_rec_data
    
    #sig_init = Signal(x0, sig.fs)
    
    v5 = np.array(gram.y5).copy()
    
    # filterbank
    y2 = cochleo_tools.coch_filt(x0, gram.coeffs, 127)
    
    y2_h =  cochleo_tools.coch_filt(x0, gram.coeffs, 128)
    
    # diff and half wave rectifier
    y3 = y2 - y2_h            # difference (L-H)
    y4 = np.maximum(y3, 0)        # half-wave rect.
    
    plt.figure()
    plt.plot(y2_h)
    plt.show()
    
    L_frm = 128
    n_frames = v5.shape[0]
    # low-pass filter
    y5 = lfilter([1.0], [1.0, - alph], y4)  # leaky integ.
    vx = y5[range(0, L_frm * n_frames, L_frm)]    # new aud. spec.
    
    
    #v5_new[ch, :] = vx
    vt = v5.copy()
    ch = 126
    # matching
    s = np.ones((n_frames, 1))
    for n in range(n_frames):
        # scaling vector
        if vx[n]:
            s[n] = vt[ch, n] / vx[n]
        elif vt[ch, n]:
            s[n] = 2   # double it for now
        else:
            s[n] = 1
    
    #?? hard scaling TODO Refactoring
    s = np.multiply(s, np.ones((1, L_frm)))
    s = s.ravel()
    #                print "repeating s :", s.shape
    
    
    if (fac == -2):            # linear hair cell
    #                    print y3.shape, s.shape
        dy = y3
        y1 = dy * s
    
    else:                # nonlinear hair cell
        ind = (y3 >= 0)
        y1[ind] = y3[ind] * s[ind]
        maxy1p = y1[ind].max()
        ind = (y3 < 0)
        y1[ind] = y3[ind] / np.abs(y3[ind].min()) * maxy1p
    
    y1_h = y1
    y2_h = y2
    
    # inverse wavelet transform
    y_cum += np.flipud(cochleo_tools.coch_filt(np.flipud(y1), self.coeffs, ch)) / NORM




#data = np.concatenate((np.concatenate((np.zeros(L),hann(2*L)*data)), np.zeros(L)))

#data -= np.mean(data)
#d = loadmat(filter_coeffs_path)
#
#coeffs = d['COCHBA'] 
#y1 = np.zeros((data.shape[0], coeffs.shape[1]-1))
#y1_inv = np.zeros((data.shape[0], coeffs.shape[1]-1))
#rec = np.zeros_like(data)
#
#for ch in range(coeffs.shape[1]-2,0,-1):
#    NORM = coeffs[0,-1].imag
#    p = int(coeffs[0, ch].real)
#    b  = coeffs[range(1,p+2), ch].real
#    a  = coeffs[range(1,p+2), ch].imag    
#    y1[:, ch] = lfilter(b,a,data)
#    y1_inv[:, ch] = lfilter(b,a,np.flipud(y1[:, ch]))
#    rec += np.flipud(y1_inv[:, ch])/NORM
#    
#rec_sig = Signal(rec, sig.fs, normalize=True)
#rec_sig.write('/sons/synth_python.wav') 
#
#plt.figure()
#plt.plot(rec_sig.data[0:2048])
#plt.plot(sig.data[0:2048],'r:', linewidth=2.0)
#plt.show()


# inverting the channels


#for i in range(coeffs.shape[1]):
#    p = int(coeffs[0, i].real)
#    
#    b  = coeffs[range(1,p+1), i].real
#    a  = coeffs[range(1,p+2), i].imag
#    inv_filts.append(lfilter(b,a,np.flipud(filt_coeffs[i])))
#    rec += np.flipud(inv_filts[-1])/NORM
#rec = np.mean(np.array(inv_filts),axis = 0) 

#rec = np.flipud(np.mean(np.array(inv_filts), axis=0))

#plt.figure()
##plt.subplot(211)
#plt.plot(rec/rec.max())
##plt.subplot(212)
#plt.plot(data,'r:')
#plt.show()
#
#plt.figure()
#plt.imshow(np.fliplr(np.array(filt_coeffs)),
#           aspect='auto')
#plt.show()

#
#plt.figure()
#plt.plot(filt_coeff)
#plt.show()

#rec_data = lfilter(a,b,np.flipud(filt_coeff))
#
#
#plt.figure()
#plt.plot(data)
#plt.plot(rec_data , 'r:')
#plt.show()

# Ok that's ok now invert
#plt.figure()
#plt.plot(sig.data,'b:')
#plt.plot(gram.invert(), 'r')
#plt.show()

#plt.figure()
#plt.imshow(np.array(gram.y5),
#           aspect='auto',
#           origin='lower',
#           interpolation='nearest');
#plt.colorbar()
#plt.show()        
#gram._toy1()


#A[np.isinf(A)] = 0
#A[np.isnan(A)] = 0
#plt.plot(A[10,:])
#plt.show()

#gram._toy2()


#
#from scipy.signal import lfilter
#from scipy.io import loadmat
#filter_coeffs_path = '/home/manu/workspace/recup_angelique/Sketches/sketches/nsltools/aud24.mat'
#d = loadmat(filter_coeffs_path)
#
#data = sig.data
#
#coeffs = d['COCHBA']
#
#def coch_filt(data, coeffs, chan_idx):
#    p  = coeffs[0, chan_idx].real    # order of ARMA filter
#    b  = coeffs[1:p+1, chan_idx].real    # moving average coefficients
#    a  = coeffs[1:p+1, chan_idx].imag    # autoregressive coefficients
#    
#    return lfilter(b, a, data); 
#
#(L, M) = coeffs.shape
#L_x = data.shape[0]
#
#import time
#y1 = np.zeros((L_x, M))
#t0 = time.time()
#for m in range(M):
#
#    p  = coeffs[0, m].real    # order of ARMA filter
#    b  = coeffs[1:p+1, m].real    # moving average coefficients
#    a  = coeffs[1:p+1, m].imag    # autoregressive coefficients
#    
#    y1[:, m] = lfilter(b, a, data); 
#
#print "%1.3f elapsed " % (time.time() - t0)
#
#
#
#
#t1 = time.time()
#y1_bis = [coch_filt(data, coeffs, m) for m in range(M)]
#
#print "%1.3f elapsed " % (time.time() - t1)

#import cProfile
#cProfile.runctx('[coch_filt(data, coeffs, m) for m in range(M)]',
#                globals(), locals())




#y1[np.isinf(y1)] = 0
#y1[np.isnan(y1)] = 0
#
#plt.figure()
#plt.imshow(np.abs(y1[:,1:-1].T), aspect='auto')
#plt.show()
