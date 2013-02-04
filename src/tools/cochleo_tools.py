'''
tools.cochleo_tools  -  Created on Feb 1, 2013
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyMP import Signal
#import dear
from dear import spectrum
from dear.spectrum import cqt, dft, auditory, SpectrogramFile
from dear.spectrogram import plot_spectrogram
#cqt_spec = spectrum.CQTSpectrum(audio_test_file)
from scipy.io import loadmat
from scipy.signal import lfilter
import dear.io as io
decoder = io.get_decoder(name='audioread')
filter_coeffs_path = '/home/manu/workspace/recup_angelique/Sketches/sketches/nsltools/aud24.mat'

class cochleogram(object):
    ''' Class handling the cochleogram parameters and methods to build 
        and synthesize '''
    
    # parameters
    data = None
    coeffs = None
    n_chans = 128
    load_coch_filt = True
    fac = 1
    # different steps of auditory processing
    y1 = None
    y2 = None
    y3 = None
    y4 = None
    y5 = None
    
    # static filter coefficients
    coeffs = None    
    
    def __init__(self, data, load_coch_filt=True, n_chans=None, fac=None):
        
        self.data = data
        self.load_coch_filt = load_coch_filt
        if self.load_coch_filt:
            d = loadmat(filter_coeffs_path)
            self.coeffs = d['COCHBA'] 
        if n_chans is not None:
            self.n_chans = n_chans
        else:
            self.n_chans = self.coeffs.shape[1]
        
        if fac is not None:
            self.fac = fac
            
        
    def _toy1(self):
        ''' compute y1 from the raw audio data '''
        self.y1 = [coch_filt(self.data, self.coeffs, m) for m in range(self.n_chans-1,0,-1)]
        
    def _toy2(self):
        ''' Cochlear filtering + non linearity '''
        
        # first make sure filtering has been performed
        if self.y1 is None:
            self._toy1()
        
        self.y2 = sigmoid(self.y1, self.fac)

    def _toaud(self , shift=0, fac=-2, dec=8):
        ''' build the auditory spectrogram 
            This is largely inspired by Matlab NSL Toolbox
            but transcribed in a more pythonic fashion 
                        
            '''
        self.fac = fac
        # octave shift, nonlinear factor, frame length, leaky integration
        L_frm  = int(round(8 * 2**(4+shift)))    # frame length (points)
    
        if dec>0:
            alph    = np.exp(-1.0/(float(dec)*2.0**(4.0+shift)))    # decaying factor
        else:
            alph    = 0;                    # short-term avg.
        
        
        # hair cell time constant in ms
        haircell_tc = 0.5;
        beta = np.exp(-1.0/(haircell_tc*2**(4+shift)))
        
        # compute number of frame
        n_frames = int(np.ceil(self.data.shape[0] / L_frm))
        # if needed pad with zeroes        
        if self.data.shape[0] < n_frames * L_frm:
            self.data = np.concatenate((self.data,
                                        np.zeros(((n_frames * L_frm)- self.data.shape[0],))))

        # initialize large containers
        self.y2 = []
        self.y4 = []
        self.y3 = []
        self.y5 = []            
                
        self.y2.append(sigmoid(coch_filt(self.data, self.coeffs, self.n_chans-1), self.fac))
        
        # hair cell membrane (low-pass <= 4 kHz); ignored for LINEAR ionic channels
        if not (fac == -2):
            self.y2[0] = lfilter(1, [1 -beta], self.y2[0])
        y2_h = self.y2[0];        

        zer_array = np.zeros_like(y2_h)
        
        # apply cochlear filter to all channels 
        self.y2.extend([sigmoid(coch_filt(self.data, self.coeffs, ch), self.fac) for ch in range(self.n_chans-2,0,-1)])
        
        # non-linearity if needed
        if not (fac == -2):
            self.y2 = [lfilter([1.0], [1.0 -beta], self.y2[c]) for c in range(self.n_chans-2,0,-1)]            
        
        # Inhibition and thresholding step: TODO optimize
        for ch in range(self.n_chans-2,0,-1):        
        
            # masked by higher (frequency) spatial response
            self.y3.append(self.y2[ch] - y2_h)
            y2_h = self.y2[ch];
        
#            % spatial smoother ---> y3 (ignored)
#            %y3s = y3 + y3_h; 
#            %y3_h = y3;
        
            # half-wave rectifier ---> y4
            self.y4.append(np.maximum(self.y3[-1], zer_array))
        
            # temporal integration window ---> y5
            if alph:    # leaky integration
                self.y5.append(lfilter([1.0], [1.0, -alph], self.y4[-1])[range(0,L_frm*n_frames,L_frm)])
            else:        # short-term average
                if (L_frm == 1):
                    self.y5.append(self.y4[-1])
                else:                    
                    print 
                    self.y5.append(np.mean(self.y4[-1].reshape((L_frm, n_frames)),axis=0))

    def invert_y2(self, shift=0, dec=8):
        ''' recompute the waveform from the auditory spectrum 
            Suppose that we still have y2 available '''
        assert (self.y2 is not None)
        y2_h = self.y2[-1]
        L_frm  = int(round(8 * 2**(4+shift)))    # frame length (points)
        n_frames = int(np.ceil(self.data.shape[0] / L_frm))
        
        x_rec = np.zeros_like(self.data)
        
        for ch in range(self.coeffs.shape[1]-2):
            x_rec += inv_coch_filt(self.y2[ch], self.coeffs, ch)
        
        return x_rec
    
    def invert(self, shift=0, dec=8):
        ''' recompute the waveform from the auditory spectrum 
            Suppose that we still have an exact y2, y3 and y4 available '''
        
        assert (self.y4 is not None)
        assert (self.y3 is not None)
        assert (self.y2 is not None)
        assert (self.y5 is not None)
        
        (l,M) = self.coeffs.shape        
        
        # extract parameters             
        L_frm  = int(round(8 * 2**(4+shift)))    # frame length (points)
        n_frames = int(np.ceil(self.data.shape[0] / L_frm))
        if dec>0:
            alph    = np.exp(-1.0/(float(dec)*2.0**(4.0+shift)))    # decaying factor
        else:
            alph    = 0  # short-term avg.
        
        y2_h = self.y2[-1]
        
        for ch in range(M-2):
            print ch
            y5 = np.zeros((self.data.shape[0],));
            
            y5[0:L_frm]=self.y4[ch][0:L_frm]
            
            y5[range(0,L_frm*n_frames,L_frm)]=self.y5[ch]
            
            y4 = lfilter([1.0, -alph],[1.0],y5);
            y4[y4==0] = self.y3[ch][y4==0];
            
            y3=y4;
            
            y2=y3+y2_h;
            y2_h=y2;
        
        
        p    = int(self.coeffs[0, 0].real)
        B    = self.coeffs[range(1,p+2), 0].real
        A    = self.coeffs[range(1,p+2), 0].imag
        return lfilter(A, B, y2);
        

def inv_coch_filt(data, coeffs, chan_idx):
    p  = int(coeffs[0, chan_idx].real)    # order of ARMA filter
    b  = coeffs[range(1,p+2), chan_idx].real    # moving average coefficients
    a  = coeffs[range(1,p+2), chan_idx].imag    # autoregressive coefficients
    
    return lfilter(a, b, data)        
        
    
def coch_filt(data, coeffs, chan_idx):
    p  = int(coeffs[0, chan_idx].real)    # order of ARMA filter
    b  = coeffs[range(1,p+2), chan_idx].real    # moving average coefficients
    a  = coeffs[range(1,p+2), chan_idx].imag    # autoregressive coefficients
    
    return lfilter(b, a, data)
         
def sigmoid(y1, fac):
    ''' SIGMOID nonlinear funcion for cochlear model
    %    y = sigmoid(y, fac);
    %    fac: non-linear factor
    %     -- fac > 0, transister-like function
    %     -- fac = 0, hard-limiter
    %     -- fac = -1, half-wave rectifier
    %     -- else, no operation, i.e., linear 
    %
    %    SIGMOID is a monotonic increasing function which simulates 
    %    hair cell nonlinearity. 
    %    See also: WAV2AUD, AUD2WAV
     
    % Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
    % v1.00: 01-Jun-97
    
    Transcribed in Python by M. Moussallam (manuel.moussallam@espci.fr)
    '''

    if fac > 0:        
        y = np.exp(-y1/fac)
        return  1.0/(1.0+y)
    elif fac == 0:
        return (y > 0)    # hard-limiter
    elif fac == -1:
        y = np.zeros_like(y1)
        y[y1>0]= y1[y1>0]
        return y   # half-wave rectifier
    else:
        return y1 # do nothing
      

def auditory_spectrum(audio_file_path):
    ''' computes an auditory spectrogram based from an acoustic array '''
        
    audio = decoder.Audio(audio_file_path)
    st = 0
    graph = 'Y5'
    N = 64
    win = 0.025
    hop = 0.010
    freqs = [110., 2*4435.]
    combine = False
    spec = [[]]
    
    gram = getattr(auditory,graph)
    gram = gram(audio)
    
    for t, freq in enumerate(gram.walk(N=N, freq_base=freqs[0], freq_max=freqs[1],
                        start=st, end=None, combine=combine, twin=win, thop=hop)):
    
        spec[0].append(freq)
                
    return np.array(spec), audio._duration

def plot_auditory(aud_spec, duration,  freqs=[110., 2*4435.]):
    
    N = aud_spec.shape[2]
    gram = getattr(auditory,'Y5')
    f_vec = (gram.erb_space(N/10, freqs[0], freqs[1])).astype(int)
    y = np.linspace(0, aud_spec.shape[2], f_vec.shape[0])
    
    t_vec = np.arange(0, duration, 0.5)
    
    
    plt.figure()
    plt.imshow(aud_spec[0,:,:].T, aspect='auto',
               cmap=cm.jet,
               origin='lower')
    plt.yticks(np.flipud(y), f_vec)
    plt.xticks(t_vec*aud_spec.shape[1]/t_vec.max(), t_vec)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
#    plt.colorbar()
    plt.show()
    
    
    
def wav2aud(audio_file_path):
    
    sig = Signal(audio_file_path)
    