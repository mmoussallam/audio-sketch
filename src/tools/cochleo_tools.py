'''
tools.cochleo_tools  -  Created on Feb 1, 2013
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyMP import Signal
# import dear
from joblib import Parallel, delayed
from dear import spectrum
from dear.spectrum import cqt, dft, auditory, SpectrogramFile
from dear.spectrogram import plot_spectrogram
# cqt_spec = spectrum.CQTSpectrum(audio_test_file)
from scipy.io import loadmat
from scipy.signal import lfilter
from numpy.fft import fft, ifft
import dear.io as io
decoder = io.get_decoder(name='audioread')
filter_coeffs_path = '/home/manu/workspace/recup_angelique/Sketches/sketches/nsltools/aud24.mat'


class Cochleogram(object):
    ''' Class handling the Cochleogram parameters and methods to build
        and synthesize '''

    # parameters
#    data = None
#    coeffs = None
#    n_chans = 128
#    load_coch_filt = True
#    fac = -2
    # different steps of auditory processing


    # static filter coefficients
#    coeffs = None

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

        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None
        self.y5 = None

    def _toy1(self):
        ''' compute y1 from the raw audio data '''
        self.y1 = [coch_filt(
            self.data, self.coeffs, m) for m in range(self.n_chans - 1, 0, -1)]

    def _toy2(self, shift=0):
        ''' Cochlear filtering + non linearity '''
        haircell_tc = 0.5
        beta = np.exp(-1.0 / (haircell_tc * 2 ** (4 + shift)))

        self.y2 = []

        # first make sure filtering has been performed
        self.y2.append(sigmoid(
            coch_filt(self.data, self.coeffs, self.n_chans - 1), self.fac))

        # hair cell membrane (low-pass <= 4 kHz); ignored for LINEAR ionic
        # channels
        if not (self.fac == -2):
            self.y2[0] = lfilter(1, [1 - beta], self.y2[0])

        # apply cochlear filter to all channels
        self.y2.extend([sigmoid(coch_filt(self.data, self.coeffs,
                       ch), self.fac) for ch in range(self.n_chans - 2, 0, -1)])

        if not (self.fac == -2):
            self.y2 = [lfilter([1.0], [1.0 - beta], self.y2[c]) for c in range(
                self.n_chans - 2, 0, -1)]

    def build_aud(self, shift=0, fac=-2, dec=8):
        ''' build the auditory spectrogram
            This is largely inspired by Matlab NSL Toolbox
            but transcribed in a more pythonic fashion

            The auditory spectrum is the 5th element in the serie of tranformation
            see [Wang and Shamma 1994] for more details
            it is stored in y5
            '''
        if np.ndim(self.data) > 1:
            raise NotImplementedError(
                "Sorry cannot process multichannel audio for now")

        self.fac = fac
        # octave shift, nonlinear factor, frame length, leaky integration
        L_frm = int(round(8 * 2 ** (4 + shift)))    # frame length (points)

        if dec > 0:
            alph = np.exp(
                -1.0 / (float(dec) * 2.0 ** (4.0 + shift)))    # decaying factor
        else:
            alph = 0
            # short-term avg.

        # hair cell time constant in ms
        haircell_tc = 0.5
        beta = np.exp(-1.0 / (haircell_tc * 2 ** (4 + shift)))

        # compute number of frame
        n_frames = int(np.ceil(self.data.shape[0] / L_frm))
        # if needed pad with zeroes
        if self.data.shape[0] < n_frames * L_frm:
            self.data = np.concatenate((self.data,
                                        np.zeros(((n_frames * L_frm) - self.data.shape[0],))))

        # initialize lists (TODO start from pre-existing y2?)
        self.y2 = []
        self.y4 = []
        self.y3 = []
        self.y5 = []

        self.y2.append(sigmoid(coch_filt(self.data, self.coeffs,
                                         self.n_chans - 1), self.fac))

        # hair cell membrane (low-pass <= 4 kHz); ignored for LINEAR ionic
        # channels
        if not (fac == -2):
            self.y2[0] = lfilter(1, [1 - beta], self.y2[0])
        y2_h = self.y2[0]

        zer_array = np.zeros_like(y2_h)

        # apply cochlear filter to all channels
        self.y2.extend([sigmoid(coch_filt(self.data, self.coeffs, ch),
                                self.fac) for ch in range(self.n_chans - 1, 0, -1)])

        # non-linearity if needed
        if not (fac == -2):
            self.y2 = [lfilter([1.0], [1.0 - beta],
                               self.y2[c]) for c in range(self.n_chans - 1, 0, -1)]

        # Inhibition and thresholding step: TODO optimize
        for ch in range(self.n_chans - 1, 0, -1):

            # masked by higher (frequency) spatial response
            self.y3.append(self.y2[ch] - y2_h)
            y2_h = self.y2[ch]

            # half-wave rectifier ---> y4
            self.y4.append(np.maximum(self.y3[-1], zer_array))

            # temporal integration window ---> y5
            if alph:    # leaky integration
                self.y5.append(lfilter([1.0], [1.0, -alph],
                                       self.y4[-1])[range(0, L_frm * n_frames, L_frm)])

            else:        # short-term average
                if (L_frm == 1):
                    self.y5.append(self.y4[-1])
                else:
                    print
                    self.y5.append(np.mean(
                        self.y4[-1].reshape((L_frm, n_frames)), axis=0))

    def invert_y2(self, shift=0, dec=8):
        ''' recompute the waveform from the auditory spectrum
            Suppose that we still have y2 available

            Here we just perform the inverse wavelet transform
            and sum on all tonotopic channels
            '''

        assert (self.y2 is not None)
        x_rec = np.zeros_like(self.data)

        for ch in range(self.coeffs.shape[1] - 2):
            x_rec += np.flipud(
                coch_filt(np.flipud(self.y2[ch]), self.coeffs, ch))

        return x_rec

    def autofix(self, v5=None):
        ''' we need the auditory spectrum to be real and non-negative
        in order to invert it '''

        # therefore simply apply this to all y5 frames
        if v5 is None:
            self.y5 = map(lambda y: np.maximum(y.real, 0), self.y5)
        else:
            return map(lambda y: np.maximum(y.real, 0), v5)

    def _process_chan(self, x0, ch, beta, y2_h, alph, L_frm, n_frames):
        y2 = sigmoid(coch_filt(x0, self.coeffs, ch), self.fac)

        if not (self.fac == -2):
            y2 = lfilter([1.0,], [1.0, - beta], y2)

        y3 = y2 - y2_h            # difference (L-H)
        y4 = np.maximum(y3, 0)        # half-wave rect.

        if alph:
            y5 = lfilter([1.0], [1.0, -alph], y4)  # leaky integ.
            vx = y5[range(L_frm-1, L_frm * n_frames, L_frm)]    # new aud. spec.
        else:
            vx = np.mean(np.reshape(y4, L_frm, n_frames))
        
        return y2, y3, vx

    def _process_mask(self, n_frames, vx, vt, ch, L_frm, y3):
        # matching
        
#        s = 2.0*np.ones((n_frames, ))
        s = np.ones((n_frames, ))
        
        ind = np.nonzero(vx)[0]
        
        s[ind] *= vt[ch, ind] / vx[ind]
        
        # I do not really understand this value...
        s[np.nonzero(vt[ch, :])[0]] *= 2.0
        
#        s[np.where((vx * vt[ch, :])==0)] = 1.;

        #?? hard scaling TODO Refactoring
#        s = np.multiply(s, np.ones((1, L_frm)))
        return s.repeat(L_frm) * y3

        
    def invert(self, v5=None, init_vec=None, shift=0, dec=8, fac=-2, rec=None, nb_iter=2, display=False):
        ''' recompute the waveform from the auditory spectrum
        stored in y5 or the given v5 list
            '''

        # if no auditory spetcrum is provided use the one that has been
        # computed
        if v5 is None:
            v5 = np.array(self.y5)

        if v5 is None:
            raise ValueError(
                "No auditory spectrum given nor previously computed")

        # If no initial guess is made, then draw first trial at random
#        if rec is None:
#            rec = np.random.randn(self.data.shape[0])

        self.fac = fac
        # octave shift, nonlinear factor, frame length, leaky integration
        L_frm = int(round(8 * 2 ** (4 + shift)))    # frame length (points)

        if dec > 0:
            alph = np.exp(
                -1.0 / (float(dec) * 2.0 ** (4.0 + shift)))    # decaying factor
            alp1 = np.exp(-8.0 / float(dec))
        else:
            alph = 0                   # short-term avg.
            alp1 = 0

        # hair cell time constant in ms
        haircell_tc = 0.5
        beta = np.exp(-1.0 / (haircell_tc * 2 ** (4 + shift)))

        # fix the auditory spectrum
        if not np.isreal(v5).all():
            self.autofix(v5)

        # convert to array if needed
#        v5 = np.array(v5)
        n_frames = v5.shape[1]
        v5_new = v5.copy()
        v5_mean = np.mean(v5)
        v5_sum2 = np.sum(v5 ** 2)
        L_x = n_frames * L_frm

        # making an initial guess: normal
        if init_vec is None:
            x0 = np.random.randn(L_x)
        else:
            x0 = init_vec

        # iteration
        xmin = x0
        # initial sequence with minimum-error
        emin = np.Inf
        # initial minimum error
        errv = []
        # error vector
        vt = v5           # target vector

        for iter_idx in range(nb_iter):

            # Initialization with the last channel
            y2_h = sigmoid(coch_filt(x0, self.coeffs, v5.shape[0]), fac)
            y_cum = np.zeros((L_x,))

            # All other channels
            NORM = self.coeffs[0, -1].imag
            
            # CANNOT Parallelize because each result is used by the next computation
            for ch in range(v5.shape[0] - 1, 0, -1):
                # filtering                
                y2, y3, vx = self._process_chan(x0, ch, beta, y2_h, alph, L_frm, n_frames)
#                print v5_new.shape, vx.shape, ch
                v5_new[ch, :] = vx
#                s = self._process_mask(n_frames, vx, vt, ch, L_frm)
#                print "repeating s :", s.shape
#                if (fac == -2):            # linear hair cell
#                    print y3.shape, s.shape
#                dy = y3
                y1 = self._process_mask(n_frames, vx, vt, ch, L_frm, y3)
#                else:                # nonlinear hair cell
#                    ind = (y3 >= 0)
#                    y1[ind] = y3[ind] * s[ind]
#                    maxy1p = y1[ind].max()
#                    ind = (y3 < 0)
#                    y1[ind] = y3[ind] / np.abs(y3[ind].min()) * maxy1p
                y2_h = y2
                # inverse wavelet transform
                y_cum += np.flipud(coch_filt(np.flipud( y1), self.coeffs, ch)) / NORM

            # previous performance
            v5_r = v5_new / np.mean(v5_new) * v5_mean    # relative v5
            err = np.sum((v5_r - v5) ** 2) / v5_sum2    # relative error
            err = round(err * 10000.0) / 100.0
            era = np.sum((v5_new - v5) ** 2) / v5_sum2    # absolute error
            era = np.round(era * 10000.0) / 100.0

            if display:
                plt.figure()
                plt.subplot(221)
                plt.imshow(v5, aspect='auto', origin='lower')
                plt.title('Original')
                plt.subplot(222)
                plt.imshow(vt, aspect='auto', origin='lower')
                plt.title('Target')
                plt.subplot(223)
                plt.imshow(v5_new, aspect='auto', origin='lower')
                plt.title('New')
                plt.subplot(224)
                plt.imshow(v5_r, aspect='auto', origin='lower')
                plt.title('Relative')
                plt.show()

            errv.extend([err, era])

            if err < emin:           # minimum error found
                emin = err
                xmin = x0
            elif (err - 100) > emin:    # blow up !
                y_cum = np.sign(y_cum) * np.random.randn(L_x)

            # inverse filtering/normalization
            x0 = y_cum #* 1.01
            # pseudo-normalization

            # display performance message
            errstr = '%d, Err.: %5.2f (rel.); %5.2f (abs.) Energy: %3.1e' % (
                iter_idx, err, era, np.sum(x0 ** 2))
            print errstr

        return xmin

    def init_inverse(self, v5=None, shift=0.0):
        ''' calculate a good initial candidate before iterating
            '''
        if v5 is None:
            v5 = np.array(self.y5)
        (M, N) = v5.shape
        print (N, M)
        x0 = 0
        SF = 16000 * 2 ** (shift)

        # characteristic frequency
        CF = 440.0 * 2.0 ** ((np.arange(1.0, self.n_chans - 1.0, 1.0) -
                              31.0) / 24.0 + shift)

        CF = CF / SF
        T = int(8.0 * SF / 1000.0)
        # of points per frame
        L = N * T                # Total  of points

        v5 = v5[0:48, :]

        for m in range(48):
            x = np.cos(2.0 * np.pi * (CF[m] * np.arange(L)))
#            m = np.ones(T, 1) * v5[:, m]
#            x = m(:) .* x;
            x *= np.repeat(v5[m, :], T, axis=0)
            x0 = x0 + x

        x0 -= np.mean(x0)
        x0 /= np.std(x0)
        return x0

    def plot_aud(self, aud_spec=None, ax=None, duration=None):
        if ax is None:
            fig =plt.figure()
            ax = plt.subplot(111)
            
        if aud_spec is None:
            aud_spec =  np.array(self.y5)
            
        ax.imshow(aud_spec,
                   aspect='auto',
                   origin='lower',
                   cmap=cm.bone_r,
                   interpolation='bilinear')
#        plt.colorbar(ax)
        N = aud_spec.shape[0]
        gram = getattr(auditory, 'Y5')
        
        f_vec = (gram.erb_space(N / 10, 180. , 7246.)).astype(int)
        y = np.linspace(0, N, f_vec.shape[0])
        
        if duration is not None:
            t_vec = np.arange(0, duration, duration/10)
            x = np.linspace(0, aud_spec.shape[1], t_vec.shape[0])
            plt.xticks(x, ["%1.1f"%t for t in t_vec])
            ax.set_xlabel('Time (s)')
        
#        ax.set_yticks(np.flipud(y), ["%d"%int(f) for f in f_vec])
        plt.yticks(np.flipud(y), ["%d"%int(f) for f in f_vec])
        print ["%d"%int(f) for f in f_vec]
        ax.set_ylabel('Frequency (Hz)')


class Corticogram(object):
    """ Cortical representation a.k.a 2D wavelet transform of a cochleogram"""
    
    def __init__(self, obj, **kwargs):
    
        self.rv = [1, 2, 4, 8, 16, 32]             
        self.sv = [0.5, 1, 2, 4, 8]
        self.corparams = [8, 8, -2, 0, 0, 0, 1]
                         
        # First test if it's instantiated with a Cochleogram
        if not isinstance(obj, Cochleogram):
            self.coch = Cochleogram(obj, **kwargs)
        else:
            self.coch = obj
    
    def build_cor(self):
        """ need some improvements but should be working """        
        if self.coch.y5 is None:
            self.coch.build_aud()
        self.cor = _build_cor(np.array(self.coch.y5).T, self.corparams, self.rv, self.sv)
#        print self.cor.shape 
#        plt.figure()
#        plt.imshow(np.abs(self.cor[0, 6, :, :]))
#        plt.show()  
        
#        print self.cor.shape
    def invert(self, cor=None):
        """ resynthesize the auditory spectrum from self or from given corticogram """
    
        if cor is None:
            cor = np.copy(self.cor)        
#        print cor.shape
        # HACK : the last parameter should now be zero?
        self.corparams = [8, 8, -2, 0, 0, 0, 0]
        self.rec = _cor2aud(cor, self.corparams, self.rv, self.sv)
   

def coch_filt(data, coeffs, chan_idx):
    p = int(coeffs[0, chan_idx].real)    # order of ARMA filter
    b = coeffs[range(1, p + 2),
               chan_idx].real    # moving average coefficients
    a = coeffs[range(1, p + 2),
               chan_idx].imag    # autoregressive coefficients

    return lfilter(b, a, data)


def sigmoid(y1, fac):
    ''' SIGMOID nonlinear funcion for cochlear model
    #    y = sigmoid(y, fac);
    #    fac: non-linear factor
    #     -- fac > 0, transister-like function
    #     -- fac = 0, hard-limiter
    #     -- fac = -1, half-wave rectifier
    #     -- else, no operation, i.e., linear
    #
    #    SIGMOID is a monotonic increasing function which simulates
    #    hair cell nonlinearity.
    #    See also: WAV2AUD, AUD2WAV

    # Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
    # v1.00: 01-Jun-97

    Transcribed in Python by M. Moussallam (manuel.moussallam@espci.fr)
    '''

    if fac > 0:
        y = np.exp(-y1 / fac)
        return 1.0 / (1.0 + y)
    elif fac == 0:
        return (y > 0)    # hard-limiter
    elif fac == -1:
        y = np.zeros_like(y1)
        y[y1 > 0] = y1[y1 > 0]
        return y   # half-wave rectifier
    else:
        return y1  # do nothing


def auditory_spectrum(audio_file_path):
    ''' computes an auditory spectrogram based from an acoustic array '''

    audio = decoder.Audio(audio_file_path)
    st = 0
    graph = 'Y5'
    N = 64
    win = 0.025
    hop = 0.010
    freqs = [110., 2 * 4435.]
    combine = False
    spec = [[]]

    gram = getattr(auditory, graph)
    gram = gram(audio)

    for t, freq in enumerate(
        gram.walk(N=N, freq_base=freqs[0], freq_max=freqs[1],
                  start=st, end=None, combine=combine, twin=win, thop=hop)):

        spec[0].append(freq)

    return np.array(spec), audio._duration


def plot_auditory(aud_spec, duration, freqs=[110., 2 * 4435.]):

    N = aud_spec.shape[2]
    gram = getattr(auditory, 'Y5')
    f_vec = (gram.erb_space(N / 10, freqs[0], freqs[1])).astype(int)
    y = np.linspace(0, aud_spec.shape[2], f_vec.shape[0])

    t_vec = np.arange(0, duration, 0.5)

    plt.figure()
    plt.imshow(aud_spec[0, :, :].T, aspect='auto',
               cmap=cm.bone_r,
               origin='lower')
    plt.yticks(np.flipud(y), f_vec)
    plt.xticks(t_vec * aud_spec.shape[1] / t_vec.max(), t_vec)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
#    plt.colorbar()
    plt.show()


def gen_cort(fc, L, STF, PASS):
    """ generate cortical filter """
    
    t = np.arange(0.0,float(L))/float(STF) * float(fc)
    h = np.sin(2*np.pi*t) *t**2.* np.exp(-3.5*t) * fc
    
    #%h = diff(h); h = [h(1)/2; h];
    h = h-np.mean(h)
    H0 = fft(h, n=2*L)
    A = np.angle(H0[:L])
    H = np.abs(H0[:L])
    maxi = np.argmax(H)
    maxH = H[maxi] 
    H = H / max(H)
    H = H.astype(complex)  
    #% passband
    if PASS[0] == 1:       #lowpass    
        H[:maxi] = np.ones((maxi, ), complex)    
    elif PASS[0] == PASS[1]:    # highpass    
        H[maxi:L] = np.ones((L-maxi, ), complex)    
    
    return H * np.exp(1j*A)

def gen_corf(fc, L, SRF, KIND):
    
    if len(KIND) == 1:
        PASS = [2, 3];   # bandpass
    else:
        PASS = KIND;
        KIND = 2;
    
    R1    = np.arange(0.0, float(L))/float(L)*float(SRF)/2.0/np.abs(float(fc))
    
    if KIND == 1:    # Gabor function
        C1      = 1.0/2.0/.3/.3;
        H       = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2);
    else:        # Gaussian Function 
        R1    = R1 ** 2;         
        H    = R1 * np.exp(1.0-R1);     # single-side filter
    
    # passband
    if PASS[0] == 1:        #lowpass
        maxi = np.argmax(H)
        maxH = H[maxi]
        sumH = np.sum(H)
        H[:maxi] = np.ones((maxi, ))
        H = H / np.sum(H) * sumH;
    elif PASS[0] == PASS[1]:    # highpass
        maxi = np.argmax(H)
        maxH = H[maxi]
        sumH = np.sum(H);
        H[maxi:L] = np.ones((L-maxi, ))
        H = H / np.sum(H) * sumH;
    
    return H

def _build_cor(y, paras1, rv, sv):
    """ Transcription of aud2cor in NSL Toolbox
        See the original Matlab code for info """
    
    K1    = len(rv);     # num of rate channel
    K2    = len(sv);     # num of scale channel
    [N, M] = y.shape    # dimensions of auditory spectrogram
    
    # spatial, temporal zeros padding 
    N1 = int(2**np.ceil(np.log2(N)))
    N2 = N1*2
    M1 = int(2**np.ceil(np.log2(M)))
    M2 = M1*2
    
    # first fourier transform (w.r.t. frequency axis)
    Y = np.zeros((N2, M1), complex)
    
    for n in range(N):
        R1 = fft(y[n, :], n=int(M2));
        Y[n, :] = R1[:M1];
    
    # second fourier transform (w.r.t. temporal axis)
    for m in range(int(M1)):
        R1 = fft(Y[:N, m], n=N2);
        Y[:, m] = R1;
    
    STF = 1000 / paras1[0]    # frame per second
    SRF = 24                 # channel per octave (fixed)
    
    DISP = 0
    FULLT = paras1[4]
    FULLX = paras1[5]
    BP = paras1[6]
    
    from numpy import floor
    # freq. index
    dM   = int(floor(M/2*FULLX))
    # TODO : this could change if dM is not zero
    mdx1 = range(M+dM)
    mdx2 = np.array([0, 0, M+1, M+1, 0])+dM
    
    # temp. index
    dN   = int(floor(N/2*FULLT))
    ndx  = range(N+2*dN)
    ndx1 = ndx
    ndx2 = [0, N+1, N+1, 0, 0]
    
    z  = np.zeros((N+2*dN, M+2*dM), complex)
    cr = np.zeros((K2, K1*2, N+2*dN, M+2*dM), complex)
    
    # loop on rates
    for rdx in range(K1):    
        fc_rt = rv[rdx]
        HR = gen_cort(fc_rt, N1, STF, [rdx+1+BP, K1+BP*2])
        # 
        for sgn in [1, -1]:            
            # rate filtering modification
            if sgn > 0:
                HR = np.concatenate((HR, np.zeros((N1, ), complex)))    # SSB -> DSB
            else:
                Clist = [HR[0]]
#                Clist = [0.0]
                Clist.extend(np.conj(np.flipud(HR[1:N2])))
                HR = np.array(Clist)            
                HR[N1] = np.abs(HR[N1+1])
            
            # first inverse fft (w.r.t. time axis)
            z1= np.zeros((N2,M1), complex); 
            for m in range(M1): 
                z1[:,m]= HR*Y[:,m];
            
            z1= ifft(z1, axis=0);
            z1= z1[ndx1,:]
            
            for sdx in range(K2):
                # scale filtering
                fc_sc = sv[sdx]
                HS = gen_corf(fc_sc, M1, SRF, [sdx+1+BP, K2+BP*2])
                
                # second inverse fft (w.r.t frequency axis)
                for n in ndx:
                    R1 = ifft( (z1[n, :]*HS), n=M2)
                    z[n, :] = R1[mdx1]
                            
                cr[sdx, rdx+(sgn==1)*K1, :, :] = z;
#                print sdx, rdx, sgn
#                plt.figure()
#                plt.imshow(np.abs(cr[sdx, rdx+(sgn==1)*K1, :, :]))
#                plt.show()
 
    return cr

def _cor2aud(cor, paras1, rv, sv):
    """ Reconstructing auditory spectrogram from 4-D cortical rep"""

    [K2, K12, N, M] = cor.shape    # dimensions of corticogram
    K1 = K12/2
    # spatial, temporal zeros padding 
    N1 = int(2**np.ceil(np.log2(N)))
    N2 = N1*2
    M1 = int(2**np.ceil(np.log2(M)))
    M2 = M1*2
    
    STF = 1000 / paras1[0]    # frame per second
    SRF = 24                 # channel per octave (fixed)
    
    DISP = 0
    FULLT = paras1[4]
    FULLX = paras1[5]
    BP = paras1[6]
    NORM = .9
    HH   = 0;
    Z_cum = 0;

    for rdx in range(K1):
        # rate filtering
        fc_rt = rv[rdx];
        HR = gen_cort(fc_rt, N1, STF, [rdx+1+BP, K1+BP*2]);
#        print HR.shape  , N, N1, N2 
        for sgn in [1, -1]:      
            
            # rate filtering modification
            if sgn > 0:
                HR = np.conj(np.concatenate( ( HR, np.zeros((N1, ),complex)))) 
            else:
                Clist = [HR[0]]
                Clist.extend(np.conj(np.flipud(HR[1:N2])))
                HR = np.array(Clist)
#                print HR.shape                
                HR[N1] = np.abs(HR[N1+1]);            
    
            for sdx in range(K2):
                # scale filtering
#                print sdx+1+BP, K2+(BP*2), BP
                HS = gen_corf(sv[sdx], M1, SRF, [sdx+1+BP, K2+BP*2]);
#                print rdx+(sgn==1)*K1, K1                
                z = np.squeeze(cor[sdx, rdx+(sgn==1)*K1, :, :]);
    
                # 2-D FFT and cumulation
#                Mout = floor(M/2*FULLX);
#                Nout = floor(N/2*FULLT);
                Z_cum, HH = _corfftc(z, Z_cum, N, N1, N2,  M, M1, M2,  HR, HS, HH);

                
    # normalization
    HH[:, 0] = HH[:, 0]*2;        # normalization for DC
    return _cornorm(Z_cum, HH, N, N1, N2, M, M1, M2, NORM);


def _corfftc(z, Z_cum, N, N1, N2,  M, M1, M2,  HR, HS, HH):
    " TODO check that: I simplified a lot"
    # 2-D FFT
    Z = np.zeros((N2, M1),complex);    
#    z[0, M2] = 0;    # why? I forgot. Oh, it is zero padding
#    print Z.shape, z.shape, HR.shape, HS.shape    
    for n in range(N):        
       R1 = fft(z[n,:], n=M2) # Oh, it is zero padding
       Z[n, :] = R1[:M1]    
    
    for m in range(M1):
       Z[:, m] = fft(Z[:, m]);
    
    # cumulation    
    R1 = np.dot(HR.reshape((len(HR),1)),HS.reshape((1,len(HS))))        
    HH = HH + R1 * np.conj(R1);    
    Z_cum = Z_cum + R1 * Z;
    return Z_cum, HH

def _cornorm(Z_cum, HH, N, N1, N2, M, M1, M2, NORM):
    """ Simplified a lot !"""
    FOUTT = 0
    FOUTX = 0
    
    # modify overall transfer function
    sumH = np.sum(HH)
    HH = NORM * HH + (1.0 - NORM) * np.max(HH);
    HH = HH / np.sum(HH) * sumH;
    
    # normalization
    ndx = range(N2)
    mdx = range(M1)
#    print ndx, mdx
#    Z_cum[ndx, mdx] /= HH[ndx, mdx];
    Z_cum /= HH
    
    # SIMPLIFIED dN=dM=0
    ndx  = range(N)                    
    ndx1 = range(N)         
    mdx1 = range(M)   
    
    y     = np.zeros((N, M1),complex)
    yh    = np.zeros((N, M),complex)
    
    # 2-D IFFT
    for m in range(M1):
        R1 = ifft(Z_cum[:, m])
        y[:, m] = R1[ndx1]

    for n in ndx:
        R1 = ifft(y[n, :], M2)
        yh[n, :] = R1[mdx1]    
    
    return yh * 2;
