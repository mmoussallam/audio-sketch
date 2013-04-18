'''
Created on Jan 30, 2013

@author: manu
'''
import numpy as np
import os
from PyMP import Signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tools import cochleo_tools

class AudioSketch(object):
    ''' This class should be used as an abstract one, specify the
    audio sketches interface. 
    
    Sketches are audio objects whose 
    parameter dimension has been reduced: any kind of approximation
    of an auditory scene can be understood as a sketch. However
    the interesting one are those whith much smaller parametrization
    than e.g. raw PCM or plain STFT.
    
    Sketches may refer to an original sound, or be synthetically 
    generated (e.g. by using a random set of parameters) interesting
    sketch features include:
    
    `params` the set of sufficient parameters (as a dictionary) to 
             synthesize the audio sketch
             
    Desirable methods of this object are:
    
    `recompute`  which will take an audio Signal object as input as 
                recompute the sketch parameters
    
    `synthesize` quite a transparent one
    
    `sparsify`   this method should allow a sparsifying of the sketch
                i.e. a reduction of the number of parameter
                e.g. it can implement a thresholding or a peak-picking method
    
    `represent`  optionnal: a nice representation of the sketch
    
    `fgpt`        build a fingerprint from the set of parameters
    '''


    params = {}             # dictionary of parameters
    orig_signal = None      # original Signal object
    rec_signal = None       # reconstructed Signal object
    rep = None              # representation handle
    sp_rep = None           # sparsified representation handle
    
    
    def __repr__(self):
        strret = ''' 
%s  %s
Params: %s ''' % (self.__class__.__name__, str(self.orig_signal),str(self.params))
        return  strret
    
    def synthesize(self, sparse=False):    
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")
    
    def represent(self, sparse=False):
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def fgpt(self):
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")
    
    def recompute(self):
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")
    
    def sparsify(self, sparsity):
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")
        
        
class STFTPeaksSketch(AudioSketch):
    ''' Sketch based on a single STFT with peak-picking as a 
    sparsifying method '''
    
    # baseline parameters: default rectangle is 10 frames by 10 bins large
    params = {'scale':1024,
              'step':512,
              'f_width':10,
              't_width':10}
            
    # TODO this is same as superclass
    def __init__(self, original_sig=None, **kwargs):        
        # add all the parameters that you want
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        
        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute() 

    def recompute(self, signal = None):

        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal
        
        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")

        import stft
        self.rep = stft.stft(self.orig_signal.data,
                             self.params['scale'],
                             self.params['step'])

    def represent(self, sparse=False, **kwargs):
        if sparse:
            rep = np.zeros_like(self.sp_rep)
            rep[np.nonzero(self.sp_rep)] = 1.
        else:
            rep = self.rep        
        
        x_tick_vec = np.linspace(0, rep.shape[2], 10).astype(int)
        x_label_vec = (x_tick_vec*float(self.params['step']))/float(self.orig_signal.fs)
        
        y_tick_vec = np.linspace(0, rep.shape[1], 6).astype(int)
        y_label_vec = (y_tick_vec/float(self.params['scale']))*float(self.orig_signal.fs)            
        
        plt.figure()
        for chanIdx in range(rep.shape[0]):
            plt.subplot(rep.shape[0],1,chanIdx+1)
            plt.imshow(10*np.log10(np.abs(rep[chanIdx,:,:])),
               aspect='auto',
               interpolation='nearest',
               origin='lower',
               cmap = cm.binary_r)
            plt.xlabel('Time (s)')
            plt.xticks(x_tick_vec, ["%1.1f"%a for a in x_label_vec])
            plt.ylabel('Frequency')
            plt.yticks(y_tick_vec, ["%d"%int(a) for a in y_label_vec])
    
    def sparsify(self, sparsity, **kwargs):
        ''' sparsity is here achieved through Peak-Picking in the 
        STFT: naive version: square TF regions'''
        if self.rep is None:
            raise ValueError("representation hasn't been computed yet..")
        
        for key in kwargs:
            self.params[key] = kwargs[key]
            
        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")            
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= np.sum(self.rep.shape)
#        else:
            # otherwise the sparsity argument take over and we divide in 
            # the desired number of regions (preserving the bin/frame ratio)
#            print self.rep.shape[1:]
        self.params['f_width'] = int(self.rep.shape[1]/np.sqrt(sparsity))
        self.params['t_width'] = int(self.rep.shape[2]/np.sqrt(sparsity))
#            print self.params['f_width'], self.params['t_width']
            
        self.sp_rep = np.zeros_like(self.rep)        
        # naive implementation: cut in non-overlapping zone and get the max
        (n_bins, n_frames) = self.rep.shape[1:]            
        (f,t) = (self.params['f_width'], self.params['t_width'])
        for x_ind in range(0, (n_frames/t)*t, t):
            for y_ind in range(0, (n_bins/f)*f, f):
                rect_data = self.rep[0,y_ind:y_ind+f,x_ind:x_ind+t]
                
                if len(rect_data)>0 and (np.sum(rect_data**2) > 0):
                    f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
                    # add the peak to the sparse rep
                    self.sp_rep[0, y_ind + f_index, x_ind + t_index] = rect_data[f_index, t_index]
        
        self.nnz = np.count_nonzero(self.sp_rep)
        print "Sparse rep of %d element computed" % self.nnz
        
        
#    def represent_sparse(self):
#        if self.sp_rep is None:
#            raise ValueError("no sparse rep constructed yet")
#        
#        plt.figure()
#        for chanIdx in range(self.sp_rep.shape[0]):
#            plt.subplot(self.sp_rep.shape[0],1,chanIdx+1)
#            plt.imshow(10*np.log10(np.abs(self.sp_rep[chanIdx,:,:])),
#               aspect='auto',
#               interpolation='nearest',
#               origin='lower')

    def synthesize(self, sparse=False):
        import stft
        
        if sparse:        
            return Signal(stft.istft(self.sp_rep,
                                            self.params['step'],
                                            self.orig_signal.length),
                                 self.orig_signal.fs, mono=True)
        else:
            return Signal(stft.istft(self.rep,
                                            self.params['step'],
                                            self.orig_signal.length),
                                 self.orig_signal.fs, mono=True)

class STFTDumbPeaksSketch(STFTPeaksSketch):
    ''' only changes the sparsifying method '''
    
    def __init__(self, original_sig=None, **kwargs):        
        # add all the parameters that you want
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        
        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
     
    def sparsify(self, sparsity):
        ''' sparsify using the peaks with no spearding on the TF plane '''
        if self.rep is None:
            raise ValueError("STFT not computed yet")
        
                
        self.sp_rep = np.zeros_like(self.rep.ravel())
#        print self.sp_rep.shape
        # peak picking
        max_indexes = np.argsort(self.rep.ravel())
        self.sp_rep[max_indexes[-sparsity:]] = self.rep.ravel()[max_indexes[-sparsity:]]
        
        self.sp_rep = np.reshape(self.sp_rep, self.rep.shape)
    

class XMDCTSparseSketch(AudioSketch):
    ''' A sketching based on MP with a union of MDCT basis '''
    
    
    # baseline parameters: default rectangle is 10 frames by 10 bins large
    params = {'scales':[128,1024,8192],
              'nature':'LODico',
              'n_atoms':1000,
              'SRR':30}

    # TODO this is same as superclass
    def __init__(self, original_sig=None, **kwargs):        
        # add all the parameters that you want
        for key in kwargs:
            self.params[key] = kwargs[key]
                
        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute() 
            
    def recompute(self, signal = None, **kwargs):
        for key in kwargs:
            self.params[key] = kwargs[key]

        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal
        
        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")

        from PyMP.mdct import Dico
        from PyMP import mp
        mdct_dico = Dico(self.params['scales'])
        self.rep = mp.mp(self.orig_signal,
                         mdct_dico,
                         self.params['SRR'],
                         self.params['n_atoms'],
                         debug=0)[0]
        
    def represent(self, fig=None,  sparse=False):
        if fig is None:
            plt.figure()

        if sparse:
            self.sp_rep.plot_tf()    
        else:   
            self.rep.plot_tf()
        
#    def represent_sparse(self, fig=None, sparse=False):                
#        if fig is None:
#            fig = plt.figure()        
#        
#        if self.sp_rep is None:
#            return self.represent(fig)
#                
#        self.sp_rep.plot_tf()
        
    def sparsify(self, sparsity, **kwargs):
        ''' here behaviour is this:
        if sparsity > current number of atoms: pursue the decomposition
        else return the desired number of atoms as sp_rep'''

        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")            
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= self.rep.recomposed_signal.length

        if self.rep is None:
            self.params['n_atoms'] = sparsity
            self.recompute(**kwargs)
            self.sp_rep = self.rep
        
        elif sparsity > self.rep.atom_number:
            self.params['n_atoms'] = sparsity
            # Sparsity asked in more than has been computed
            from PyMP.mdct import Dico
            from PyMP import mp
            self.sp_rep = mp.mp_continue(self.rep,
                                         self.orig_signal,
                                         Dico(self.params['scales']),
                                         self.params['SRR'],
                                         self.params['n_atoms']-self.rep.atom_number,
                                         pad=False)[0]
        
        else:
            # thanks to getitem: it will construct a new approx object
            # with the #sparsity biggest atoms
            self.sp_rep = self.rep[:sparsity]
        
        self.nnz = self.sp_rep.atom_number        
        print "Sparse rep of %d element computed" % self.nnz
    
    def synthesize(self, sparse=False):
        if not sparse:
            return self.rep.recomposed_signal
        else:
            return self.sp_rep.recomposed_signal
        

class CochleoDumbPeaksSketch(AudioSketch):
    ''' Sketch based on a cochleogram and pairs of peaks as a sparsifier '''
    
    # parameters
    n_bands = 64; # number of cochlear filters
    
    
    def __init__(self, original_sig=None, **kwargs):        
        # add all the parameters that you want
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        
        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
        
            
    def synthesize(self, sparse=False):    
        ''' synthesize the sparse rep or the original rep?'''
        if sparse:
            v5 = self.sp_rep
        else:
            v5 = np.array(self.cochleogram.y5)
            
        # initialize invert
        init_vec =  self.cochleogram.init_inverse(v5)
        # then do 20 iteration (TODO pass as a parameter)
        return Signal(self.cochleogram.invert(v5, init_vec, nb_iter=10, display=False),
                      self.orig_signal.fs)
    
    def represent(self, fig=None, sparse=False):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)            
        else:
            ax = plt.gca()
            
        if sparse:
            self.cochleogram.plot_aud(ax = ax,
                                      aud_spec=self.sp_rep,
                                      duration= self.orig_signal.get_duration())
        else:
            self.cochleogram.plot_aud(ax = ax,
                                      duration= self.orig_signal.get_duration())
    
    def fgpt(self):
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")
    
    def recompute(self, signal = None, **kwargs):
        ''' recomputing the cochleogram'''
        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal
        
        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        
        self.cochleogram = cochleo_tools.cochleogram(self.orig_signal.data)
        self.cochleogram.build_aud()
    
    def sparsify(self, sparsity):
        ''' sparsify using the peaks '''
        if self.cochleogram.y5 is None:
            raise ValueError("cochleogram not computed yet")
        
        v5 = np.array(self.cochleogram.y5)        
        self.sp_rep = np.zeros_like(v5.ravel())
#        print self.sp_rep.shape
        # peak picking
        
        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")            
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= np.sum(self.sp_rep.shape)
                
        max_indexes = np.argsort(v5.ravel())
        self.sp_rep[max_indexes[-sparsity:]] = v5.ravel()[max_indexes[-sparsity:]]
        
        self.sp_rep = np.reshape(self.sp_rep, v5.shape)

class CochleoPeaksSketch(CochleoDumbPeaksSketch):
    """ A slightly less stupid way to select the coefficients : by spreading them
        in the TF plane 
        
        only need to rewrite sparsify
        """
    def __init__(self, original_sig=None, **kwargs):        
        # add all the parameters that you want
        super(CochleoPeaksSketch, self).__init__(original_sig=original_sig, **kwargs)
     
    def sparsify(self, sparsity):
        ''' sparsify using the peaks with spreading on the TF plane '''
        if self.rep is None:
            raise ValueError("STFT not computed yet")
        
                
        self.sp_rep = np.zeros_like(self.rep.ravel())
#        print self.sp_rep.shape
        # peak picking
        max_indexes = np.argsort(self.rep.ravel())
        self.sp_rep[max_indexes[-sparsity:]] = self.rep.ravel()[max_indexes[-sparsity:]]
        
        self.sp_rep = np.reshape(self.sp_rep, self.rep.shape)
        
    
class SWSSketch(AudioSketch):
    """ Sine Wave Speech """     
    params = {'n_formants':5,
              'time_step':0.01,
              'windowSize':0.025,
              'preEmphasis':50,
              'script_path':'/home/manu/workspace/audio-sketch/src/tools'}
    formants = None
               
    def __init__(self, orig_sig = None, **kwargs):
        for key in kwargs:
            self.params[key] = kwargs[key]
        self.call_str = 'praat %s/getsinewavespeech.praat'%self.params['script_path']
        if orig_sig is not None:
            self.recompute(orig_sig, **kwargs )
            
    
    def synthesize(self, sparse=False):    
        if not sparse:
            return self.rep
        else:
            return self.sp_rep
    
    def represent(self, fig=None, sparse=False):
        """ plotting the sinewave speech magnitude spectrogram"""
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)            
        else:
            ax = plt.gca()
        if not sparse:
            self.rep.spectrogram(2**int(np.log2(self.params['windowSize']*self.rep.fs)),
                              2**int(np.log2(self.params['time_step']*self.rep.fs)),
                            order=1, log=True, ax=ax)
        else:
            self.sp_rep.spectrogram(2**int(np.log2(self.params['windowSize']*self.rep.fs)),
                              2**int(np.log2(self.params['time_step']*self.rep.fs)),
                            order=1, log=True, ax=ax)

    def fgpt(self):
        raise NotImplementedError("NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")
    

    def _extract_sws(self, signal, kwargs):
        """ internal routine to extract the sws"""
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        if signal is None or not os.path.exists(signal):
            raise ValueError("A valid path to a wave file must be provided")
        self.current_sig = signal
        os.system('%s %s %1.3f %d %1.3f %d' % (self.call_str, signal, self.params['time_step'], self.params['n_formants'], self.params['windowSize'], self.params['preEmphasis']))
    # Now retrieve the coefficients and the resulting audio
        self.formants = []
        for forIdx in range(1, 4):
            formant_file = signal[:-4] + '_formant' + str(forIdx) + '.mtxt'
            fid = open(formant_file, 'rU')
            vals = fid.readlines()
            fid.close() # remove first 3 lines and convert to numpy
            self.formants.append(np.array(vals[3:], dtype='float'))

    def recompute(self, signal = None, **kwargs):                 
        self._extract_sws(signal, kwargs)
        
        # save as the original signal if it's not already set
        if self.orig_signal is None:
            self.orig_signal = Signal(signal, normalize=True)
        # and the audio we said
        self.rep = Signal(signal[:-4] + '_sws.wav', normalize=True) 
    
    def sparsify(self, sparsity):
        """ It has already been sparsified """
        # @TODO use sparsity to determine the number of formants and window size/steps
        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")            
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= self.rep.length
        
        new_tstep = float(int(sparsity)/self.params['n_formants'])/float(self.rep.fs)
        print "New time step of %1.3f seconds"%new_tstep
        self._extract_sws(self.current_sig, {'time_step': new_tstep})
        self.sp_rep = Signal(self.current_sig[:-4] + '_sws.wav', normalize=True) 