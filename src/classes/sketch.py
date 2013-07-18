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
from joblib import Memory
memory = Memory(cachedir='/tmp/joblib', verbose=0)


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
Params: %s ''' % (self.__class__.__name__, str(self.orig_signal), str(self.params))
        return strret

    def get_sig(self):
        """ Returns a string specifying most important parameters """
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def synthesize(self, sparse=False):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def represent(self, sparse=False):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def fgpt(self):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def recompute(self):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def sparsify(self, sparsity):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

class STFTPeaksSketch(AudioSketch):
    ''' Sketch based on a single STFT with peak-picking as a
    sparsifying method '''

    # TODO this is same as superclass
    def __init__(self, original_sig=None, **kwargs):
        
        # baseline parameters: default rectangle is 10 frames by 10 bins large        
        self.params = {'scale': 1024,
              'step': 512,
              'f_width': 10,
              't_width': 10}
        
        # add all the parameters that you want
        for key in kwargs:
            self.params[key] = kwargs[key]

        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
        

    def get_sig(self):
        strret = '_scale-%d_fw-%d_tw-%d' % (self.params['scale'],
                                            self.params['f_width'],
                                            self.params['t_width'])
        return strret

    def recompute(self, signal=None, **kwargs):
        for key in kwargs:
            self.params[key] = kwargs[key]
        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        self.params['fs'] = self.orig_signal.fs
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
        x_label_vec = (
            x_tick_vec * float(self.params['step'])) / float(self.orig_signal.fs)

        y_tick_vec = np.linspace(0, rep.shape[1], 6).astype(int)
        y_label_vec = (y_tick_vec / float(
            self.params['scale'])) * float(self.orig_signal.fs)

        plt.figure()
        for chanIdx in range(rep.shape[0]):
            plt.subplot(rep.shape[0], 1, chanIdx + 1)
            plt.imshow(10 * np.log10(np.abs(rep[chanIdx, :, :])),
                       aspect='auto',
                       interpolation='nearest',
                       origin='lower',
                       cmap=cm.coolwarm)
            plt.xlabel('Time (s)')
            plt.xticks(x_tick_vec, ["%1.1f" % a for a in x_label_vec])
            plt.ylabel('Frequency')
            plt.yticks(y_tick_vec, ["%d" % int(a) for a in y_label_vec])

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
        self.params['f_width'] = int(self.rep.shape[1] / np.sqrt(sparsity))
        self.params['t_width'] = int(self.rep.shape[2] / np.sqrt(sparsity))
#            print self.params['f_width'], self.params['t_width']

        self.sp_rep = np.zeros_like(self.rep)
        # naive implementation: cut in non-overlapping zone and get the max
        (n_bins, n_frames) = self.rep.shape[1:]
        (f, t) = (self.params['f_width'], self.params['t_width'])
        for x_ind in range(0, (n_frames / t) * t, t):
            for y_ind in range(0, (n_bins / f) * f, f):
                rect_data = self.rep[0, y_ind:y_ind + f, x_ind:x_ind + t]

                if len(rect_data) > 0 and (np.sum(rect_data ** 2) > 0):
                    f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
                    # add the peak to the sparse rep
                    self.sp_rep[0, y_ind + f_index,
                                x_ind + t_index] = rect_data[f_index, t_index]

        self.nnz = np.count_nonzero(self.sp_rep)
#        print "Sparse rep of %d element computed" % self.nnz

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


    def fgpt(self, sparse=False):
        """ This only has a meaning if the peaks have been selected """
        if self.sp_rep is None:
            print "WARNING : default peak-picking of 100"
            self.sparsify(100)        
        return self.sp_rep

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
        self.sp_rep[max_indexes[-sparsity:]] = self.rep.ravel(
        )[max_indexes[-sparsity:]]

        self.sp_rep = np.reshape(self.sp_rep, self.rep.shape)

    def get_sig(self):
        strret = '_scale-%d' % (self.params['scale'])
        return strret

# class GreedySparseSketch(AudioSketch):
#    """ An interface for various dictionary-based sketches """
#    def __init__(self, original_sig=None, **kwargs):
#        # add all the parameters that you want
#        self.params = {'scales':[128,1024,8192],
#              'nature':'MDCT',
#              'n_atoms':1000,
#              'SRR':30}


class XMDCTSparseSketch(AudioSketch):
    ''' A sketching based on MP with a union of MDCT basis '''

    # baseline parameters: default rectangle is 10 frames by 10 bins large
    # TODO this is same as superclass
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        self.params = {'scales': [128, 1024, 8192],
                       'nature': 'MDCT',
                         'n_atoms': 1000,
                         'SRR': 30}

        for key in kwargs:
            self.params[key] = kwargs[key]

        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
        

    def get_sig(self):
        strret = '_%dx%s' % (len(self.params['scales']),
                                      self.params['nature'])
        return strret

    def _get_dico(self):
        from PyMP.mdct import Dico, LODico
        from PyMP.mdct.dico import SpreadDico
        from PyMP.wavelet import dico as wavelet_dico

        if self.params['nature'] == 'LOMDCT':
            mdct_dico = LODico(self.params['scales'])

        elif self.params['nature'] == 'MDCT':
            mdct_dico = Dico(self.params['scales'])
        
        elif self.params['nature'] == 'SpreadMDCT':
            mdct_dico = SpreadDico(self.params['scales'])
        else:
            raise ValueError("Unrecognized nature %s" % self.params['nature'])
        return mdct_dico

    def recompute(self, signal=None, **kwargs):
        for key in kwargs:
            self.params[key] = kwargs[key]

        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        self.params['fs'] = self.orig_signal.fs
        mdct_dico = self._get_dico()

        from PyMP import mp
        self.rep = mp.mp(self.orig_signal,
                         mdct_dico,
                         self.params['SRR'],
                         self.params['n_atoms'],
                         debug=0)[0]

    def represent(self, fig=None, sparse=False):
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
            self.params['n_atoms'] = int(sparsity)
            self.recompute(**kwargs)
            self.sp_rep = self.rep

        elif sparsity > self.rep.atom_number:
            self.params['n_atoms'] = int(sparsity)
            # Sparsity asked in more than has been computed

            from PyMP import mp
            self.sp_rep = mp.mp_continue(self.rep,
                                         self.orig_signal,
                                         self._get_dico(),
                                         self.params['SRR'],
                                         self.params[
                                             'n_atoms'] - self.rep.atom_number,
                                         pad=False)[0]

        else:
            # thanks to getitem: it will construct a new approx object
            # with the #sparsity biggest atoms
            self.sp_rep = self.rep[:int(sparsity)]

        self.nnz = self.sp_rep.atom_number
        print "Sparse rep of %d element computed" % self.nnz

    def synthesize(self, sparse=False):
        if not sparse:
            return self.rep.recomposed_signal
        else:
            return self.sp_rep.recomposed_signal

    def fgpt(self, sparse=False):
        """ In this case it is quite simple : just return the approx objects """
        return self.sp_rep if sparse else self.rep

class WaveletSparseSketch(XMDCTSparseSketch):
    """ Same as Above but using a wavelet dictionary """
    def __init__(self, original_sig=None, **kwargs):
        """ just call the superclass constructor nothing needs to be changed """
        super(WaveletSparseSketch, self).__init__(original_sig=None, **kwargs)

    def get_sig(self):
        strret = '_%dx%s-atoms-%d' % (len(self.params['wavelets']),
                                      self.params['nature'],
                                      self.params['n_atoms'])
        return strret

    def _get_dico(self):
        from PyMP.mdct import Dico, LODico
        from PyMP.wavelet import dico as wavelet_dico

        if self.params['nature'] == 'Wavelet':
            dico = wavelet_dico.WaveletDico(self.params['wavelets'], pad=0)
        else:
            raise ValueError("Unrecognized nature %s" % self.params['nature'])
        return dico

    def represent(self, fig=None, sparse=False):
        if fig is None:
            plt.figure()

        if sparse:
            self.sp_rep.recomposed_signal.spectrogram(
                wsize=512, tstep=256, order=1, log=True, ax=plt.gca())
        else:
            self.rep.recomposed_signal.spectrogram(
                wsize=512, tstep=256, order=1, log=True, ax=plt.gca())


class CochleoSketch(AudioSketch):
    """ meta class for all cochleogram-based sketches     
    
    Subclass should implement their own sparsification method
    """
    
    
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want        
        
        self.params = {'n_bands': 64, 'shift':0, 'fac':-2,'n_inv_iter':5}
        self.cochleogram = None
        for key in kwargs:
            self.params[key] = kwargs[key]

        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
        

    def get_sig(self):
        strret = '_bands-%d_' % (self.params['n_bands'])
        return strret

    def synthesize(self, sparse=False):
        ''' synthesize the sparse rep or the original rep?'''
        if sparse:
            v5 = self.sp_rep
        else:
            v5 = np.array(self.cochleogram.y5)

        # initialize invert
        init_vec = self.cochleogram.init_inverse(v5)
        # then do 20 iteration (TODO pass as a parameter)
        return Signal(
            self.cochleogram.invert(v5, init_vec, nb_iter=self.params['n_inv_iter'], display=False),
            self.orig_signal.fs)

    def represent(self, fig=None, sparse=False):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()

        if sparse:
            self.cochleogram.plot_aud(ax=ax,
                                      aud_spec=self.sp_rep,
                                      duration=self.orig_signal.get_duration())
        else:
            self.cochleogram.plot_aud(ax=ax,
                                      duration=self.orig_signal.get_duration())

    def fgpt(self, sparse=False):
        """ This only has a meaning if the peaks have been selected """
        if self.sp_rep is None:
            print "WARNING : default peak-picking of 100"
            self.sparsify(100)        
        return self.sp_rep

    def recompute(self, signal=None, **kwargs):
        ''' recomputing the cochleogram'''
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        
        if self.params.has_key('downsample'):
            self.orig_signal.downsample(self.params['downsample'])
        
        # cleaning
        if self.cochleogram is not None:
            del self.cochleogram
               
        self.cochleogram = cochleo_tools.Cochleogram(self.orig_signal.data, **self.params)
        self.cochleogram.build_aud()
        self.rep = np.array(self.cochleogram.y5)

class CorticoSketch(AudioSketch):
    """ meta class for all corticogram-based sketches     
    
    Subclass should implement their own sparsification method
    """
    
    def __init__(self, obj=None, **kwargs):
        # add all the parameters that you want        
        
        self.params = {'n_bands': 64,
                       'shift':0,                       
                       'rv':[1, 2, 4, 8, 16, 32],
                       'sv':[0.5, 1, 2, 4, 8]}
        self.orig_signal = None
        self.rec_aud = None
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        if obj is not None:
            self.orig_signal = obj
            self.recompute()
#        if isinstance(obj, cochleo_tools.Cochleogram):
#            self.coch = obj
#            self.orig_signal = self.coch.
#        elif isinstance(obj, Signal):        
#            self.orig_signal = obj
#            self.coch = cochleo_tools.Cochleogram(self.orig_signal.data)
#            self.recompute()
#        else:
#            raise TypeError("Object %s is neither a cochleogram nor a signal"%str(obj))

    def get_sig(self):
        strret = '_bands-%d_%drates_%scales' % (self.params['n_bands'],
                                                len(self.params['rv']),
                                                len(self.params['sv']))
        return strret

    def synthesize(self, sparse=False):
        ''' synthesize the sparse rep or the original rep?'''
        if sparse:
            cor = self.cort
            # sparse auditory spectrum should already have been computed
#            if self.rec_aud is None:
            self.rec_aud = cochleo_tools._cor2aud(self.sp_rep, **self.params)
            v5 = np.abs(self.rec_aud).T
        else:
            cor = self.cort
            # inverting the corticogram
            v5 = np.abs(cor.invert()).T                    


        # then do 20 iteration (TODO pass as a parameter)
        if self.orig_signal is not None:
            return Signal(
            self.coch.invert(v5, self.orig_signal.data, nb_iter=self.params['n_inv_iter'], display=False),
            self.orig_signal.fs)
        else:
            # initialize invert        
            init_vec = self.coch.init_inverse(v5)
            return Signal(
            self.coch.invert(v5, init_vec, nb_iter=self.params['n_inv_iter'], display=False),
            8000)

    def represent(self, fig=None, sparse=False):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()

        if sparse:            
            self.cort.plot_cort(cor=self.sp_rep)
        else:
            self.cort.plot_cort()

    def fgpt(self, sparse=False):
        """ return the 4-D sparsified representation """
        if sparse:
            return self.sp_rep
        return self.rep

    def recompute(self, signal=None, **kwargs):
        ''' recomputing the cochleogram'''
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        
        if self.params.has_key('downsample'):
            self.orig_signal.downsample(self.params['downsample'])
                
        self.coch = cochleo_tools.Cochleogram(self.orig_signal.data, **self.params)
        self.coch.build_aud()
        self.cort = cochleo_tools.Corticogram(self.coch, **self.params)
        self.cort.build_cor()
        self.rep = np.array(self.cort.cor)

class CochleoDumbPeaksSketch(CochleoSketch):
    ''' Sketch based on a sparse peaks in the cochleogram '''

    # number of cochlear filters
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CochleoDumbPeaksSketch, self).__init__(
            original_sig=original_sig, **kwargs)
    
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
        self.sp_rep[max_indexes[-sparsity:]] = v5.ravel(
        )[max_indexes[-sparsity:]]

        self.sp_rep = np.reshape(self.sp_rep, v5.shape)
    


class CochleoPeaksSketch(CochleoSketch):
    """ A slightly less stupid way to select the coefficients : by spreading them
        in the TF plane

        only need to rewrite sparsify @TODO
        """
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CochleoPeaksSketch, self).__init__(
            original_sig=original_sig, **kwargs)

    def fgpt(self, sparse=False):
        """ This only has a meaning if the peaks have been selected """
        if self.sp_rep is None:
            print "WARNING : default peak-picking of 100"
            self.sparsify(100)        
        return self.sp_rep

    def sparsify(self, sparsity, **kwargs):
        '''sparsify using the peak-picking with spreading on the TF plane '''
        if self.rep is None:
            self.rep = np.array(self.cochleogram.y5)
            
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
        self.params['f_width'] = int(self.rep.shape[0] / np.sqrt(sparsity))
        self.params['t_width'] = int(self.rep.shape[1] / np.sqrt(sparsity))
#            print self.params['f_width'], self.params['t_width']

        self.sp_rep = np.zeros_like(self.rep)
        # naive implementation: cut in non-overlapping zone and get the max
        (n_bins, n_frames) = self.rep.shape
        (f, t) = (self.params['f_width'], self.params['t_width'])
        for x_ind in range(0, (n_frames / t) * t, t):
            for y_ind in range(0, (n_bins / f) * f, f):
                rect_data = self.rep[y_ind:y_ind + f, x_ind:x_ind + t]

                if len(rect_data) > 0 and (np.sum(rect_data ** 2) > 0):
                    f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
                    # add the peak to the sparse rep
                    self.sp_rep[y_ind + f_index,
                                x_ind + t_index] = rect_data[f_index, t_index]

        self.nnz = np.count_nonzero(self.sp_rep)
        
class CorticoPeaksSketch(CorticoSketch):
    """ Peack Picking on the 4-D corticogram as the sparsification process
    """
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CorticoPeaksSketch, self).__init__(
            original_sig=original_sig, **kwargs)
                
        self.params['n_inv_iter'] = 2   # number of reconstructive steps
        for k in kwargs:
            self.params[k] = kwargs[k]

    def sparsify(self, sparsity, **kwargs):
        """ Sparfifying using plain peak picking """
        
        if self.rep is None:
            self.rep = self.cort.cor
        
        if self.rep is None:
            raise ValueError("Not computed yet!")
        
        self.sp_rep = np.ones(self.rep.shape, bool)
        alldims = range(len(self.rep.shape))
        for id in alldims:
            # compute the diff in the first axis after swaping
            d = np.diff(np.swapaxes(self.rep, 0, id), axis=0)
            
            self.sp_rep = np.swapaxes(self.sp_rep, 0, id)
            self.sp_rep[:-1,...] &= d < 0
            self.sp_rep[1:,...] &= d > 0
            
            self.sp_rep = np.swapaxes(self.sp_rep, 0, id)

        self.sp_rep = self.sp_rep.astype(int)
        r_indexes = np.flatnonzero(self.sp_rep)        
        r_values = self.rep.flatten()[r_indexes]
        inds = np.abs(r_values).argsort()
        
        self.sp_rep = np.zeros_like(self.rep.flatten())
        self.sp_rep[inds[-sparsity:]] = r_values[inds[-sparsity:]]
        self.sp_rep = np.reshape(self.sp_rep, self.rep.shape)
        # no only keep the k biggest values
        

class CorticoIHTSketch(CorticoSketch):
    """ Iterative Hard Thresholding on a 4-D corticogram spectrum 
    
    Inherit from CorticoSketch and only implements a different sparisication
    method
    """
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CorticoIHTSketch, self).__init__(
            original_sig=original_sig, **kwargs)
        
        self.params['max_iter'] = 5     # number of IHT iterations
        self.params['n_inv_iter'] = 2   # number of reconstructive steps
        for k in kwargs:
            self.params[k] = kwargs[k]
    
    def get_sig(self):
        strret = '_%diter_frmlen%d' % (self.params['max_iter'],
                                            self.params['frmlen'])
        return strret
    
    def sparsify(self, sparsity, **kwargs):
        """ sparsification is performed using the 
        Iterative Hard Thresholding Algorithm """
        L = sparsity
        if  self.cort is None:
            raise ValueError("No representation has been computed yet")
        
        if self.coch.y5 is None:
            self.coch.build_aud()
        
        for key in kwargs:
            self.params[key] = kwargs[key]

        # We go back and forth from the auditory spectrum to the 4-D corticogram                
        X = np.array(self.coch.y5).T
        # dimensions
        K1    = len(self.params['rv']);     # of rate channel
        K2    = len(self.params['sv']);     # of scale channel
        (N2, M1)  = X.shape    # dimensions of auditory spectrogram
        
        # initialize output and residual
        A = np.zeros((K2,2*K1,N2,M1), complex)        
        residual = X
        
        n_iter = 0
        
        while n_iter < self.params['max_iter']:
            print "IHT Iteration %d"%n_iter       
            A_old = np.copy(A)     
            # build corticogram                  
            projection = cochleo_tools._build_cor(residual, **self.params)

            # sort the elements and hard threshold        
            A_buff = A + projection
            A_flat = A_buff.flatten()
            idx_order = np.abs(A_flat).argsort()
            A = np.zeros(A_flat.shape, complex)
            A[idx_order[-L:]] = A_flat[idx_order[-L:]]
            A = A.reshape(A_buff.shape)
            
            # Reconstruct auditory spectrum
            rec_aud = cochleo_tools._cor2aud(A, **self.params)
            
            # update residual
            residual = X - rec_aud
            
            n_iter += 1
                
        self.sp_rep = A
        self.rec_aud = rec_aud


class CochleoIHTSketch(CochleoSketch):
    """ Iterative Hard Thresholding on an auditory spectrum 
    
    Inherit from CochleoSketch and only implements a different sparisication
    method
    """
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CochleoIHTSketch, self).__init__(
            original_sig=original_sig, **kwargs)
        
        self.params['max_iter'] = 5     # number of IHT iterations
        self.params['n_inv_iter'] = 2   # number of reconstructive steps
        for k in kwargs:
            self.params[k] = kwargs[k]
    
    def get_sig(self):
        strret = '_bands-%d_%diter_frmlen%d' % (self.params['n_bands'],
                                            self.params['max_iter'],
                                            self.params['frmlen'])
        return strret
    
    def sparsify(self, sparsity, **kwargs):
        """ sparsification is performed using the 
        Iterative Hard Thresholding Algorithm """
        L = sparsity
        if  self.cochleogram.y5 is None:
            self.cochleogram.build_aud()
        
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        cand_rep = np.array(self.cochleogram.y5)
            
        A = np.zeros(cand_rep.shape)
        original = self.cochleogram.invert(nb_iter=1, init_vec=self.cochleogram.data)
        original /= np.max(original)
        original *= 0.9
        residual = np.copy(original)
        
        n_iter = 0
        res_coch = cochleo_tools.Cochleogram(residual, **self.params)
        while n_iter < self.params['max_iter']:
            print "IHT Iteration %d"%n_iter
            if n_iter>0 or cand_rep is None:    
                            
                res_coch.build_aud()
                projection = np.array(res_coch.y5)
            else:
                projection = cand_rep
            # sort the elements            
            A_buff = A + projection
            A_flat = A_buff.flatten()
            idx_order = np.abs(A_flat).argsort()
            A = np.zeros(A_flat.shape)
            A[idx_order[-L:]] = A_flat[idx_order[-L:]]
            A = A.reshape(A_buff.shape)
                
            rec_a = res_coch.invert(v5=A, init_vec=original,
                                    nb_iter=self.params['n_inv_iter'])

            rec_a /= np.max(rec_a)
            rec_a *= 0.9
            residual = original - rec_a;
            res_coch.data = residual
            
            n_iter += 1
                
        self.sp_rep = A
        self.rec_a = rec_a
    
    def synthesize(self, sparse = False):
        ''' synthesize the sparse rep or the original rep?'''
        if sparse:
            if self.rec_a is not None:
                return Signal(self.rec_a, self.orig_signal.fs)
            v5 = self.sp_rep
        else:
            v5 = np.array(self.cochleogram.y5)

        # initialize invert
        init_vec = self.cochleogram.init_inverse(v5)
        # then do 20 iteration (TODO pass as a parameter)
        return Signal(
            self.cochleogram.invert(v5, init_vec, nb_iter=10, display=False),
            self.orig_signal.fs)

class SWSSketch(AudioSketch):
    """ Sine Wave Speech """

    def __init__(self, orig_sig=None, **kwargs):

        self.params = {'n_formants': 5,
                       'time_step': 0.01,
                       'windowSize': 0.025,
                       'preEmphasis': 50,
                       'script_path': '/home/manu/workspace/audio-sketch/src/tools'}

        for key in kwargs:
            self.params[key] = kwargs[key]
        self.call_str = 'praat %s/getsinewavespeech.praat' % self.params[
            'script_path']
        if orig_sig is not None:
            self.recompute(orig_sig, **kwargs)
        

    def get_sig(self):
        strret = '_nformants-%d' % (self.params['n_formants'])
        return strret

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
            self.rep.spectrogram(
                2 ** int(np.log2(self.params['windowSize'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['time_step'] * self.rep.fs)),
                order=1, log=True, ax=ax)
        else:
            self.sp_rep.spectrogram(
                2 ** int(np.log2(self.params['windowSize'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['time_step'] * self.rep.fs)),
                order=1, log=True, ax=ax)

    def fgpt(self):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def _extract_sws(self, signal, kwargs):
        """ internal routine to extract the sws"""
        for key in kwargs:
            self.params[key] = kwargs[key]

        if signal is None or not os.path.exists(signal):
            raise ValueError("A valid path to a wave file must be provided")
        self.current_sig = signal        
        os.system('%s %s %1.3f %d %1.3f %d' % (self.call_str, signal,
                                               self.params['time_step'],
                                               self.params['n_formants'],
                                               self.params['windowSize'],
                                               self.params['preEmphasis']))
    # Now retrieve the coefficients and the resulting audio
        self.formants = []
        for forIdx in range(1, 4):
            formant_file = signal[:-4] + '_formant' + str(forIdx) + '.mtxt'
            fid = open(formant_file, 'rU')
            vals = fid.readlines()
            fid.close()  # remove first 3 lines and convert to numpy
            self.formants.append(np.array(vals[3:], dtype='float'))

    def recompute(self, signal=None, **kwargs):
        self._extract_sws(signal, kwargs)

        # save as the original signal if it's not already set
        if self.orig_signal is None:
            self.orig_signal = Signal(signal, normalize=True)
        # and the audio we said
        self.rep = Signal(signal[:-4] + '_sws.wav', normalize=True)

    def sparsify(self, sparsity):
        """ use sparsity to determine the number of formants and window size/steps """

        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= self.rep.length

        new_tstep = float(
            int(sparsity) / self.params['n_formants']) / float(self.rep.fs)
        new_wsize = new_tstep * 2
        print "New time step of %1.3f seconds" % new_tstep
        self._extract_sws(self.current_sig, {'time_step':
                          new_tstep, 'windowSize': new_wsize})
        self.sp_rep = Signal(
            self.current_sig[:-4] + '_sws.wav', normalize=True)


class KNNSketch(AudioSketch):
    """ Reconstruct the target audio from most similar
        ones in a database """

    def __init__(self, original_sig=None, **kwargs):
        """ creating a KNN-based sketchifier """

            # default parameters
        self.params = {
            'n_frames': 100000,
            'shuffle': 0,
            'wintime': 0.032,
            'steptime': 0.008,
            'sr': 16000,
            'frame_num_per_file': 2000,
            'features': ['zcr', 'OnsetDet', 'energy', 'specstats'],  # ,'mfcc','chroma','pcp']
            'location': '',
            'n_neighbs': 3,
            'l_filt': 3}

        for key in kwargs:
            self.params[key] = kwargs[key]

        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
    
    def __del__(self):
        """ some cleanup of instantiated variables """
        try:
            del self.l_feats_all, self.l_magspecs_all, self.learn_files_all
            del self.sp_rep, self.sp_rep
        except:
            pass

    def synthesize(self, sparse=False):
        if not sparse:
            return self.rep
        else:
            return self.sp_rep

    def get_sig(self):
        strret = '_nframes-%d_seed-%d_%dNN' % (self.params['n_frames'],
                                               self.params['shuffle'],
                                               self.params['n_neighbs'])
        return strret

    def represent(self, fig=None, sparse=False):
        """ plotting the sinewave speech magnitude spectrogram"""
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()
        if not sparse:
            self.rep.spectrogram(
                2 ** int(np.log2(self.params['wintime'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['steptime'] * self.rep.fs)),
                order=1, log=True, ax=ax)
        else:
            self.sp_rep.spectrogram(
                2 ** int(np.log2(self.params['wintime'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['steptime'] * self.rep.fs)),
                order=1, log=True, ax=ax)

    def fgpt(self):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def recompute(self, signal=None, **kwargs):
        for key in kwargs:
            self.params[key] = kwargs[key]

        """ recomputing the target """
        if signal is not None:
            self.orig_signal = Signal(signal, normalize=True, mono=True)

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")

        # Loading the database
        from scipy.io import loadmat
        from feat_invert.features import load_data_one_audio_file
        savematname = '%slearnbase_allfeats_%d_seed_%d.mat' % (self.params['location'],
                                                               self.params[
                                                               'n_frames'],
                                                               self.params['shuffle'])
        lstruct = loadmat(savematname)
        self.l_feats_all = lstruct['learn_feats_all']
        self.l_magspecs_all = lstruct['learn_magspecs_all']
        self.learn_files_all = lstruct['learn_files']

        # Loading the features for the signal
        [self.t_magspecs,
         self.t_feats,
         self.t_data] = load_data_one_audio_file(signal,
                                                self.params[
                                                'sr'],
                                                sigma_noise=0,
                                                wintime=self.params['wintime'],
                                                steptime=self.params['steptime'],
                                                max_frame_num_per_file=3000,
                                                startpoint=0,
                                                features=self.params['features'])

        # Retrieve the best candidates using the routine
        from feat_invert.regression import ann
        from feat_invert.transforms import gl_recons

        n_feats = self.t_feats.shape[1]

        estimated_spectrum, neighbors = ann(self.l_feats_all[:, 0:n_feats].T,
                                            self.l_magspecs_all.T,
                                            self.t_feats.T,
                                            self.t_magspecs.T,
                                            K=self.params['n_neighbs'])

        win_size = int(self.params['wintime'] * self.params['sr'])
        step_size = int(self.params['steptime'] * self.params['sr'])
        # sliding median filtering ?
        if self.params['l_filt'] > 1:
            from scipy.ndimage.filters import median_filter
            estimated_spectrum = median_filter(
                estimated_spectrum, (1, self.params['l_filt']))

        # reconstruction
        init_vec = np.random.randn(step_size * estimated_spectrum.shape[1])
        self.rep = Signal(gl_recons(estimated_spectrum, init_vec, 010,
                                    win_size, step_size, display=False),
                          self.params['sr'], normalize=True)

    def sparsify(self, sparsity):
        """ sparsity determines the number of used neighbors? """
        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= self.rep.length

        # for now just calculate the number of needed coefficients
        self.sp_rep = self.rep
