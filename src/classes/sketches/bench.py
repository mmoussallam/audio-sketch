'''
classes.sketches.benchsketch  -  Created on Jul 25, 2013
@author: M. Moussallam
'''

from base import *

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


