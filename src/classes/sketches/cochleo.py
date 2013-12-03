'''
classes.sketches.cochleosketch  -  Created on Jul 25, 2013
@author: M. Moussallam
'''

import os, sys
SKETCH_ROOT = os.environ['SKETCH_ROOT']
sys.path.append(SKETCH_ROOT)
from src.classes.sketches.base import *
from src.tools import  cochleo_tools

class CochleoSketch(AudioSketch):
    """ meta class for all cochleogram-based sketches     
    
    Subclass should implement their own sparsification method
    """
    
    
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want        
        
        self.params = {'n_bands': 64, 'shift':0, 'fac':-2,'n_inv_iter':5, 'frmlen':8}
        self.cochleogram = None
        for key in kwargs:
            self.params[key] = kwargs[key]

        if original_sig is not None:
#            self.orig_signal = original_sig
            self.recompute(original_sig)
        

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
            sparsity *= np.prod(self.rep.shape)
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

                if len(rect_data) > 0 and (np.sum(np.abs(rect_data) ** 2) > 0):
                    f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
                    # add the peak to the sparse rep
                    self.sp_rep[y_ind + f_index,
                                x_ind + t_index] = rect_data[f_index, t_index]

        self.nnz = np.count_nonzero(self.sp_rep)
        
class CochleoIHTSketch(CochleoSketch):
    """ Iterative Hard Thresholding on an auditory spectrum 
    
    Inherit from CochleoSketch and only implements a different sparsification
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
        
        if  self.cochleogram.y5 is None:
            self.cochleogram.build_aud()
        
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        cand_rep = np.array(self.cochleogram.y5)
          
        A = np.zeros(cand_rep.shape)
        
        if sparsity>1.0:
            L = sparsity
        else:
            L = sparsity * np.prod(A.shape)
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
    
    def synthesize(self, sparse = False, cheat=False):
        ''' synthesize the sparse rep or the original rep?'''
        if sparse:
            if self.rec_a is not None and cheat:
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
