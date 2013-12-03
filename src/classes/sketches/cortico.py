'''
classes.sketches.corticosketch  -  Created on Jul 25, 2013
@author: M. Moussallam
'''
import os.path as op
from src.classes.sketches.base import *
from src.tools import  cochleo_tools

class CorticoSketch(AudioSketch):
    """ meta class for all corticogram-based sketches     
    
    Subclass should implement their own sparsification method
    """
    
    def __init__(self, obj=None, **kwargs):
        # add all the parameters that you want        
        
        self.params = {'n_bands': 64,
                       'shift':0,                       
                       'rv':[1, 2, 4, 8, 16, 32],
                       'sv':[0.5, 1, 2, 4, 8],
                       'pre_comp':None,
                       'rep_class': cochleo_tools.Corticogram}
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
        strret = '%drates_%scales' % (len(self.params['rv']),
                                      len(self.params['sv']))
        return strret

    def synthesize(self, sparse=False):
        ''' synthesize the sparse rep or the original rep?'''
        if sparse:
#            cor = self.cort
            # sparse auditory spectrum should already have been computed
#            if self.rec_aud is None:
            #self.rec_aud = cochleo_tools._cor2aud(self.sp_rep, **self.params)
            #v5 = np.abs(self.rec_aud).T
            return Signal( self.cort.invert_signal(self.sp_rep), self.orig_signal.fs)
        else:
            return Signal( self.cort.invert_signal(self.rep), self.orig_signal.fs)
            # inverting the corticogram
#            v5 = np.abs(self.rep.invert()).T                    
#        
#            # then do 20 iteration (TODO pass as a parameter)
#            if self.orig_signal is not None:
#                return Signal(
#                              self.cort.coch.invert(v5, self.orig_signal.data, 
#                                 nb_iter=self.params['n_inv_iter'], 
#                                 display=False),
#                              self.orig_signal.fs)
#            else:
#                # initialize invert        
#                init_vec = self.coch.init_inverse(v5)
#                return Signal(
#                self.coch.invert(v5, init_vec, nb_iter=self.params['n_inv_iter'], display=False),
#                8000)

    def represent(self, fig=None, sparse=False):
        if fig is None:
            fig = plt.figure()

        if sparse:            
            self.cort.plot_cort(fig= fig, cor=self.sp_rep)
        else:
            self.cort.plot_cort(fig= fig)

    def fgpt(self, sparse=True):
        """ return the 4-D sparsified representation """
        if sparse:
            return self.sp_rep
        return self.rep

    def recompute(self, signal=None, **kwargs):
        ''' recomputing the cochleogram'''
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        if self.params['pre_comp'] is not None:
            in_name = "%s_seg_%d.%s"%(kwargs['sig_name'], kwargs['segIdx'], 'npy')
            target = op.join(self.params['pre_comp'],in_name)
#            print "Looking for %s"%target
            if op.exists(target):
                self.rep = np.load(target)                
                return
            else:
                print " --- not Found"
        
        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        
        if self.params.has_key('downsample'):
            self.orig_signal.downsample(self.params['downsample'])
                
   #self.coch = cochleo_tools.Cochleogram(self.orig_signal.data, **self.params)
        #self.coch.build_aud()
        #self.cort = cochleo_tools.Corticogram(self.coch, **self.params)
        self.cort = self.params['rep_class'](self.orig_signal.data, **self.params)
        self.cort.build_cor()
        self.rep = np.array(self.cort.cor)


class CorticoPeaksSketch(CorticoSketch):
    """ Peack Picking on the 4-D corticogram as the sparsification process
    """
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CorticoPeaksSketch, self).__init__(
            original_sig=original_sig, **kwargs)
                
        self.params['n_inv_iter'] = 2   # number of reconstructive steps
        self.params['sub_slice'] = None
        
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
            d = np.diff(np.swapaxes(np.abs(self.rep), 0, id), axis=0)
            
            self.sp_rep = np.swapaxes(self.sp_rep, 0, id)
            self.sp_rep[:-1,...] &= d < 0
            self.sp_rep[1:,...] &= d > 0
            
            self.sp_rep = np.swapaxes(self.sp_rep, 0, id)

#        self.sp_rep = self.sp_rep.astype(int)
#        r_indexes = np.flatnonzero(self.sp_rep)        
#        r_values = self.rep.flatten()[r_indexes]
#        inds = np.abs(r_values).argsort()
#        
#        self.sp_rep = np.zeros_like(self.rep.flatten(), complex)
##        self.sp_rep[r_indexes[inds[-sparsity:]]] = r_values[inds[-sparsity:]]
#        self.sp_rep = np.reshape(self.sp_rep, self.rep.shape)
        # no only keep the k biggest values
        

class CorticoSubPeaksSketch(CorticoSketch):
    """ Peack Picking on the 4-D corticogram as the sparsification process    
        But limited to only one of the Scale/Rate combination
        
    """
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CorticoSubPeaksSketch, self).__init__(
            original_sig=original_sig, **kwargs)
                
        self.params['n_inv_iter'] = 2   # number of reconstructive steps
        self.params['sub_slice'] = (0,len(self.params['rv']))
        
        for k in kwargs:
            self.params[k] = kwargs[k]

    def get_sig(self):
        strret = '%d_over_%d_rates_%d_over_%d_scales_' % (self.params['sub_slice'][0],
                                                len(self.params['rv']),
                                                self.params['sub_slice'][1],
                                                len(self.params['sv']))
        return strret
    
    def sparsify(self, sparsity, **kwargs):
        """ Sparfifying using plain peak picking """
        
        if self.rep is None:
            self.rep = self.cort.cor
        
        if self.rep is None:
            raise ValueError("Not computed yet!")

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
#            print self.params['f_width'], self.params['t_width']

        self.sp_rep = np.zeros_like(self.rep)
        
        # target sub graph
        (scaleIdx, rateIdx) = self.params['sub_slice']
        sub_rep = self.rep[scaleIdx,rateIdx,:,:].T
#        print sub_rep.shape
        # naive implementation: cut in non-overlapping zone and get the max
        (n_bins, n_frames) = sub_rep.shape
        
        self.params['f_width'] = int(n_bins / np.sqrt(sparsity))
        self.params['t_width'] = int(n_frames / np.sqrt(sparsity))
        
        (f, t) = (self.params['f_width'], self.params['t_width'])
        
#        print range(0, (n_frames / t) * t, t)
#        print range(0, (n_bins / f) * f, f)
        
        for x_ind in range(0, (n_frames / t) * t, t):
            for y_ind in range(0, (n_bins / f) * f, f):
                rect_data = sub_rep[y_ind:y_ind + f, x_ind:x_ind + t]
                
                
#                if len(rect_data) > 0 and (np.sum(rect_data ** 2) > 0):
                f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
                # add the peak to the sparse rep
                self.sp_rep[scaleIdx,rateIdx,x_ind + t_index,
                            y_ind + f_index] = rect_data[f_index, t_index]
        # no only keep the k biggest values

    def fgpt(self, sparse=True):
        """ return the 2-D sparsified representation (only the sub-scale/rate plot)"""
        (scaleIdx, rateIdx) = self.params['sub_slice']
        if sparse:
            return np.abs(self.sp_rep[scaleIdx,rateIdx,:,:]).T
        
        else:
            raise ValueError('Not intended for non sparse fpgt')

#    def represent(self, fig=None, sparse=False):
#        if fig is None:
#            fig = plt.figure()
#
#        if sparse:            
#            self.cort.plot_cort(fig= fig, cor=self.sp_rep, binary=True)
#        else:
#            self.cort.plot_cort(fig= fig)

class CorticoIndepSubPeaksSketch(CorticoSketch):
    """ Independently sparsify in each of the sub representation (scale/rate plot)"""
    def __init__(self, original_sig=None, **kwargs):
        # add all the parameters that you want
        super(CorticoIndepSubPeaksSketch, self).__init__(
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
#            print self.params['f_width'], self.params['t_width']

        self.sp_rep = np.zeros_like(self.rep)
        
        # For each target sub graph
        for scaleIdx in range(self.sp_rep.shape[0]):
            for rateIdx in range(self.sp_rep.shape[1], self.sp_rep.shape[1]):        
                                
                
                sub_rep = self.rep[scaleIdx,rateIdx,:,:].T        
                # naive implementation: cut in non-overlapping zone and get the max
                (n_bins, n_frames) = sub_rep.shape
                
                self.params['f_width'] = int(n_bins / np.sqrt(sparsity))
                self.params['t_width'] = int(n_frames / np.sqrt(sparsity))
                
                (f, t) = (self.params['f_width'], self.params['t_width'])
                
                for x_ind in range(0, (n_frames / t) * t, t):
                    for y_ind in range(0, (n_bins / f) * f, f):
                        rect_data = sub_rep[y_ind:y_ind + f, x_ind:x_ind + t]
                        
                        f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
                        # add the peak to the sparse rep
                        self.sp_rep[scaleIdx,rateIdx,x_ind + t_index,
                                    y_ind + f_index] = rect_data[f_index, t_index]
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
        
        self.params['max_iter'] = 3     # number of IHT iterations
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
        
        if self.cort.coch.y5 is None:
            self.cort.coch.build_aud()
        
        for key in kwargs:
            self.params[key] = kwargs[key]

        # We go back and forth from the auditory spectrum to the 4-D corticogram                
        X = np.array(self.cort.coch.y5).T
        # dimensions
        K1    = len(self.params['rv']);     # of rate channel
        K2    = len(self.params['sv']);     # of scale channel
        (N2, M1)  = X.shape    # dimensions of auditory spectrogram
        
        # initialize output and residual
        A = np.zeros((K2,2*K1,N2,M1), complex)        
        residual = X
        
        n_iter = 0
        oldlist = []
        while n_iter < self.params['max_iter']:
            print "IHT Iteration %d"%n_iter       
            A_old = np.copy(A)     
            # build corticogram                  
            projection = cochleo_tools._build_cor(np.abs(residual), **self.params)

            # sort the elements and hard threshold        
            A_buff = A + projection
            A_flat = A_buff.flatten()
            idx_order = np.abs(A_flat).argsort()
            A = np.zeros(A_flat.shape, complex)
            A[idx_order[-L:]] = A_flat[idx_order[-L:]]
            A = A.reshape(A_buff.shape)
            

            # Reconstruct auditory spectrum
            rec_aud = cochleo_tools._cor2aud(A, **self.params)
            
            
#            newlist = A.flatten().nonzero()[0]
#            if len(oldlist)>0:              
#                print "%1.2f of indices in common"%(float(len(np.intersect1d(oldlist, newlist)))/float(len(oldlist)))
#            oldlist = newlist
            
            # update residual
#            print np.max(X), np.max(np.abs(rec_aud))
            residual = X - rec_aud
            
            
            n_iter += 1
                
        self.sp_rep = A
        self.rec_aud = rec_aud





###############################################################################






class QuorticoSketch(AudioSketch):
    """ meta class for all corticogram-based sketches     
    
    Subclass should implement their own sparsification method
    """
    
    def __init__(self, obj=None, **kwargs):
        # add all the parameters that you want        
        
        self.params = {'n_bands': 64,
                       'shift':0,                       
                       'rv':[1, 2, 4, 8, 16, 32],
                       'sv':[0.5, 1, 2, 4, 8],
                       'pre_comp':None}
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
        strret = '%drates_%scales' % (len(self.params['rv']),
                                      len(self.params['sv']))
        return strret

#    def synthesize(self, sparse=False):
#        ''' synthesize the sparse rep or the original rep?'''
#        if sparse:
##            cor = self.cort
#            # sparse auditory spectrum should already have been computed
##            if self.rec_aud is None:
#            self.rec_aud = cochleo_tools._cor2aud(self.sp_rep, **self.params)
#            v5 = np.abs(self.rec_aud).T
#        else:
#            
#            # inverting the corticogram
#            v5 = np.abs(self.rep.invert()).T                    
#
#
#        # then do 20 iteration (TODO pass as a parameter)
#        if self.orig_signal is not None:
#            return Signal(
#                          self.coch.invert(v5, self.orig_signal.data, 
#                             nb_iter=self.params['n_inv_iter'], 
#                             display=False),
#                          self.orig_signal.fs)
#        else:
#            # initialize invert        
#            init_vec = self.coch.init_inverse(v5)
#            return Signal(
#            self.coch.invert(v5, init_vec, nb_iter=self.params['n_inv_iter'], display=False),
#            8000)

    def represent(self, fig=None, sparse=False):
        if fig is None:
            fig = plt.figure()

        if sparse:            
            self.cort.plot_cort(fig= fig, cor=self.sp_rep)
        else:
            self.cort.plot_cort(fig= fig)

    def fgpt(self, sparse=True):
        """ return the 4-D sparsified representation """
        if sparse:
            return self.sp_rep
        return self.rep

    def recompute(self, signal=None, **kwargs):
        ''' recomputing the cochleogram'''
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        if self.params['pre_comp'] is not None:
            in_name = "%s_seg_%d.%s"%(kwargs['sig_name'], kwargs['segIdx'], 'npy')
            target = op.join(self.params['pre_comp'],in_name)
#            print "Looking for %s"%target
            if op.exists(target):
                self.rep = np.load(target)                
                return
            else:
                print " --- not Found"
        
        if signal is not None:
            if isinstance(signal, str):
                # TODO allow for stereo signals
                signal = Signal(signal, normalize=True, mono=True)
            self.orig_signal = signal

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")
        
        if self.params.has_key('downsample'):
            self.orig_signal.downsample(self.params['downsample'])
                
        [self.cqt, self.f, self.t] = cqt.cqtS(self.orig_signal, self.noyau,
                            self.params['K'], self.params['freq_min'], 
                            self.params['bins'],self.params['overl'])

        #self.params['f'] = self.f
        self.quort = cochleo_tools.Corticogram(self.cqt, **self.params)
        
        self.rep = np.array(self.quort.cor)


#class CorticoPeaksSketch(CorticoSketch):
#    """ Peack Picking on the 4-D corticogram as the sparsification process
#    """
#    def __init__(self, original_sig=None, **kwargs):
#        # add all the parameters that you want
#        super(CorticoPeaksSketch, self).__init__(
#            original_sig=original_sig, **kwargs)
#                
#        self.params['n_inv_iter'] = 2   # number of reconstructive steps
#        self.params['sub_slice'] = None
#        
#        for k in kwargs:
#            self.params[k] = kwargs[k]
#
#    def sparsify(self, sparsity, **kwargs):
#        """ Sparfifying using plain peak picking """
#        
#        if self.rep is None:
#            self.rep = self.cort.cor
#        
#        if self.rep is None:
#            raise ValueError("Not computed yet!")
#        
#        self.sp_rep = np.ones(self.rep.shape, bool)
#        alldims = range(len(self.rep.shape))
#        for id in alldims:
#            # compute the diff in the first axis after swaping
#            d = np.diff(np.swapaxes(self.rep, 0, id), axis=0)
#            
#            self.sp_rep = np.swapaxes(self.sp_rep, 0, id)
#            self.sp_rep[:-1,...] &= d < 0
#            self.sp_rep[1:,...] &= d > 0
#            
#            self.sp_rep = np.swapaxes(self.sp_rep, 0, id)
#
#        self.sp_rep = self.sp_rep.astype(int)
#        r_indexes = np.flatnonzero(self.sp_rep)        
#        r_values = self.rep.flatten()[r_indexes]
#        inds = np.abs(r_values).argsort()
#        
#        self.sp_rep = np.zeros_like(self.rep.flatten(), complex)
#        self.sp_rep[r_indexes[inds[-sparsity:]]] = r_values[inds[-sparsity:]]
#        self.sp_rep = np.reshape(self.sp_rep, self.rep.shape)
#        # no only keep the k biggest values
#        
#
#class CorticoSubPeaksSketch(CorticoSketch):
#    """ Peack Picking on the 4-D corticogram as the sparsification process    
#        But limited to only one of the Scale/Rate combination
#        
#    """
#    def __init__(self, original_sig=None, **kwargs):
#        # add all the parameters that you want
#        super(CorticoSubPeaksSketch, self).__init__(
#            original_sig=original_sig, **kwargs)
#                
#        self.params['n_inv_iter'] = 2   # number of reconstructive steps
#        self.params['sub_slice'] = (0,len(self.params['rv']))
#        
#        for k in kwargs:
#            self.params[k] = kwargs[k]
#
#    def get_sig(self):
#        strret = '%d_over_%d_rates_%d_over_%d_scales_' % (self.params['sub_slice'][0],
#                                                len(self.params['rv']),
#                                                self.params['sub_slice'][1],
#                                                len(self.params['sv']))
#        return strret
#    
#    def sparsify(self, sparsity, **kwargs):
#        """ Sparfifying using plain peak picking """
#        
#        if self.rep is None:
#            self.rep = self.cort.cor
#        
#        if self.rep is None:
#            raise ValueError("Not computed yet!")
#
#        for key in kwargs:
#            self.params[key] = kwargs[key]
#
#        if sparsity <= 0:
#            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")
#        elif sparsity < 1:
#            # interprete as a ratio
#            sparsity *= np.sum(self.rep.shape)
##        else:
#            # otherwise the sparsity argument take over and we divide in
#            # the desired number of regions (preserving the bin/frame ratio)
##            print self.rep.shape[1:]
##            print self.params['f_width'], self.params['t_width']
#
#        self.sp_rep = np.zeros_like(self.rep)
#        
#        # target sub graph
#        (scaleIdx, rateIdx) = self.params['sub_slice']
#        sub_rep = self.rep[scaleIdx,rateIdx,:,:].T
##        print sub_rep.shape
#        # naive implementation: cut in non-overlapping zone and get the max
#        (n_bins, n_frames) = sub_rep.shape
#        
#        self.params['f_width'] = int(n_bins / np.sqrt(sparsity))
#        self.params['t_width'] = int(n_frames / np.sqrt(sparsity))
#        
#        (f, t) = (self.params['f_width'], self.params['t_width'])
#        
##        print range(0, (n_frames / t) * t, t)
##        print range(0, (n_bins / f) * f, f)
#        
#        for x_ind in range(0, (n_frames / t) * t, t):
#            for y_ind in range(0, (n_bins / f) * f, f):
#                rect_data = sub_rep[y_ind:y_ind + f, x_ind:x_ind + t]
#                
#                
##                if len(rect_data) > 0 and (np.sum(rect_data ** 2) > 0):
#                f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
#                # add the peak to the sparse rep
#                self.sp_rep[scaleIdx,rateIdx,x_ind + t_index,
#                            y_ind + f_index] = rect_data[f_index, t_index]
#        # no only keep the k biggest values
#
#    def fgpt(self, sparse=True):
#        """ return the 2-D sparsified representation (only the sub-scale/rate plot)"""
#        (scaleIdx, rateIdx) = self.params['sub_slice']
#        if sparse:
#            return np.abs(self.sp_rep[scaleIdx,rateIdx,:,:]).T
#        
#        else:
#            raise ValueError('Not intended for non sparse fpgt')
#
##    def represent(self, fig=None, sparse=False):
##        if fig is None:
##            fig = plt.figure()
##
##        if sparse:            
##            self.cort.plot_cort(fig= fig, cor=self.sp_rep, binary=True)
##        else:
##            self.cort.plot_cort(fig= fig)
#
#class CorticoIndepSubPeaksSketch(CorticoSketch):
#    """ Independently sparsify in each of the sub representation (scale/rate plot)"""
#    def __init__(self, original_sig=None, **kwargs):
#        # add all the parameters that you want
#        super(CorticoIndepSubPeaksSketch, self).__init__(
#            original_sig=original_sig, **kwargs)
#                
#        self.params['n_inv_iter'] = 2   # number of reconstructive steps        
#        
#        for k in kwargs:
#            self.params[k] = kwargs[k]
#
#    def sparsify(self, sparsity, **kwargs):
#        """ Sparfifying using plain peak picking """
#        
#        if self.rep is None:
#            self.rep = self.cort.cor
#        
#        if self.rep is None:
#            raise ValueError("Not computed yet!")
#
#        for key in kwargs:
#            self.params[key] = kwargs[key]
#
#        if sparsity <= 0:
#            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")
#        elif sparsity < 1:
#            # interprete as a ratio
#            sparsity *= np.sum(self.rep.shape)
##        else:
#            # otherwise the sparsity argument take over and we divide in
#            # the desired number of regions (preserving the bin/frame ratio)
##            print self.rep.shape[1:]
##            print self.params['f_width'], self.params['t_width']
#
#        self.sp_rep = np.zeros_like(self.rep)
#        
#        # For each target sub graph
#        for scaleIdx in range(self.sp_rep.shape[0]):
#            for rateIdx in range(self.sp_rep.shape[1]/2, self.sp_rep.shape[1]):        
#                                
#                
#                sub_rep = self.rep[scaleIdx,rateIdx,:,:].T        
#                # naive implementation: cut in non-overlapping zone and get the max
#                (n_bins, n_frames) = sub_rep.shape
#                
#                self.params['f_width'] = int(n_bins / np.sqrt(sparsity))
#                self.params['t_width'] = int(n_frames / np.sqrt(sparsity))
#                
#                (f, t) = (self.params['f_width'], self.params['t_width'])
#                
#                for x_ind in range(0, (n_frames / t) * t, t):
#                    for y_ind in range(0, (n_bins / f) * f, f):
#                        rect_data = sub_rep[y_ind:y_ind + f, x_ind:x_ind + t]
#                        
#                        f_index, t_index = divmod(np.abs(rect_data).argmax(), t)
#                        # add the peak to the sparse rep
#                        self.sp_rep[scaleIdx,rateIdx,x_ind + t_index,
#                                    y_ind + f_index] = rect_data[f_index, t_index]
#                # no only keep the k biggest values
#        
#class CorticoIHTSketch(CorticoSketch):
#    """ Iterative Hard Thresholding on a 4-D corticogram spectrum 
#    
#    Inherit from CorticoSketch and only implements a different sparisication
#    method
#    """
#    def __init__(self, original_sig=None, **kwargs):
#        # add all the parameters that you want
#        super(CorticoIHTSketch, self).__init__(
#            original_sig=original_sig, **kwargs)
#        
#        self.params['max_iter'] = 5     # number of IHT iterations
#        self.params['n_inv_iter'] = 2   # number of reconstructive steps
#        for k in kwargs:
#            self.params[k] = kwargs[k]
#    
#    def get_sig(self):
#        strret = '_%diter_frmlen%d' % (self.params['max_iter'],
#                                            self.params['frmlen'])
#        return strret
#    
#    def sparsify(self, sparsity, **kwargs):
#        """ sparsification is performed using the 
#        Iterative Hard Thresholding Algorithm """
#        L = sparsity
#        if  self.cort is None:
#            raise ValueError("No representation has been computed yet")
#        
#        if self.coch.y5 is None:
#            self.coch.build_aud()
#        
#        for key in kwargs:
#            self.params[key] = kwargs[key]
#
#        # We go back and forth from the auditory spectrum to the 4-D corticogram                
#        X = np.array(self.coch.y5).T
#        # dimensions
#        K1    = len(self.params['rv']);     # of rate channel
#        K2    = len(self.params['sv']);     # of scale channel
#        (N2, M1)  = X.shape    # dimensions of auditory spectrogram
#        
#        # initialize output and residual
#        A = np.zeros((K2,2*K1,N2,M1), complex)        
#        residual = X
#        
#        n_iter = 0
#        
#        while n_iter < self.params['max_iter']:
#            print "IHT Iteration %d"%n_iter       
#            A_old = np.copy(A)     
#            # build corticogram                  
#            projection = cochleo_tools._build_cor(residual, **self.params)
#
#            # sort the elements and hard threshold        
#            A_buff = A + projection
#            A_flat = A_buff.flatten()
#            idx_order = np.abs(A_flat).argsort()
#            A = np.zeros(A_flat.shape, complex)
#            A[idx_order[-L:]] = A_flat[idx_order[-L:]]
#            A = A.reshape(A_buff.shape)
#            
#            # Reconstruct auditory spectrum
#            rec_aud = cochleo_tools._cor2aud(A, **self.params)
#            
#            # update residual
#            residual = X - rec_aud
#            
#            n_iter += 1
#                
#        self.sp_rep = A
#        self.rec_aud = rec_aud