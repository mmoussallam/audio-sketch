#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
'''
manu_sandbox.pymp_objects  -  Created on Oct 8, 2013
@author: M. Moussallam
'''


from PyMP.mdct.block import *
from PyMP.mdct.dico import *
from PyMP.log import Log
_Logger = Log("PenalizedMDCTBlock")

class PenalizedMDCTDico(Dico):
    """ inherit from Dico, apply a penalty mask based on 
    a Boltzmann machine model (biais + co-occurence) """
    
    def __init__(self,scales, biaises, Ws, lambdas, **kwargs):
        # caling superclass constructor
        
        super(PenalizedMDCTDico,self).__init__(scales, **kwargs)
        self.biaises = biaises
        self.ws = Ws
        self.lambdas = lambdas
        
    def initialize(self, residual_signal):
        ''' Create the collection of blocks specified by the MDCT sizes '''
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for scale, biais, Ws , lambd in zip(self.sizes, self.biaises, self.ws, self.lambdas):
            self.blocks.append(PenalizedMDCTBlock(scale,
                                           residual_signal, biais, Ws, lambd))

    def get_best_atom(self, debug):
        if self.best_current_block == None:
            raise ValueError("no best block constructed, make sure inner product have been updated")

        if debug > 2:
            self.best_current_block.plot_proj_matrix()

        best_atom = self.best_current_block.get_max_atom()
        for blocki in range(len(self.blocks)):            
            self.blocks[blocki].update_mask(best_atom)

        return best_atom
    
class PenalizedMDCTBlock(Block):
    """ inherit from Block, change the selection by using
    a penalty based on a predefined occurence matrix W 
    
    Should not need to modify something else than the compute_transform
    """
    
    def __init__(self, length=0, res_sig=None, biais=None, W=None, lambd=0.01,
                 debug_level=None):
        
        if debug_level is not None:
            _Logger.set_level(debug_level)

        self.scale = length
        self.residual_signal = res_sig
        self.frame_len = length / 2   
            
        if self.residual_signal == None:
            raise ValueError("no signal given")

        self.framed_data_matrix = self.residual_signal.data
        self.frame_num = len(self.framed_data_matrix) / self.frame_len
        self.projs_matrix = np.zeros(len(self.framed_data_matrix))
        if biais is None:
           self.biais = np.zeros_like(self.projs_matrix) 
        else:
           self.biais = biais
        
        self.W = W 
        
        self.lambd = lambd
        _Logger.info('new PenalizedMDCTBlock block constructed size : ' + str(self.scale))
        
        # initialize the penalty mask: a priori this should only by the biais
        if not self.biais.shape == self.projs_matrix.shape:
#            print "extending"
            self.pen_mask = np.tile(self.biais, self.frame_num+1)   
            self.pen_mask = self.pen_mask[:self.projs_matrix.shape[0]]      
        else:
            self.pen_mask = self.biais    
        self.add_mask = np.zeros_like(self.biais)
        
    def update_mask(self, new_atom):
        """ Update the current mask by adding the contribution of pairwise products
            with the given index"""
        if self.W is not None: 
            # convert last selected atom into an index
            # so far it will only be a frequency index
#            freq = int(new_atom.reduced_frequency *  new_atom.fs)
            # The Ws matrix is assumed scaled to this frequency
            atom_idx_in_w =  int(new_atom.reduced_frequency * self.scale)
#            print self.add_mask.shape, self.W.shape
            self.add_mask += self.W[atom_idx_in_w,:]
            # now tile it and add to the penalty mask term
            add_term = np.tile(self.add_mask, self.frame_num+1) 
            add_term = add_term[:self.projs_matrix.shape[0]]  
            
            self.pen_mask += add_term
    
    # inner product computation through MDCT
    def compute_transform(self, startingFrame=1, endFrame=-1):
        """ inner product computation through MDCT """
        if self.w_long is None:
            self.initialize()

        if endFrame < 0:
            endFrame = self.frame_num - 2

        # debug -> changed from 1: be sure signal is properly zero -padded
        if startingFrame < 1:
            startingFrame = 2

        # @TODO REMOVE: ONLY FOR BENCHMARKING
        self.nb_up_frame += endFrame-startingFrame
        # Wrapping C code call for fast implementation
#        if self.use_c_optim:
        
        # Specificity: Use the updated penalty mask        
        try:
            parallelProjections.project_penalized_mdct(self.framed_data_matrix, self.best_score_tree,
                                             self.projs_matrix,
                                             self.pen_mask,
                                             self.locCoeff,
                                             self.post_twid_vec,
                                             startingFrame,
                                             endFrame,
                                             self.scale, self.lambd)

        except SystemError:
            print sys.exc_info()[0]
            print sys.exc_info()[1]
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise