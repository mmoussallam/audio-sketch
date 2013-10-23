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
    
    def __init__(self,scales, biaises, Wfs, Wts, lambdas,debug_level=0,entropic=True, **kwargs):
        # caling superclass constructor
        
        super(PenalizedMDCTDico,self).__init__(scales, **kwargs)
        self.biaises = biaises
        self.wfs = Wfs
        self.wts = Wts
        self.lambdas = lambdas
        self.debug = debug_level
        self.entropic = entropic
    def initialize(self, residual_signal):
        ''' Create the collection of blocks specified by the MDCT sizes '''
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for scale, biais, Ws, Wt , lambd in zip(self.sizes, self.biaises, self.wfs,self.wts, self.lambdas):
            self.blocks.append(PenalizedMDCTBlock(scale,
                                           residual_signal, biais, Ws, Wt, lambd,
                                           debug_level=self.debug,entropic=self.entropic))

    def update(self, residualSignal, iteratioNumber=0, debug=0):
        ''' Update the projections in each block, only where it needs to be done as specified '''
        self.max_block_score = 0
        self.best_current_block = None
        
        # update each block
        for block in self.blocks:
            startingTouchedFrame = int(
                math.floor(self.starting_touched_index / (block.scale / 2)))
            if self.ending_touched_index > 0:
                endingTouchedFrame = int(math.floor(self.ending_touched_index /
                                        (block.scale / 2))) + 1  # TODO check this
            else:
                endingTouchedFrame = -1
            _Logger.info("block: " + str(block.scale) + " : " +
                         str(startingTouchedFrame) + " " + str(endingTouchedFrame))

            block.update(
                residualSignal, startingTouchedFrame, endingTouchedFrame)

            if abs(block.max_value) > self.max_block_score:
#                print block.max_value
                self.max_block_score = abs(block.max_value)
                self.best_current_block = block

    def get_best_atom(self, debug):
        if self.best_current_block == None:            
#            import matplotlib.pyplot as plt
#            plt.figure()
#            for bI, b in enumerate(self.blocks):
#                plt.subplot(len(self.blocks),1, bI+1)
#                plt.plot(b.pen_projs_matrix)
#                plt.plot(b.projs_matrix,'k--')
#            plt.show()
#            raise ValueError("no best block constructed, make sure inner product have been updated")
            print "FALLBACK MODE : the penalization has produced a weird result"        
            for block in self.blocks:
                self.best_score_tree = 0
                self.pen_projs_matrix = 0
                bmax = np.abs(block.projs_matrix).max()
                if bmax > self.max_block_score:
#                    print bmax
                    self.best_current_block = block
                    block.maxIdx = np.abs(block.projs_matrix).argmax()
                    block.max_value = block.projs_matrix[block.maxIdx]
                    
        if debug > 2:
            self.best_current_block.plot_proj_matrix()

        best_atom = self.best_current_block.get_max_atom()
#        print best_atom
        for blocki in range(len(self.blocks)):            
            self.blocks[blocki].update_mask(best_atom)

        return best_atom
    
class PenalizedMDCTBlock(Block):
    """ inherit from Block, change the selection by using
    a penalty based on a predefined occurence matrix W 
    
    Given a Wf matrix and a Wt matrix that will associate a weight
    to any chosen atom to all the projection: use this to penalize the 
    normalized projections
    """
    
    def __init__(self, length=0, res_sig=None, biais=None, Wf=None, Wt=None, lambd=0.01,
                 debug_level=None, entropic=True):
        
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
        self.pen_projs_matrix = np.zeros(len(self.framed_data_matrix))
        if biais is None:
           self.biais = np.zeros_like(self.projs_matrix) 
        else:
           self.biais = biais
        
        self.Wf = Wf
        self.Wt = Wt 
        self.entropic = entropic
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
#        self.entropies = self.zeros_like(self.pen_mask)
        
        self.const = (np.log(2)/2.0)
        self.entropies = self._entropy(self.pen_mask)
    
    def update_mask(self, new_atom):
        """ Update the current mask by adding the contribution of pairwise products
            with the given index"""
        if self.Wf is not None: 
            # convert last selected atom into an index
            # so far it will only be a frequency index
#            freq = int(new_atom.reduced_frequency *  new_atom.fs)
            # The Ws matrix is assumed scaled to this frequency
            atom_idx_in_w =  int(new_atom.reduced_frequency * self.scale)
#            print self.add_mask.shape, self.W.shape
#            self.add_mask += self.W[atom_idx_in_w,:]
#            self.add_mask = self.Wf[atom_idx_in_w,:]
#            # now tile it and add to the penalty mask term
#            add_term = np.tile(self.add_mask, self.frame_num)             
##            add_term = add_term[:self.projs_matrix.shape[0]]  
#            self.pen_mask += add_term
#            self.entropies = self._entropy(self.pen_mask)
            # replicate it only on neighboring frames: don't need to 
            # penalize far from this point
            new_mask = self.Wf[atom_idx_in_w,:]
            trans_frame = new_atom.time_position / (self.scale/2)
            # HEURISTIC HERE: SHOULD BE REPLACED BY A Wt matrix
            if self.Wt is not None:
                nb_tile_frames = self.Wt
            else:
                nb_tile_frames = int(np.log2(new_atom.length))
            
#            print nb_tile_frames, self.frame_num
            add_term = np.tile(new_mask, nb_tile_frames)
            start_pos = max(0,self.scale/4 + (trans_frame- nb_tile_frames/2)*(self.scale/2)) 
            L = min(add_term.shape[0], self.pen_mask.shape[0]-start_pos)
#            print nb_tile_frames, start_pos
            self.pen_mask[start_pos:start_pos+L] += add_term[:L]
            
            self.entropies[start_pos:start_pos+L] = self._entropy(self.pen_mask[start_pos:start_pos+L])
#            import matplotlib.pyplot as plt
#            plt.imshow(self.pen_mask[:(self.scale/2)*self.frame_num].reshape((self.frame_num,self.scale/2)))
#            plt.show()            

    def find_max(self):
        """Search among the inner products the one that maximizes correlation
        the best candidate for each frame is already stored in the best_score_tree """
        treeMaxIdx = self.best_score_tree.argmax()
        # get the max in the penalized projection
        maxIdx = self.pen_projs_matrix[treeMaxIdx * self.scale /
            2: (treeMaxIdx + 1) * self.scale / 2].argmax()
        
#        if maxIdx==0:
#        import matplotlib.pyplot as plt
##            plt.figure()
##            plt.plot(self.best_score_tree)
###            plt.show()
#        plt.figure()
#        plt.plot(self.pen_projs_matrix)
#        plt.plot(self.projs_matrix)
#        plt.show()
        
        self.maxIdx = maxIdx + treeMaxIdx * self.scale / 2
        self.max_value = self.projs_matrix[self.maxIdx]
#        print treeMaxIdx, maxIdx, self.maxIdx, self.max_value
        
    def _entropy(self, x):
        if self.entropic:
            p = 1.0+np.exp(-x)
            return np.log(p)/p - self.const
        else:
            return x
        
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

        # normalize the data
#        res_norm = np.linalg.norm(self.framed_data_matrix)
#        self.framed_data_matrix /= res_norm 
#        self.entropies = np.log(1.0+np.exp(-self.pen_mask))/(1.0+np.exp(-self.pen_mask)) - (np.log(2)/2.0)
#        entropies = -self.pen_mask
#        import matplotlib.pyplot as plt
#        plt.figure()
#        plt.plot(entropies)       
#        plt.plot(self.pen_mask,'r--') 
#        plt.show()
#        
        #Specificity: Use the updated penalty mask        
        try:
            parallelProjections.project_penalized_mdct(self.framed_data_matrix,
                                                       self.best_score_tree,
                                                     self.projs_matrix,
                                                     self.pen_projs_matrix,
                                                     self.entropies,
                                                     self.locCoeff,
                                                     self.post_twid_vec,
                                                     startingFrame,
                                                     endFrame,
                                                     self.scale, self.lambd,0)

        except SystemError:
            print sys.exc_info()[0]
            print sys.exc_info()[1]
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        
        # remultiplies the projections
#        self.framed_data_matrix *= res_norm
#        self.projs_matrix *= res_norm

    def draw_mask(self):
        """ draw the mask """
        import matplotlib.pyplot as plt
#        plt.figure()
        plt.imshow(self.pen_mask[:(self.scale/2)*self.frame_num].reshape((self.frame_num,self.scale/2)))
#        plt.show()


class PenalizedLOMDCTDico(PenalizedMDCTDico):
    """ with LO """
    def initialize(self, residual_signal):
        ''' Create the collection of blocks specified by the MDCT sizes '''
        self.blocks = []
        self.best_current_block = None
        self.starting_touched_index = 0
        self.ending_touched_index = -1
        for scale, biais, Ws, Wt , lambd in zip(self.sizes, self.biaises, self.wfs,self.wts, self.lambdas):
            self.blocks.append(PenalizedLOMDCTBlock(scale,
                                           residual_signal, biais, Ws, Wt, lambd,
                                           debug_level=self.debug,entropic=self.entropic))

class PenalizedLOMDCTBlock(PenalizedMDCTBlock):
    """ inherit from both LOBlock and penalized MDCT """
    def compute_transform(self, startingFrame=1, endFrame=-1):
        """ inner product computation through MDCT """
        if self.w_long is None:
            self.initialize()

        if endFrame < 0:
            endFrame = self.frame_num - 2

        # debug -> changed from 1: be sure signal is properly zero -padded
        if startingFrame < 1:
            startingFrame = 2        
        #Specificity: Use the updated penalty mask        
        try:
            parallelProjections.project_penalized_mdct(self.framed_data_matrix,
                                                       self.best_score_tree,
                                                     self.projs_matrix,
                                                     self.pen_projs_matrix,
                                                     self.entropies,
                                                     self.locCoeff,
                                                     self.post_twid_vec,
                                                     startingFrame,
                                                     endFrame,
                                                     self.scale, self.lambd,1)

        except SystemError:
            print sys.exc_info()[0]
            print sys.exc_info()[1]
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        

    def get_max_atom(self, debug=0):
        self.max_frame_idx = floor(self.maxIdx / (0.5 * self.scale))
        self.max_bin_idx = self.maxIdx - self.max_frame_idx * (0.5 * self.scale)

        # hack here : let us project the atom waveform on the neighbouring
        # signal in the FFt domain,
        # so that we can find the maximum correlation and best adapt the time-
        # shift
        Atom = atom.Atom(self.scale, 1, max((self.max_frame_idx * self.scale / 2) - self.scale / 4, 0), self.max_bin_idx, self.residual_signal.fs)
        Atom.frame = self.max_frame_idx

        Atom.mdct_value = self.max_value
        # new version : compute also its waveform through inverse MDCT
        Atom.waveform = self.synthesize_atom(value=1)
        Atom.time_shift = 0
        Atom.proj_score = 0.0

        input1 = self.framed_data_matrix[(self.max_frame_idx - 1.5) *
             self.scale / 2: (self.max_frame_idx + 2.5) * self.scale / 2]
        input2 = np.concatenate((np.concatenate(
            (np.zeros(self.scale / 2), Atom.waveform)), np.zeros(self.scale / 2)))


        if len(input1) != len(input2):
            print self.max_frame_idx, self.maxIdx, self.frame_num
            print len(input1), len(input2)
            #if debug>0:
            print "atom in the borders , no timeShift calculated"
            return Atom

        # retrieve additional timeShift
#        if self.use_c_optim:
        scoreVec = np.array([0.0])
        Atom.time_shift = parallelProjections.project_atom(
            input1, input2, scoreVec, self.scale)

        if abs(Atom.time_shift) > Atom.length / 2:
            print "out of limits: found time shift of", Atom.time_shift
            Atom.time_shift = 0
            return Atom

        self.maxTimeShift = Atom.time_shift
        Atom.time_position += Atom.time_shift

        # retrieve newly projected waveform
        Atom.proj_score = scoreVec[0]
        Atom.waveform *= Atom.proj_score

        return Atom
