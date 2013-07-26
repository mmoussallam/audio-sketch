'''
classes.sketches.AudioSketch  -  Created on Jul 25, 2013
@author: M. Moussallam
'''

import numpy as np
import os
from PyMP import Signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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