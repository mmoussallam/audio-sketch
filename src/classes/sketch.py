'''
Created on Jan 30, 2013

@author: manu
'''

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
    '''


def cochleogram(data, params):
    ''' Build a cochleogram object from a set of parameters for
    given data, in the form of an auditory spectrum ''' 
