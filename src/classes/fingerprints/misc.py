'''
classes.fingerprints.misc  -  Created on Sep 30, 2013
@author: M. Moussallam
'''

import struct 
import numpy as np
from math import floor, ceil
from src.classes.fingerprints.base import *

class SWSBDB(FgptHandle):
    """  A handle class for SineWave Speech based fingerprinting
    
    FGPT is built out of the frequency delta of the formants
     
    """
    def __init__(self, dbName, load=False,
                 persistent=None,dbenv=None,
                 **kwargs):
        # Call superclass constructor
        # Call superclass constructor
        super(SWSBDB, self).__init__(dbName, load=load,
                                     persistent=persistent,
                                     dbenv=dbenv)
        self.params = {'wall':True,
                       'time_max':100,
                       'time_n_bits':12,
                       'n_deltas':4,        # number of freq intervals to be considered
                       'delta_max':4000,    # maximum delta : will be quantized in delta_n_bits
                       'delta_n_bits':8}   # maximum of bits for the key is 32, so use 10 for each of 3 deltas, 8 for 4 etc..
            
        for key in kwargs:
            self.params[key] = kwargs[key]
    
        # quantization constants
        self.Q = self.params['delta_max']/(2**self.params['delta_n_bits'] -1) +1
        self.beta_r = 2**self.params['time_n_bits']
    
    def format_value(self, fileIndex, t1):
        """ Format the value according to the parameters """
        return floor(((t1 / self.params['time_max']) * (2 ** self.params['time_n_bits'] - 1)) + fileIndex * (2 ** self.params['time_n_bits']))        
    
    def read_value(self, Bin_value):
        songID = floor(Bin_value/self.beta_r)                   
        # and quantized time
        timeofocc = Bin_value-songID*(self.beta_r)        
        timeofocc = float(timeofocc)/(self.beta_r-1)*self.params['time_max'] 
        return songID, timeofocc
    
    def format_key(self, key):
        """ each key is a collection of floating point delta-frequencies 
            that need to be quantized """
        if not len(key) == self.params['n_deltas']:
            print len(key), self.params
            raise ValueError("Quantization constants not properly initialized")
        
        binkey = 0
        for k in range(len(key)):
            binkey += int(key[k]/self.Q)*(2**(k*self.params['delta_n_bits']))
        return binkey
    
    def add(self, Pairs, fileIndex):
        ''' add all key/value pairs to base '''
        for key, value in Pairs:
                    
            t1 = value                                                
            
            Bin_value = int(self.format_value(fileIndex, t1))
            Bin_key = int(self.format_key(key))
            
            # To retrieve each element
#            print Bin_value, Bin_key
            Bbin = struct.pack('<I4', Bin_value)
            Kbin = struct.pack('<I4', Bin_key)
            
            try:
                self.dbObj.put(Kbin, Bbin)
            except db.DBKeyExistError:
                if self.params['wall']:
                    print "Warning existing Key/Value pair " + str(Bin_key) + ' ' + str(Bin_value)
    
    def get(self, key):
        ''' retrieve all values associated with key '''
        Bin_key = self.format_key(key)
#        print key, Bin_key
        Kbin = struct.pack('<I4', Bin_key)
        
        # retrieving in db
        # since multiple data can be retrieved for one key, use a cursor
        if self.cursor is None:
            self.cursor = self.dbObj.cursor()
        Bbin = self.cursor.get(Kbin, flags=db.DB_SET)
        estTime = []
        fileIdx = []

        if Bbin is None:
            return estTime, fileIdx
        # iterating and populating candidates 
        for _ in range(self.cursor.count()):            
            B = struct.unpack('<I4', Bbin[1])
            # translate to file index and time of occurence
            f , t = self.read_value(B[0])
#            print f,t             
            fileIdx.append(f)
            estTime.append(t)
            # go to next 
            Bbin = self.cursor.next_dup()

        return estTime, fileIdx

    def _build_pairs(self, fgpt, params, offset):
        diffmatrix = np.abs(np.diff(fgpt, 1, axis=0))
        # iterating on the time axis:
        keys = []
        values = []
        for t in np.arange(.0, diffmatrix.shape[1]):
            values.append(t*params['time_step'] + offset)
            keys.append(diffmatrix[:,t].tolist())
        return keys, values
    
    def populate(self, fgpt, params, fileIndex, offset=0, max_pairs=None):
        ''' Populate the database using the given fingerprint and parameters 
        
        Here each column of the fgpt matrix will serve as a key and the value is the time 
        stamp associated (uniformly sampled on the time axis)
        '''
        keys, values = self._build_pairs(fgpt, params, offset)        
                
        self.add(zip(keys, values), fileIndex)

    def retrieve(self, fgpt, params, offset=0, nbCandidates=10, precision=1.0):
        if not isinstance(fgpt, np.ndarray):
            raise TypeError('Given fingerprint is not a Numpy Array')
                
        histogram = np.zeros((floor(self.params['time_max'] / precision), nbCandidates))                       
        
        keys, values = self._build_pairs(fgpt, params, offset)
        # results is a list of histogram coordinates, for each element, we need to increment
        # the corresponding value in histogram by one.
        results = map(self.get , keys)
        # or maybe we can just sum all the values in the desired dimension?
#        print keys[1:10]
#        print results
        for keyIdx in range(len(keys)): 
#            if len(results[keyIdx][0])>1:
#                print  results[keyIdx]          
            histogram[results[keyIdx]] +=1            

        # voting for best candidate
        return histogram

    def draw_fgpt(self, fgpt, params, ax=None):
        """ illustrate the fingerprint considered"""
        diffmatrix = np.abs(np.diff(fgpt, 1, axis=0))
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
        ax.plot(diffmatrix.T,'+')
        plt.xlabel('Time')
        plt.ylabel('Formants differences')