'''
classes.fingerprints.bench  -  Created on Sep 30, 2013
@author: M. Moussallam
'''

import numpy as np
from math import ceil, floor, log
import struct
from src.classes.fingerprints.base import *
import PyMP
from PyMP.approx import Approx

class STFTPeaksBDB(FgptHandle):
    ''' handling the fingerprints based on a pairing of STFT peaks 
    
    A key is the triplet (f1, f2, delta_t) the value is the time of occurrence
    
    Attributes
    ----------
    params : dict
        a dictionary of parameters among which: *fmax*, *key_total_nbits*, *value_total_bits*
    '''    
    def __init__(self, dbName, load=False, persistent=None,dbenv=None,
                 **kwargs):
        
        # Call superclass constructor
        super(STFTPeaksBDB, self).__init__(dbName, load=load, persistent=persistent,dbenv=dbenv)
        
        # default parameters
        self.params = {'delta_t_max':3.0,
                       'fmax': 8000.0,
                       'key_total_nbits':32,
                        'f1_n_bits': 10,
                        'f2_n_bits': 10,
                        'dt_n_bits': 10,
                        'value_total_bits':32,
                        'file_index_n_bits':20,
                        'time_n_bits':12,
                        'time_max':60.0* 20.0,
                        'wall':True}
        
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        # Formatting the key - FORMAT 1 : Absolute
        self.alpha = ceil((self.params['fmax'])/(2**self.params['f1_n_bits']-1))
        self.beta = ceil((self.params['fmax'])/(2**self.params['f2_n_bits']-1))
        
        # BUGFIX remove the ceiling function cause it causes all values to be zeroes
        self.gamma = self.params['delta_t_max']/(2**self.params['dt_n_bits']-1)
#        self.gamma = ceil(self.params['delta_t_max']/(2**self.params['dt_n_bits']-1))        
#        print self.alpha, self.beta, self.gamma, self.params['f1_n_bits'], self.params['f2_n_bits'], self.params['dt_n_bits']

        self.alpha_r = 2**self.params['file_index_n_bits']
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
        """ Format the Key as [f1 , f2, delta_t] """
        (f1, f2, delta_t) = key
        return floor((f1 / self.alpha) * 2 ** (self.params['f2_n_bits'] + self.params['dt_n_bits'])) + floor((f2 / self.beta) * 2 ** (self.params['dt_n_bits'])) + floor((delta_t) / self.gamma)

    def add(self, Pairs, fileIndex):
        '''
        Putting the values in the database
        '''        
        for key, value in Pairs:
            # @TODO this is SHAZAM's format : parameterize            
            t1 = value                                                
            
            Bin_value = int(self.format_value(fileIndex, t1))
            Bin_key = int(self.format_key(key))
#            print Bin_value, Bin_key
            # To retrieve each element
            Bbin = struct.pack('<I4', Bin_value)
            Kbin = struct.pack('<I4', Bin_key)
            
            try:
                self.dbObj.put(Kbin, Bbin)
            except db.DBKeyExistError:
                if self.params['wall']:
                    print "Warning existing Key/Value pair " + str(Bin_key) + ' ' + str(Bin_value)
    
    
    def get(self, key):
        '''
        Retrieve the values in the database associated with this key
        '''
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
            fileIdx.append(f)
            estTime.append(t)
            # go to next 
            Bbin = self.cursor.next_dup()

        return estTime, fileIdx
    
    def _build_pairs(self, sparse_stft, params, offset=0, display=False,ax=None):
        ''' internal routine to build key/value pairs from sparse STFT
        given the parameters '''
        keys = []
        values = []
        if display:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            if ax is None:        
                fig = plt.figure()
                ax = fig.add_subplot(111)            
            ax.spy(sparse_stft[0,:,:], cmap=cm.bone_r, aspect='auto')
            
        peak_indexes = np.nonzero(sparse_stft[0,:,:])
        f_target_width = 3*params['f_width']
        t_target_width = 3*params['t_width']
        freq_step = float(params['fs'])/float(params['scale'])
        time_step = float(params['step'])/float(params['fs'])
#        print "Params : ",f_target_width,t_target_width,freq_step,time_step
        # then for each of them look in the target zone for other
        for pIdx in range(len(peak_indexes[0])):
            peak_ind = (peak_indexes[0][pIdx], peak_indexes[1][pIdx])
            target_points_i, target_points_j = np.nonzero(sparse_stft[0,
                                                        peak_ind[0]: peak_ind[0]+f_target_width,
                                                        peak_ind[1]: peak_ind[1]+t_target_width])
            # now we can build a pair of peaks , and thus a key
            for i in range(1,len(target_points_i)):
                f1 = np.round(float(peak_ind[0]) *freq_step)
                f2 = np.round(float(peak_ind[0]+target_points_i[i]) * freq_step)
                t1 = float(peak_ind[1]) *time_step
                # HACK HERE: rounding at 1 decimals to robustify
                delta_t = np.round(float(target_points_j[i]) *time_step, decimals=1)
                if display:                    
                    ax.arrow(peak_ind[1], peak_ind[0],target_points_j[i], target_points_i[i], head_width=0.05, head_length=0.1, fc='k', ec='k')
#                
#                print (f1, f2, delta_t) , t1
                keys.append((f1, f2, delta_t))
                values.append(t1 + offset)
        return keys, values
    
    def populate(self, fgpt, params, file_index, offset=0, debug=False, max_pairs=None, display=False,ax=None):
        """ populate by creating pairs of peaks """
        # get all non zero elements            
        keys, values = self._build_pairs(fgpt, params, offset, display=display,ax=ax)
        if self.params['wall']:
            print " %d key/value pairs"%len(keys)
        Set = list(set(zip(keys, values)))
        
        if max_pairs is not None:
            Set = Set[:max_pairs]
#            print "Limiting to %d key/value pairs"%len(Set)
        self.add(Set, file_index)
        
    def retrieve(self, fgpt, params, offset=0, nbCandidates=10, precision=1.0,debug=False):
        '''
        Retrieve in base the candidates based on the sparse rep and return best candidate
        '''
        if not isinstance(fgpt, np.ndarray):
            raise TypeError('Given fingerprint is not a Numpy Array')
                
        # refactoring : 2D histogram: col = offset, line = songIndex
        # implemented as a double dictionary : key = songIndex , value =
        # {offset : nbCount}
        histogram = np.zeros((floor(self.params['time_max'] / precision), nbCandidates))                       
        
        keys, values = self._build_pairs(fgpt, params, offset)
        # results is a list of histogram coordinates, for each element, we need to increment
        # the corresponding value in histogram by one.
        results = map(self.get , keys)
        if debug:
            print "Found %d out of %d keys"%(len(results), len(keys))
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
        """ Draw the fingerprint """
        self._build_pairs(fgpt, params, 0, display=True, ax=ax)
        
        
        
class XMDCTBDB(FgptHandle):
    '''
    PyMP approx berkeley database handle
    '''
    keyformat = 0
        # closing the db for now ?
#        self.dbObj.close();

    def __init__(self, dbName, load=False,                
                 persistent=True,dbenv=None, **kwargs):
        '''
        Constructor
        '''
        # Call superclass constructor
        super(XMDCTBDB, self).__init__(dbName, load=load, persistent=persistent,dbenv=dbenv)
                    
        # DEFAULT VALUES
        self.params = {'total_n_bits':16, # total number of bits allowed for storing the key in base
                       'fmax': 8000.0,
                       'freq_n_bits':14,
                       'time_max':600.0, # 10 minutes is the biggest allowed time interval
                       'time_res':1.0,
                       'wall':True}
        
        # populating optional parameters
        for karg in kwargs:
            self.params[karg] = kwargs[karg]
            
        # define quantization parameters
        self.beta = ceil((self.params['fmax']) / (2.0 ** self.params['freq_n_bits'] - 1.0))        
        self.t_ratio = (2.0 ** self.params['total_n_bits'] - 1) / (self.params['time_max']) 


    def __repr__(self):
        return """ 
%s handler (based on Berkeley DB): %s
Key in %d bits,
Resolution: Time: %1.3f (s) %2.2f Hz 
""" % (self.__class__.__name__, self.db_name,
       self.params['total_n_bits'],self.params['time_res'], self.beta)
    

#    def add(self, Keys , Values, fileIndex):
    def add(self, Pairs, fileIndex):
        '''
        Putting the values in the database
        '''
        
        for key, value in Pairs:

            if value > self.params['time_max']:
#                print value
                print 'Warning: Tried to add a value bigger than maximum'
                continue

            Tbin = floor(value * self.t_ratio)
            # Coding over 16bits, max time offset = 10min
            
            B = int(fileIndex * (2**self.params['total_n_bits']) + Tbin)

# K = int(floor(key)*2**(self.params['freq_n_bits'])+floor(float(key)/float(self.beta)));
            K = self.format(key)
#            print key, value
            Bbin = struct.pack('<I4', B)
            Kbin = struct.pack('<I4', K)

#            print key , K , value , B
            # search for already existing key
#            BbinEx = self.dbObj.get(Kbin);
#            if BbinEx is None:
            try:
                self.dbObj.put(Kbin, Bbin)
            except db.DBKeyExistError:
                if self.params['wall']:
                    print "Warning existing Key/Value pair " + str(key) + ' ' + str(value)


    def get(self, Key, const=0):        
        '''
        @TODO refactor so that all add and get use the same formalism
        Retrieve the values in the database
        '''
#        K = int(floor(Key)*2**(self.params['freq_n_bits'])+floor(float(Key)/float(self.beta)));
        K = self.format(Key)
        Kbin = struct.pack('<I4', K)

        # retrieving in db
        # since multiple data can be retrieved for one key, use a cursor
        if self.cursor is None:
            self.cursor = self.dbObj.cursor()
        Bbin = self.cursor.get(Kbin, flags=db.DB_SET)
#        print Bbin
#        print self.dbObj.get(Kbin);
#        nbrep = 
        estTime = []
        fileIdx = []

        if Bbin is None:
            return estTime, fileIdx
        

        for _ in range(self.cursor.count()):
            
            B = struct.unpack('<I4', Bbin[1])
#            print B , self.total_n_bits
#            estTimeI = B[0] % self.total_n_bits
            
            f , t = divmod(B[0], 2**self.params['total_n_bits'])
            
            fileIdx.append(f)
            estTime.append((t / self.t_ratio))

            Bbin = self.cursor.next_dup()

        return estTime, fileIdx



    def populate(self, fgpt, params, fileIndex, offset=0, largebases=False,
                 max_pairs=None, debug=False, display=False, ax=None):
        ''' Populate the database using the given fingerprint and parameters
        
        Here the fingerprint object is the PyMP.approx class
        parameters can all be infered directly from the object
        '''
        if not isinstance(fgpt, Approx):
            raise TypeError(
                'Given argument is not a valid PyMP.Approx Object')

        
#        " else : take all atoms and feed the database"
        F = []
        T = []
        S = []
        for atom in fgpt.atoms:
            if largebases and (log(atom.length, 2) < 13):
                continue
            S.append(log(atom.length, 2))
            F.append(atom.reduced_frequency * fgpt.fs)
            T.append(( offset + (float(atom.time_position) / float(fgpt.fs))))
            
        # look for duplicates and removes them: construct a set of zipped elements
#        print zip(F , T)
        if self.keyformat is None:
            Set = set(zip(F, T))
        elif self.keyformat == 0:
            Set = set(zip(zip(S, F), T))
#            Set = set(zip(list(set(zip(S,F))),T));

#        lst = zip(F,T);
#        for item in Set:
#            print item ,  lst.count(item)
    
        self.add(list(Set), fileIndex)
        if display:
            self.draw_fgpt(fgpt, params, ax)

    def kform(self, atom):
        if self.keyformat is None:
            return atom.reduced_frequency * atom.fs
        elif self.keyformat == 0:
            return [log(atom.length, 2), atom.reduced_frequency * atom.fs]
        

    def retrieve(self, fgpt, params, offset=0, nbCandidates=10, precision=1.0):
        '''
        Retrieve in base the candidates based on the atoms from the app and return best candidate
        '''
        if not isinstance(fgpt, Approx):
            raise TypeError(
                'Given argument is not a valid py_pursuit_Approx Object')
        # refactoring : 2D histogram: col = offset, line = songIndex
        # implemented as a double dictionary : key = songIndex , value =
        # {offset : nbCount}
        histogram = np.zeros((floor(self.params['time_max'] / precision), nbCandidates))                       
        
        # results is a list of histogram coordinates, for each element, we need to increment
        # the corresponding valu in histogram by one.
        results = map(self.get , map(self.kform, fgpt.atoms),
                      [a.time_position  / fgpt.fs for a in fgpt.atoms])
        # or maybe we can just sum all the values in the desired dimension?
        
#        print results
        for atomIdx in range(fgpt.atom_number):
            histogram[results[atomIdx]] +=1            

        # voting for best candidate
        return histogram
#        return Candidates , Offsets

    def format(self, key):
        ''' In thi function format the key according to the predefined template
        '''        
        if self.keyformat == 0:
            return int(floor(key[0]) * 2 ** (self.params['freq_n_bits']) + floor(float(key[1]) / float(self.beta)))
#            return np.int32((floor(key[1])))
        # defaut case
        return int(floor(key) * 2 ** (self.params['freq_n_bits']) + floor(float(key) / float(self.beta)))

    def draw_fgpt(self, fgpt, params, ax=None):
        """ Draw the fingerprint """
        fgpt.plot_tf()
        
class SparseFramePairsBDB(STFTPeaksBDB):
    '''
    PyMP approx berkeley database handle
    build pairs of atoms as keys
    '''
    keyformat = 0
        # closing the db for now ?
#        self.dbObj.close();

    def __init__(self, dbName, load=False,                
                 persistent=True,dbenv=None, **kwargs):
        '''
        Constructor
        '''
        # Call superclass constructor
        super(SparseFramePairsBDB, self).__init__(dbName, load=load, persistent=persistent,dbenv=dbenv)                
        
        self.params['delta_f_max'] = 667 # in hertz
        self.params['delta_f_min'] = 25 # in hertz
        self.params['delta_f_bits'] = 6 # in hertz
        self.params['delta_t_max'] = 0.5
        self.params['delta_t_min'] = 0.05
        self.params['time_res'] = 1
        self.params['freq_res'] = 22
        self.params['nb_neighbors_max'] = 5
        
        
        # populating optional parameters
        for karg in kwargs:
            self.params[karg] = kwargs[karg]

        self.alpha = ceil((self.params['fmax'])/(2**self.params['f1_n_bits']-1))
        self.beta = ceil((self.params['delta_f_max'])/(2**self.params['delta_f_bits']-1))

    def __repr__(self):
        return """ 
%s handler (based on Berkeley DB): %s
Key in %d bits,
Resolution: Time: %1.3f (s) %2.2f Hz 
""" % (self.__class__.__name__, self.db_name,
       self.params['key_total_nbits'],self.params['time_res'], self.params['freq_res'])
            

    def _build_pairs(self, sparse_rep, params, offset=0, display=False,ax=None):
        """ build pairs given a PyMP.approx object: need to find 
            pairs.. first idea: take every pairwise combination """
        keys = []
        values =[]
        if display:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            if ax is None:        
                fig = plt.figure()
                ax = fig.add_subplot(111) 
            sparse_rep.plot_tf()
        for atIdx, atom in enumerate(sparse_rep.atoms):
            f1 = atom.reduced_frequency * atom.fs
            t_anchor = (atom.time_position + atom.length/2) / atom.fs
            proximities = [((neigh.time_position + neigh.length/2) / neigh.fs) - t_anchor for neigh in sparse_rep.atoms[atIdx+1:]]
            # find the nb_max_closest
            closest = np.argsort(np.abs(proximities))[:self.params['nb_neighbors_max']]
            # find its neighbors
            for relAtomIdx in closest:
                
                neigh = sparse_rep.atoms[atIdx+1+relAtomIdx]
                neighb_f = neigh.reduced_frequency * neigh.fs
                neighb_t = (neigh.time_position + neigh.length/2) / neigh.fs
                if abs(neighb_t - t_anchor) > self.params['delta_t_max']:
                    continue
                if abs(neighb_f - f1) > self.params['delta_f_max']:
                    continue
                # in same zone: build the pair
                if display:          
#                    print  t_anchor, f1, neighb_t - t_anchor, neighb_f - f1    
#                    ax.arrow(120,120,50,50,head_width=0.05, head_length=0.1, fc='k', ec='k')  
                    ax.arrow(t_anchor, f1, neighb_t - t_anchor, neighb_f - f1, head_width=0.05, head_length=0.1, fc='k', ec='k')
#                
#                print (f1,  neighb_f-f1, neighb_t - t_anchor) , t_anchor
                keys.append((f1, neighb_f-f1, neighb_t - t_anchor))
                values.append(t_anchor + offset)
        return keys, values

    def format_key(self, key):
        """ Format the Key as [f1 , delta_f, delta_t] 
        
        The message is formated as [F1 | Sign| Delta_f | sign | Delta_t]
        """
        (f1, delta_f, delta_t) = key
        f1mult =  2 ** (self.params['delta_f_bits'] + self.params['dt_n_bits'] + 2) 
        deltafmult = 2 ** (self.params['dt_n_bits'] + 1)
        return floor((f1 / self.alpha) * f1mult) + \
              (delta_f>0) * 2**(self.params['delta_f_bits'] + self.params['dt_n_bits'] + 1) \
             + floor((abs(delta_f) / self.beta) * deltafmult) + \
             + (delta_t>0) *  2 ** (self.params['dt_n_bits']) + \
             floor((abs(delta_t)) / self.gamma)


    def retrieve(self, fgpt, params, offset=0, nbCandidates=10, precision=1.0,debug=False):
        '''
        Retrieve in base the candidates based on the sparse rep and return best candidate
        '''
        if not isinstance(fgpt, PyMP.approx.Approx):
            raise TypeError('Given fingerprint is not a PyMP approx')
                
        # refactoring : 2D histogram: col = offset, line = songIndex
        # implemented as a double dictionary : key = songIndex , value =
        # {offset : nbCount}
        histogram = np.zeros((floor(self.params['time_max'] / precision), nbCandidates))                       
        
        keys, values = self._build_pairs(fgpt, params, offset)
        # results is a list of histogram coordinates, for each element, we need to increment
        # the corresponding value in histogram by one.
        results = map(self.get , keys)
        if debug:
            print "Found %d out of %d keys"%(len(results), len(keys))
        # or maybe we can just sum all the values in the desired dimension?
#        print keys[1:10]
#        print results
        for keyIdx in range(len(keys)): 
#            if len(results[keyIdx][0])>1:
#                print  results[keyIdx]          
            histogram[results[keyIdx]] +=1            

        # voting for best candidate
        return histogram