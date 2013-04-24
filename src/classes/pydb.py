#*****************************************************************************/
#                                                                            */
#                               pydb.py                                     */
#                                                                            */
#                        Matching Pursuit Library                            */
#                                                                            */
# M. Moussallam                                             Wed Sep 07 2011  */
# -------------------------------------------------------------------------- */


import bsddb.db as db
import os
import numpy as np
from math import floor, ceil, log
import struct
import PyMP
from PyMP.approx import Approx


class FgptHandle(object):
    ''' Super class for Fingerprint 
    
    Whatever the chosen model, the handle should be able to 
    populate and retrieve keys from a BerkeleyDB database. Creation of
    the BDB object is thus always performed in the same manner
    
    Attributes
    ----------
    db_name : str
        a unique string path to the database on disk
    dbObj : bsddb.db.DB
        the db object
    persistent :  bool
        a boolean indicating whether the db is to be kept or destroy 
        when object id deleted
    
    '''    
    
    
    def __init__(self, dbName, 
                 load=False,                 
                 persistent=True):
        """ Common Constructor """
        if os.path.exists(dbName) and not load:
            os.remove(dbName)
        self.db_name = dbName
        self.dbObj = db.DB()
        
        # allow multiple key entries
        # TODO :  mettre en DB_DUPSORT
        self.dbObj.set_flags(db.DB_DUP | db.DB_DUPSORT)

        if not load:
            try:
                self.dbObj.open(self.db_name, dbtype=db.DB_HASH, flags=db.DB_CREATE)
            except:                
                raise IOError('Failed to create %s ' % self.db_name)
        else:
            if self.db_name is None:
                raise ValueError('No Database name provided')
            self.dbObj.open(dbName, dbtype=db.DB_HASH)
            print "Loaded DB:", dbName
    
        # keep in mind if the db is to be kept or destroy
        self.persistent = persistent
        # cursor object : might get instantiated later
        self.cursor = None
    
    def __del__(self):
        self.dbObj.close()
        if not self.persistent:
            if os.path.exists(self.db_name):
                print 'Destroying Db ', self.db_name
                os.remove(self.db_name)

                del self.dbObj, self.db_name


    def __repr__(self):
        return """ %s handler (based on Berkeley DB): %s""" % (self.__class__.__name__,
                                                               self.db_name)    

    def get_stats(self):
        ''' retrieve the number of keys in the table
        '''
        return self.dbObj.stat()
    
    def add(self):
        raise NotImplementedError("Not Implemented")
    
    def get(self):
        raise NotImplementedError("Not Implemented")

    def populate(self):
        raise NotImplementedError("Not Implemented")

    def retrieve(self):
        raise NotImplementedError("Not Implemented")
    

class STFTPeaksBDB(FgptHandle):
    ''' handling the fingerprints based on a pairing of STFT peaks 
    
    Attributes
    ----------
    params : dict
        a dictionary of parameters among which: *fmax*, *key_total_nbits*, *value_total_bits*
    '''    
    def __init__(self, dbName, load=False, persistent=None,
                 **kwargs):
        # Call superclass constructor
        # Call superclass constructor
        super(STFTPeaksBDB, self).__init__(dbName, load=load, persistent=persistent)
        
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
                        'time_max':60.0* 20.0}
        
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        # Formatting the key - FORMAT 1 : Absolute
        self.alpha = ceil((self.params['fmax'])/(2**self.params['f1_n_bits']-1))
        self.beta = ceil((self.params['fmax'])/(2**self.params['f2_n_bits']-1))
        self.gamma = ceil(self.params['delta_t_max']/(2**self.params['dt_n_bits']-1))
    
    

    def format_value(self, fileIndex, t1):
        """ Format the value according to the parameters """
        return floor(((t1 / self.params['time_max']) * (2 ** self.params['time_n_bits'] - 1)) + fileIndex * (2 ** self.params['file_index_n_bits']))        

    def read_value(self, Bin_value):
        songID = floor(Bin_value/2**self.params['file_index_n_bits'])                   
        # and quantized time
        timeofocc = Bin_value-songID*(2**self.params['file_index_n_bits'])        
        timeofocc = float(timeofocc)/(2**self.params['time_n_bits']-1)*self.params['time_max'] 
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
            
            Bin_value = self.format_value(fileIndex, t1)
            Bin_key = self.format_key(key)
            
            # To retrieve each element
            Bbin = struct.pack('<I4', Bin_value)
            Kbin = struct.pack('<I4', Bin_key)
            
            try:
                self.dbObj.put(Kbin, Bbin)
            except db.DBKeyExistError:
                print "Warning existing Key/Value pair " + str(key) + ' ' + str(value)
    
    
    def get(self, key):
        '''
        Retrieve the values in the database
        '''
        Bin_key = self.format_key(key)
        Kbin = struct.pack('<I4', Bin_key)

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
        print Bbin
        # iterating and populating candidates 
        for _ in range(self.cursor.count()):            
            B = struct.unpack('<I4', Bbin[1])
            # translate to file index and time of occurence
            print B
            f , t = self.read_value(B[0])            
            fileIdx.append(f)
            estTime.append(t)
            # go to next 
            Bbin = self.cursor.next_dup()

        return estTime, fileIdx
    
    def populate(self, sparse_stft, file_index):
        """ populate by creating pairs of peaks """
        pass
    

class XMDCTBDB(FgptHandle):
    '''
    PyMP approx berkeley database handle
    '''
    keyformat = 0
        # closing the db for now ?
#        self.dbObj.close();

    def __init__(self, dbName, load=False,
                 fmax=None, F_N=None,
                 persistent=True, 
                 maxOffset=None,
                 time_res = None):
        '''
        Constructor
        '''
        # Call superclass constructor
        super(XMDCTBDB, self).__init__(dbName, load=load, persistent=persistent)
                    
        # DEFAULT VALUES
        # total number of bits allowed for storing the key in base
        self.total_n_bits = 2 ** 16
        self.fmax = 5500.0
        # max frequency: plain quantization
        self.freq_n_bits = 14
        # for quantization
        self.beta = 0
        self.max_time_offset = 600.0
        # 10 minutes is the biggest allowed time interval
        
        self.time_res = 1.0
        # in seconds
        

        

        # populating optional parameters
        if fmax is not None:
            self.fmax = fmax
        if F_N is not None:
            self.freq_n_bits = F_N
        if maxOffset is not None:
            self.max_time_offset = maxOffset
        # define quantization parameters
        self.beta = ceil((self.fmax) / (2.0 ** self.freq_n_bits - 1.0))        

        if time_res is not None:
            self.time_res = time_res

        self.t_ratio = (self.total_n_bits - 1) / (self.max_time_offset) 


    def __repr__(self):
        return """ 
%s handler (based on Berkeley DB): %s
Key in %d bits,
Resolution: Time: %1.3f (s) %2.2f Hz 
""" % (self.__class__.__name__, self.db_name, self.total_n_bits, self.time_res, self.beta)
    

#    def add(self, Keys , Values, fileIndex):
    def add(self, Pairs, fileIndex):
        '''
        Putting the values in the database
        '''
        
        for key, value in Pairs:

            if value > self.max_time_offset:
#                print value
                print 'Warning: Tried to add a value bigger than maximum'
                continue

            Tbin = floor(value * self.t_ratio)
            # Coding over 16bits, max time offset = 10min
            
            B = int(fileIndex * self.total_n_bits + Tbin)

# K = int(floor(key)*2**(self.freq_n_bits)+floor(float(key)/float(self.beta)));
            K = self.format(key)
            Bbin = struct.pack('<I4', B)
            Kbin = struct.pack('<I4', K)

#            print key , K , value , B
            # search for already existing key
#            BbinEx = self.dbObj.get(Kbin);
#            if BbinEx is None:
            try:
                self.dbObj.put(Kbin, Bbin)
            except db.DBKeyExistError:
                print "Warning existing Key/Value pair " + str(key) + ' ' + str(value)


    def get(self, Key, const=0):        
        '''
        @TODO refactor so that all add and get use the same formalism
        Retrieve the values in the database
        '''
#        K = int(floor(Key)*2**(self.freq_n_bits)+floor(float(Key)/float(self.beta)));
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
            
            f , t = divmod(B[0], self.total_n_bits)
            
            fileIdx.append(f)
            estTime.append((t / self.t_ratio))

            Bbin = self.cursor.next_dup()

        return estTime, fileIdx



    def populate(self, app, fileIndex, offset=0, largebases=False):
        '''
        Populate the database using the given MP approximation
        '''
        if not isinstance(app, Approx):
            raise TypeError(
                'Given argument is not a valid py_pursuit_Approx Object')

#        " else : take all atoms and feed the database"
        F = []
        T = []
        S = []
        for atom in app.atoms:
            if largebases and (log(atom.length, 2) < 13):
                continue
            S.append(log(atom.length, 2))
            F.append(atom.reduced_frequency * app.fs)
            T.append((float(offset + atom.time_position) / float(app.fs)))

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

    def kform(self, atom):
        if self.keyformat is None:
                return atom.reduced_frequency * atom.fs
        elif self.keyformat == 0:
                return [log(atom.length, 2), atom.reduced_frequency * atom.fs]
        

    def retrieve(self, app, offset=0, nbCandidates=10, precision=1.0):
        '''
        Retrieve in base the candidates based on the atoms from the app and return best candidate
        '''
        if not isinstance(app, Approx):
            raise TypeError(
                'Given argument is not a valid py_pursuit_Approx Object')
#        Candidates = {};
#        Offsets = {};
        # refactoring : 2D histogram: col = offset, line = songIndex
        # implemented as a double dictionary : key = songIndex , value =
        # {offset : nbCount}
        histogram = np.zeros((floor(self.max_time_offset / precision), nbCandidates))                       
        
        # results is a list of histogram coordinates, for each element, we need to increment
        # the corresponding valu in histogram by one.
        results = map(self.get , map(self.kform, app.atoms),
                      [a.time_position  / app.fs for a in app.atoms])
        # or maybe we can just sum all the values in the desired dimension?
        
#        print results
        for atomIdx in range(app.atom_number):
            histogram[results[atomIdx]] +=1            

        # voting for best candidate
        return histogram
#        return Candidates , Offsets

    def format(self, key):
        ''' In thi function format the key according to the predefined template
        '''        
        if self.keyformat == 0:
            return int(floor(key[0]) * 2 ** (self.freq_n_bits) + floor(float(key[1]) / float(self.beta)))
        # defaut case
        return int(floor(key) * 2 ** (self.freq_n_bits) + floor(float(key) / float(self.beta)))


    def get_candidate(self, app, nbCandidates=10, **kwargs):
        """ make a guess from the given approximant object """
        
        histograms = self.retrieve(app, nbCandidates= nbCandidates, **kwargs)
        
        maxI = np.argmax(histograms[:])
        OffsetI = maxI / nbCandidates
        estFileI = maxI % nbCandidates
        return estFileI, OffsetI

# class ppBDBSegment(XMDCTBDB):
#    ''' Subclass to encode signature segments by segments
#    '''
#
#    numSeg = 0;
#
#    def add(self, Pairs, segIndex):
#        '''
#        Putting the values in the database
#        '''
#        for key,value in Pairs:
#
#            if value > self.max_time_offset:
##                print value
#                print 'Warning: Tried to add a value bigger than maximum'
#                continue
#
#            Tbin = floor(value/(self.max_time_offset)*(self.total_n_bits-1));  #Coding over 16bits, max time offset = 10min
#
#            B = int(segIndex*self.total_n_bits+Tbin);
#
##            K = int(floor(key)*2**(self.freq_n_bits)+floor(float(key)/float(self.beta)));
#            K = self.format(key)
#            Bbin = struct.pack('<I4',B);
#            Kbin = struct.pack('<I4',K);
#
##            print key , K , value , B
#            try:
#                self.dbObj.put(Kbin, Bbin)
#            except db.DBKeyExistError:
#                print "Warning existing Key/Data pair " + str(key) +' ' + str(value);
#
