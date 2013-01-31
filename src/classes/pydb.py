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
    ''' abstract class for Fgpt methods '''
    
    # parameters
    db_name = None
    dbObj = None
    
    # total number of bits allowed for storing the key in base
    total_n_bits = 2 ** 16
    fmax = 5500.0
    # max frequency: plain quantization
    freq_n_bits = 14
    # for quantization
    beta = 0
    max_time_offset = 600.0
    # 10 minutes is the biggest allowed time interval
    persistent = True
    
    time_res = 1.0
    # in seconds
    cursor = None
    
    def __repr__(self):
        return """ 
%s handler (based on Berkeley DB): %s
Key in %d bits,
Resolution: Time: %1.3f (s) %2.2f Hz 
""" % (self.__class__.__name__, self.db_name, self.total_n_bits, self.time_res, self.beta)
    

    def __del__(self):
        self.dbObj.close()
        if not self.persistent:
            if os.path.exists(self.db_name):
                print 'Destroying Db ', self.db_name
                os.remove(self.db_name)

                del self.dbObj, self.db_name


    def __init__(self, dbName, load=False,
                 fmax=None, F_N=None,
                 persistent=None, 
                 maxOffset=None,
                 time_res = None):
        '''
        Constructor
        '''

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

        # populating optional parameters
        if fmax is not None:
            self.fmax = fmax
        if F_N is not None:
            self.freq_n_bits = F_N
        if maxOffset is not None:
            self.max_time_offset = maxOffset
        # define quantization parameters
        self.beta = ceil((self.fmax) / (2.0 ** self.freq_n_bits - 1.0))
        if persistent is not None:
            self.persistent = persistent

        if time_res is not None:
            self.time_res = time_res

        self.t_ratio = (self.total_n_bits - 1) / (self.max_time_offset) 


    def get_stats(self):
        ''' retrieve the number of keys in the table
        '''
        return self.dbObj.stat()

class PeakPairsBDB(FgptHandle):
    ''' handling the fingerprints based on a pairing of STFT peaks '''
    
    

class ppBDB(FgptHandle):
    '''
    PyPursuit berkeley database handle
    '''

    keyformat = 0


        # closing the db for now ?
#        self.dbObj.close();



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




# class ppBDBSegment(ppBDB):
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
