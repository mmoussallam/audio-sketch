'''
classes.fingerprints.base  -  Created on Sep 30, 2013
@author: M. Moussallam
'''
import bsddb.db as db
import os
import numpy as np


class FgptHandle(object):
    ''' Super class for Fingerprints 
    
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
        when object is deleted
    
    '''    
    
    
    def __init__(self, dbName, 
                 load=False,                 
                 persistent=True, dbenv=None,
                 rd_only=False, cachesize=(0,512)):
        """ Common Constructor """
        
        if dbName is None:
            #Ok so we want a pure RAM-based DB, let's do it
            self.pureRAM = True
            load = False
            persistent = False
        else:
            if os.path.exists(dbName) and not load:
                os.remove(dbName)
        self.db_name = dbName
        if dbenv is not None:
            self.dbObj = db.DB(dbenv)
        else:
            env = db.DBEnv()
            # default cache size is 200Mbytes
            env.set_cachesize(cachesize[0],cachesize[1]*1024*1024,0)            
            env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
            env.log_set_config(db.DB_LOG_IN_MEMORY, 1)
            env.open(None, env_flags)
            self.dbObj = db.DB(env)
        self.opened = False
        # allow multiple key entries
        # TODO :  mettre en DB_DUPSORT
        self.dbObj.set_flags(db.DB_DUP | db.DB_DUPSORT)
#        self.dbObj.set_flags(db.DB_DUP)

        if not load:
            try:
                self.dbObj.open(self.db_name, dbtype=db.DB_HASH, flags=db.DB_CREATE)
                self.opened = True
            except:                
                raise IOError('Failed to create %s ' % self.db_name)
            print "Created DB:", dbName
        else:
            if self.db_name is None:
                raise ValueError('No Database name provided for loading')
                
            if not os.path.exists(self.db_name):
                self.dbObj.open(self.db_name, dbtype=db.DB_HASH, flags=db.DB_CREATE)
            else:
                if rd_only:
                    self.dbObj.open(dbName, dbtype=db.DB_HASH, flags=db.DB_RDONLY)
                else:
                    self.dbObj.open(dbName, dbtype=db.DB_HASH)
            self.opened = True
            print "Loaded DB:", dbName
    
        # keep in mind if the db is to be kept or destroy
        self.persistent = persistent
        # cursor object : might get instantiated later
        self.cursor = None
    
    def __del__(self):        
        self.dbObj.sync()
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
    
    def get_size(self):
        ''' retrieve the size on disk '''
        return os.stat(self.db_name).st_size        
    
    def add(self, Pairs, fileIndex):
        ''' add all key/value pairs to base '''
        raise NotImplementedError("Not Implemented")
    
    def get(self, key):
        ''' retrieve all values associated with key '''
        raise NotImplementedError("Not Implemented")

    def populate(self, fgpt, params, fileIndex, offset=0):
        ''' Populate the database using the given fingerprint and parameters '''
        raise NotImplementedError("Not Implemented")

    def retrieve(self, fgpt, params, offset=0, nbCandidates=10, precision=1.0):
        raise NotImplementedError("Not Implemented")
    
    def get_candidate(self, fgpt, params, nbCandidates=10, smooth=1):
        """ make a guess from the given approximant object 
        
        All subclasses should use the same architecture so this method should not
        need be overridden : just a wrapper for a single guess
        """        
        histograms = self.retrieve(fgpt, params, nbCandidates= nbCandidates)
        
        if smooth>1:
            from scipy.ndimage.filters import median_filter
            histograms = median_filter(histograms, (smooth, 1 ))
        
        maxI = np.argmax(histograms[:])
        OffsetI = maxI / nbCandidates
        estFileI = maxI % nbCandidates
        return estFileI, OffsetI
