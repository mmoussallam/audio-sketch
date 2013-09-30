'''
classes.fingerprints.cortico  -  Created on Sep 30, 2013
@author: M. Moussallam
'''

from src.classes.fingerprints.base import *
from src.classes.fingerprints.cochleo import *

class CorticoIndepSubPeaksBDB(FgptHandle):
    """ 4-D peaks of corticograms used directly as fingerprints 
    
    To use with a CorticoIndepSubPeaksSketch sketchifier:
    - for each sub representaion (scale/rate plot) we handle an independant database
    So this is rather a FgptHandle Collection

    """
    
    def __init__(self, dbNameroot, handletype = CochleoPeaksBDB,
                 load=False, persistent=None,dbenv=None,cachesize=512,
                 **kwargs):
        """ DO NOT Call superclass constructor since we need to instantiate a 
        collection of handles
        
        parameters
        ----------
            dbNameroot : root name of the bdb
            handletype : the type of BDB
        """
        self.db_names = []
        self.dbObj = []
        self.params = {'delta_t_max':3.0,
                       'fmax': 8000.0,                       
                        'n_sv': 5,
                        'n_rv': 6,
                        'n_jobs':4,
                        'max_pairs':None,                        
                        'wall':True}
        
        for key in kwargs:
            self.params[key] = kwargs[key]
        
        
        
        if dbNameroot is None:
            #Ok so we want a pure RAM-based DB, let's do it
            self.pureRAM = True
            load = False
            persistent = False

        self.db_root = dbNameroot
        self.db_name = self.db_root
        
        if not os.path.exists(self.db_root):
            os.mkdir(self.db_root)
        
        if dbenv is None:
            dbenv = db.DBEnv()
            # default cache size is 200Mbytes
            dbenv.set_cachesize(10,cachesize*1024*1024,0)            
            env_flags = db.DB_CREATE | db.DB_PRIVATE | db.DB_INIT_MPOOL#| db.DB_INIT_CDB | db.DB_THREAD
            dbenv.log_set_config(db.DB_LOG_IN_MEMORY, 1)
            dbenv.open(None, env_flags)
        # We use 2D arrays for consistency, list could have been done also
        
        
        
        for scaleIdx in range(self.params['n_sv']):
            self.db_names.append([])
            self.dbObj.append([])
            for rateIdx in range(self.params['n_rv']):        
                 # For each scale/rate pair create a database
                self.db_names[scaleIdx].append("%s_%d_%d.db"%(self.db_root, scaleIdx, rateIdx))                
                self.dbObj[scaleIdx].append( handletype(self.db_names[scaleIdx][rateIdx],
                                                           load=load, persistent=persistent,
                                                           dbenv=dbenv,
                                                           **kwargs))                
        
    
    def __del__(self):    
        for scaleIdx in range(self.params['n_sv']):
            for rateIdx in range(self.params['n_rv']):    
                del self.dbObj[scaleIdx][rateIdx]


    def __repr__(self):
        return """ %s handler (based on Berkeley DB): %s""" % (self.__class__.__name__,
                                                               self.db_name)    

#    def sub_populate(self, coords, fgpt, params, fileIndex, offset, debug):
#        (scaleIdx,rateIdx) = coords
#        print coords
#        self.dbObj[scaleIdx,rateIdx].populate(fgpt[scaleIdx,self.params['n_rv']+rateIdx,:,:],
#                                                      params, fileIndex,
#                                                      offset=offset, debug=debug)

    def populate(self, fgpt, params, fileIndex, offset=0,debug=False,max_pairs=None):
        """ populate each sub db independantly """        
        
        assert fgpt.shape[:2] == (self.params['n_sv'],2*self.params['n_rv'])
#        # assign for each incoming fingerprint to the corresponding db
#        Parallel(n_jobs=self.params['n_jobs'])(delayed(_sub_populate)([],scaleIdx, fgpt, params, fileIndex, offset, debug)
#                                                    for scaleIdx in range(self.params['n_sv']))
#        

        if max_pairs is None:
            max_pairs = self.params['max_pairs']
        for scaleIdx in range(self.params['n_sv']):
            for rateIdx in range(self.params['n_rv']):
                self.dbObj[scaleIdx][rateIdx].populate(fgpt[scaleIdx,self.params['n_rv']+rateIdx,:,:],
                                                      params, fileIndex,
                                                      offset=offset, debug=debug,max_pairs=max_pairs)
    
    def retrieve(self, fgpt, params,  offset=0, nbCandidates=10, precision=1.0):
        """ retrieve  """        
        assert fgpt.shape[:2] == (self.params['n_sv'],2*self.params['n_rv'])
        # assign for each incoming fingerprint to the corresponding db
        
        histograms = []
        
        for scaleIdx in range(self.params['n_sv']):
            for rateIdx in range(self.params['n_rv']):
                histograms.append(self.dbObj[scaleIdx][rateIdx].retrieve(fgpt[scaleIdx,
                                                                             self.params['n_rv']+rateIdx,:,:],
                                                                        params,
                                                                        offset=offset,
                                                                        nbCandidates=nbCandidates,
                                                                        precision=precision))
        return histograms

    def get_candidate(self, fgpt, params, nbCandidates=10, smooth=1):
        estFiles = np.empty((self.params['n_sv'],self.params['n_rv']))
        Offsets = np.empty((self.params['n_sv'],self.params['n_rv']))
        
        for scaleIdx in range(self.params['n_sv']):
            for rateIdx in range(self.params['n_rv']):
                
                x,y = self.dbObj[scaleIdx][rateIdx].get_candidate(fgpt[scaleIdx,
                                                                        self.params['n_rv']+rateIdx,:,:],
                                                                  params,
                                                                  nbCandidates=nbCandidates,
                                                                  smooth=smooth)
                estFiles[scaleIdx,rateIdx]=x
                Offsets[scaleIdx,rateIdx]=y
                
        return estFiles, Offsets

    def get_db_sizes(self):
        """ return the size of the dbs """
        import os
        sizes = np.empty((self.params['n_sv'],self.params['n_rv']))
        for scaleIdx in range(self.params['n_sv']):
            for rateIdx in range(self.params['n_rv']):
                path = self.dbObj[scaleIdx][rateIdx].db_name
                sizes[scaleIdx,rateIdx] = os.stat(path).st_size

        return sizes
    
    def draw_fgpt(self, fgpt, params, ax=None):
        """ Draw the fingerprint """
#        self._build_pairs(fgpt, params, 0, display=True, ax=ax)
        
        import matplotlib.pyplot as plt
        fig = plt.figure() 
        for n in range(self.params['n_sv']):
            for m in range(self.params['n_rv']):
                ax = plt.subplot(self.params['n_sv'], self.params['n_rv'],
                                 (n* self.params['n_rv']) + m+1)
                self.dbObj[n][m]._build_pairs(fgpt[n,self.params['n_rv']+m,:,:],
                                              params, display=True, ax=ax)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(self.params['n_sv'], self.params['n_rv'], m+1)
                plt.title(str(params['rv'][m]))
            plt.subplot(self.params['n_sv'], self.params['n_rv'], (n* self.params['n_rv']) + 1)
            plt.ylabel(str(params['sv'][n]))
        plt.subplots_adjust(left=0.06, bottom=0.05, top=0.92,right=0.96)
        