'''
feat_invert.features  -  Created on Feb 21, 2013
@author: M. Moussallam
'''
from yaafelib import *

def get_yaafe_dict(win_size,step_size):
    YaafeDict = {'mfcc':{'name':'mfcc',
                     'featName':'MFCC',
                     'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
               'zcr':{'name':'zcr',
                      'featName':'ZCR',
                      'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
               'loudness':{'name':'Loudness',
                            'featName':'Loudness',
                            'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
               'lpc-4':{'name':'lpc',
                       'featName':'LPC',
                       'params':'LPCNbCoeffs=4 blockSize=%d stepSize=%d'%(win_size,step_size)},
               'OnsetDet':{'name':'CDOD',
                            'featName':'ComplexDomainOnsetDetection',
                            'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
               'magspec':{'name':'magspec',
                            'featName':'MagnitudeSpectrum',
                            'params':'blockSize=%d stepSize=%d'%(win_size,step_size)}
               }
    return YaafeDict

def get_yaafe_features(featuresList,audiofile):
    """ extracting the features 
        each element in featuresList must be a dictionary with:
        name         (ex: mfcc)
        featName     (ex: MFCC)
        params       (ex: blockSize=512 stepSize=256)
    """
    
    fp = FeaturePlan(sample_rate=32000)
    
    for feat in featuresList:
        feat_str = '%s: %s %s'%(feat['name'], feat['featName'], feat['params'])
        fp.addFeature(feat_str)
    
        
    df = fp.getDataFlow()
    
    engine = Engine()
    engine.load(df)        
    
    afp = AudioFileProcessor()
    afp.processFile(engine,audiofile)
    
    # extracting the features
    feats = engine.readAllOutputs()
    return feats
