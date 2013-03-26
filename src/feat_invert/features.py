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
               'OnsetDet':{'name':'OnsetDet',
                            'featName':'ComplexDomainOnsetDetection',
                            'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
               'magspec':{'name':'magspec',
                            'featName':'MagnitudeSpectrum',
                            'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
               'mfcc_d1':{'name':'mfcc_d1',
                            'featName':'MFCC',
                            'params':'blockSize=%d stepSize=%d > Derivate DOrder=1'%(win_size,step_size)},
                 'energy':{'name':'energy',
                           'featName':'Energy',
                           'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
                 'specflat':{'name':'specflat',
                           'featName':'SpectralFlatness',
                           'params':'FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d'%(win_size,step_size)},
                 'specflux':{'name':'specflux',
                           'featName':'SpectralFlux',
                           'params':'FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d'%(win_size,step_size)},
                 'specstats':{'name':'specstats',
                           'featName':'SpectralShapeStatistics',
                           'params':'FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d'%(win_size,step_size)},
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

#function [Specs, Feats, x] = 
def load_data_one_audio_file(filepath, fs, sigma_noise,
                               wintime = 0.032,
                               steptime = 0.004,
                               max_frame_num_per_file=10000,
                               startpoint=1,
                               features=[]):
    
    N1 = startpoint

    N = max_frame_num_per_file*hoptime*fs;
    
    print 'Loading from file ', filepath
#    if n_sam>N
#        disp(['Cropping at ' num2str(N)]);
#        [x, Fs] = wavread(filepath, [N1, N1+N]);
#    else
#        [x, Fs] = wavread(filepath);
#    end
    
    # resample ?
#    if sr ~= Fs
#        x = resample(x, sr, Fs);
#    end
#    %add some noise ?
#    if sigma_noise > 0
#        x = x + sigma_noise*randn(size(x));
#    end

    yaafe_dict = get_yaafe_dict(int(wintime*fs),int(steptime*fs))

    featureList = {}
    # we already know we want the magnitude spectrum
    featureList['magspec'] = yaafe_dict['magspec']
    for feature in features:
        featureList[feature] = yaafe_dict[feature]
            
    feats = get_yaafe_features(featureList, filepath)
    
    Feats = np.array()
    
    for feature in featureList:
        if feats.has_key(feature):
           print 'Loading ' , feature        
           Feats = np.concatenate((Feats, feats[feature]))
        else:
           print 'Warning, feature ', feature, ' not found' 
    
    return feats['magspec'], feats
    