'''
feat_invert.features  -  Created on Feb 21, 2013
@author: M. Moussallam
'''
from yaafelib import *
import numpy as np
from PyMP import signals
from .transforms import get_stft

def get_yaafe_dict(win_size, step_size):
    YaafeDict = {'mfcc': {'name': 'mfcc',
                          'featName': 'MFCC',
                          'params': 'blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'zcr': {'name': 'zcr',
                         'featName': 'ZCR',
                         'params': 'blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'loudness': {'name': 'Loudness',
                              'featName': 'Loudness',
                              'params': 'blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'lpc-4': {'name': 'lpc',
                           'featName': 'LPC',
                           'params': 'LPCNbCoeffs=4 blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'OnsetDet': {'name': 'OnsetDet',
                              'featName': 'ComplexDomainOnsetDetection',
                              'params': 'blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'magspec': {'name': 'magspec',
                             'featName': 'MagnitudeSpectrum',
                             'params': 'blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'mfcc_d1': {'name': 'mfcc_d1',
                             'featName': 'MFCC',
                             'params': 'blockSize=%d stepSize=%d > Derivate DOrder=1' % (win_size, step_size)},
                 'energy': {'name': 'energy',
                            'featName': 'Energy',
                            'params': 'blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'specflat': {'name': 'specflat',
                              'featName': 'SpectralFlatness',
                              'params': 'FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'specflux': {'name': 'specflux',
                              'featName': 'SpectralFlux',
                              'params': 'FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d' % (win_size, step_size)},
                 'specstats': {'name': 'specstats',
                               'featName': 'SpectralShapeStatistics',
                               'params': 'FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d' % (win_size, step_size)},
                 }
    return YaafeDict


def get_yaafe_features(featuresList, audiofile, target_fs=32000):
    """ extracting the features
        each element in featuresList must be a dictionary with:
        name         (ex: mfcc)
        featName     (ex: MFCC)
        params       (ex: blockSize=512 stepSize=256)
    """

    fp = FeaturePlan(sample_rate=target_fs, resample=True)
    for feat in featuresList:
        feat_str = '%s: %s %s' % (feat['name'], feat['featName'],
                                  feat['params'])
        fp.addFeature(feat_str)

    df = fp.getDataFlow()
    engine = Engine()
    engine.load(df)
    afp = AudioFileProcessor()
    afp.processFile(engine, audiofile)

    # extracting the features
    feats = engine.readAllOutputs()
    return feats

# function [Specs, Feats, x] =


def load_data_one_audio_file(filepath, fs, sigma_noise=0,
                             wintime=0.032,
                             steptime=0.008,
                             max_frame_num_per_file=3000,
                             startpoint = 0,
                             features=['zcr', ]):

    N = max_frame_num_per_file * steptime * fs

    if startpoint >0:
        # guess which segment need to be selected
        segIdx = int(float(startpoint * fs)/N)
        print "Starting at segment", segIdx
    else:
        segIdx = 0

#    print 'Loading from file ', filepath
    # pre loading the file using a longSignal object
    preload = signals.LongSignal(filepath, frame_size=N, mono=True)

    if preload.n_seg < 1:
        del preload
        sigx = signals.Signal(filepath, mono=True)
    else:
        print 'Cropping at %d' % N
        sigx = preload.get_sub_signal(segIdx, 1)

    # resample ?
    if not fs == sigx.fs:        
        sigx.downsample(fs)

#    %add some noise ?
    if sigma_noise > 0.0:
        sigx.data += sigma_noise * np.random.randn(sigx.data.shape[0],)

    x = sigx.data
    fs = sigx.fs

    yaafe_dict = get_yaafe_dict(int(wintime * fs), int(steptime * fs))

    featureList = []
    # we already know we want the magnitude spectrum
    featureList.append(yaafe_dict['magspec'])
    for feature in features:
        featureList.append(yaafe_dict[feature])

    feats = get_yaafe_features(featureList, filepath, target_fs=fs)

    Feats = np.array([])
    featseq = []

    for feature in features:
        if feature in feats:
            featseq.append(feats[feature])
        else:
            print 'Warning, feature ', feature, ' not found'

    Feats = np.hstack(featseq)
    return feats['magspec'], Feats, x


def getoptions(paramsdict, optioname, default_value):
    if optioname in paramsdict:
        return paramsdict[optioname]
    else:
        return default_value


def get_filepaths(audio_path, random_seed=None, forbid_list=[]):
    """function [file_paths] = get_filepaths(audio_path, random_seed)
    % retrieves all the wav file names and relative path given the directory
    % if random_seed is specified: it applies a random suffling of the files
    % paths"""

    import os
    import os.path as op
    file_paths = []
    # root
    dir_list = os.listdir(audio_path)
    
    # recursive search
    for dir_ind in range(len(dir_list)):

        if op.isdir(op.join(audio_path, dir_list[dir_ind])):

            sub_files = get_filepaths(op.join(audio_path,
                                              dir_list[dir_ind]),
                                      forbid_list=forbid_list)
            file_paths.extend(sub_files)
        else:
            if '.wav' in dir_list[dir_ind]:
                if not dir_list[dir_ind] in forbid_list:                                
                    file_paths.append(op.join(audio_path, dir_list[dir_ind]))

    if random_seed is not None:
        # use the random_seed to initialize random state
        np.random.seed(random_seed)
        file_paths = np.random.permutation(file_paths)

    return file_paths


def load_yaafedata(params,                   
                   n_learn_frames=2000,
                   use_custom_stft=True):

    """%LOAD_DATA load feature and magnitude spectrum matrices from the given
    %location with specified parameters
    %   Detailed explanation goes here"""

    audio_file_path = getoptions(params, 'location', '/sons/voxforge/data/Learn/')
    # if no number specified, use n_learn_frames
    n_frames = getoptions(params, 'n_frames', n_learn_frames)
    sr = getoptions(params, 'sr', 16000)
    sigma_noise = getoptions(params, 'sigma', 0.0)
    random_seed = getoptions(params, 'shuffle', 1001)
    features = getoptions(params, 'features', [])
    wintime = getoptions(params, 'wintime', 0.032)
    steptime = getoptions(params, 'steptime', 0.008)
    startpoint = getoptions(params, 'startpoint', 0)
    forbid_list = getoptions(params, 'forbidden_names', [])
    
#    wintime = float(win_size)/float(sr)
#    steptime = float(step_size)/float(sr)
    
    win_size = int(wintime*sr)
    step_size = int(steptime*sr)
    print wintime, steptime, win_size, step_size
    # apply sub_routine to all the files until a condition is met
    n_frames_reached = 0

    all_file_paths = get_filepaths(audio_file_path,
                                   random_seed,
                                   forbid_list = forbid_list)
    file_index = 0

    specseq = []
    featseq = []
    dataseq = []
    n_files_used = 0

    while (n_frames_reached < n_frames):
        file_index = file_index + 1
        filepath = all_file_paths[file_index]
        n_files_used = n_files_used + 1

        [loc_magSTFT, loc_Feats, locDatas] = load_data_one_audio_file(
                                                filepath, sr,
                                                wintime=wintime,
                                                steptime=steptime,
                                                sigma_noise=sigma_noise,
                                                startpoint = startpoint,
                                                features=features)
#        if get_data:
#            [loc_magSTFT, loc_Feats, locDatas] = load_data_one_file_melspec(filepath, sr, sigma_noise, params);
#            Data = [Data , locDatas'];
#        else
#            [loc_magSTFT, loc_Feats, ~] = load_data_one_file_melspec(filepath, sr, sigma_noise, params);
#        end
        if not use_custom_stft:
            specseq.append(loc_magSTFT)
        else:
            specseq.append(np.abs(get_stft(locDatas,
                                          win_size,
                                          step_size,
                                          sigma = sigma_noise)).T)
            
        featseq.append(loc_Feats)
        dataseq.append(locDatas)
        
        n_frames_reached += min(loc_magSTFT.shape[0], loc_Feats.shape[0])
        
    
    Spectrums = np.vstack(specseq)
    Features = np.vstack(featseq)
    Data = np.hstack(dataseq)

    n_frames_reached = min(n_frames_reached, n_frames)
    Spectrums = Spectrums[0:n_frames_reached,:]
    Features = Features[0:n_frames_reached,:]
    used_files = all_file_paths[0:n_files_used]

    return Features, Spectrums, n_frames_reached, Data, used_files
