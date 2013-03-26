function [yaafe_dataflow] = get_yaafeFeat(names, params)

feat_plan_file = fopen('featureplan','w+');
% writing the feature plan file
for nameIdx=1:length(names)
    switch names{nameIdx}   
        case 'mfcc'
            fprintf(feat_plan_file,'mfcc: MFCC blockSize=%d stepSize=%d\n',params.win_size, params.step_size); 

        case 'loudness'
            fprintf(feat_plan_file,'loudness: Loudness blockSize=%d stepSize=%d\n',params.win_size, params.step_size); 
        
        case 'lpc'
            fprintf(feat_plan_file,'lpc: LPC LPCNbCoeffs=4 blockSize=%d stepSize=%d\n',params.win_size, params.step_size); 
        
        case 'OnsetDet'
            fprintf(feat_plan_file,'OnsetDet: ComplexDomainOnsetDetection blockSize=%d stepSize=%d\n',params.win_size, params.step_size); 
        
        case 'mfcc_d1'
            fprintf(feat_plan_file,'mfcc_d1: MFCC blockSize=%d stepSize=%d > Derivate DOrder=1\n',params.win_size, params.step_size); 
        
        case 'zcr'
            fprintf(feat_plan_file,'zcr: ZCR blockSize=%d stepSize=%d\n',params.win_size, params.step_size); 
        
        case 'melspec'
            fprintf(feat_plan_file, 'melspec: MelSpectrum FFTWindow=Hanning  MelMaxFreq=6854.0  MelMinFreq=130.0  MelNbFilters=40  blockSize=%d  stepSize=%d\n',params.win_size, params.step_size);
        
        case 'magspec'
            fprintf(feat_plan_file,'magspec: MagnitudeSpectrum FFTLength=%d FFTWindow=Hanning blockSize=%d stepSize=%d\n',params.win_size, params.win_size, params.step_size);
        
        case 'energy'
            fprintf(feat_plan_file,'energy: Energy blockSize=%d  stepSize=%d\n',params.win_size, params.step_size);
                    
        case 'specflat'
            fprintf(feat_plan_file,'specflat:SpectralFlatness FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d\n',params.win_size, params.step_size);
        
        case 'specflux'
            fprintf(feat_plan_file,'specflux:SpectralFlux FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d\n',params.win_size, params.step_size);
        
        case 'specrolloff'
            fprintf(feat_plan_file,'SpectralRolloff:SpectralRolloff FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d\n',params.win_size, params.step_size);
                   
        case 'specstats'
            fprintf(feat_plan_file,'specstats:SpectralShapeStatistics FFTLength=0 FFTWindow=Hanning blockSize=%d stepSize=%d\n',params.win_size, params.step_size);
    end
end

% getting the dataflow
[status, result] = system(['yaafe.py -r ' num2str(params.fs) ' -c featureplan --dump-dataflow=yaflow']);


% computing
yaafe_dataflow = Yaafe();
yaafe_dataflow.load('yaflow')
% 
% YaafeDict.mfcc.name='mfcc',
%                      'featName':'MFCC',
%                      'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
%                'zcr':{'name':'zcr',
%                       'featName':'ZCR',
%                       'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
%                'loudness':{'name':'Loudness',
%                             'featName':'Loudness',
%                             'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
%                'lpc-4':{'name':'lpc',
%                        'featName':'LPC',
%                        'params':'LPCNbCoeffs=4 blockSize=%d stepSize=%d'%(win_size,step_size)},
%                'OnsetDet':{'name':'CDOD',
%                             'featName':'ComplexDomainOnsetDetection',
%                             'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
%                'magspec':{'name':'magspec',
%                             'featName':'MagnitudeSpectrum',
%                             'params':'blockSize=%d stepSize=%d'%(win_size,step_size)},
%                'mfcc_d1':{'name':'mfcc_d1',
%                             'featName':'MFCC',
%                             'params':'blockSize=%d stepSize=%d > Derivate DOrder=1'%(win_size,step_size)}                            
%                }


end