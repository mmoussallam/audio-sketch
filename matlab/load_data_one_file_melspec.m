function [Specs, Feats, x] = load_data_one_file_melspec(filepath, sr, sigma_noise, params)

if isfield(params,'hoptime')
    hoptime = params.hoptime;
else
    parameters;
end

if isfield(params,'sr')
    sr = params.sr;
else
    parameters;
end

if isfield(params,'wintime')
    wintime = params.wintime;
else
    parameters;
end

if isfield(params,'max_frame_num_per_file')
    max_frame_num_per_file = params.max_frame_num_per_file;
else
    parameters;
end

if isfield(params,'startpoint')
    N1 = params.startpoint;
else
    N1 = 1;
    
end

[siz, fs] = wavread(filepath , 'size');
n_sam = siz(1);

N = max_frame_num_per_file*hoptime*fs;

disp(['Loading from file ' filepath ' size : ' num2str(n_sam)]);
if n_sam>N
    disp(['Cropping at ' num2str(N)]);
    [x, Fs] = wavread(filepath, [N1, N1+N]);
else
    [x, Fs] = wavread(filepath);
end

% resample ?
if sr ~= Fs
    x = resample(x, sr, Fs);
end
%add some noise ?
if sigma_noise > 0
    x = x + sigma_noise*randn(size(x));
end

% % perform STFT
% sig = Signal(x, sr);
% sig.windowLength = wintime*sr;
% sig.fs = sr;
% sig.overlapRatio = 1-hoptime/wintime;
% sig.nfft = 2^(ceil(log(wintime*sr)/log(2)));
% sig.STFT()
% 
% magSTFT = abs(sig.S);

% [ellis_mfcc,~,~] = melfcc(x,sr,'maxfreq',maxfreq, ...
%                                 'wintime',wintime,'hoptime',hoptime, ...
%                                 'nbands',nbands,'numcep',numcep);

% loading the features!
yaafeopts.win_size = wintime*sr;
yaafeopts.step_size = hoptime*sr;
yaafeopts.fs = sr;
yaafe_df = get_yaafeFeat(params.features, yaafeopts);

FeatStruct = yaafe_df.process(x');            
Feats = [];

for featnameidx=1:length(params.features)
    featname = params.features{featnameidx};
    
    if strcmp(featname,'magspec') || strcmp(featname,'melspec')
        continue;
    end
    
    if isfield(FeatStruct, featname)
        disp(['Loading ' featname]);
        curFeat =  getfield(FeatStruct, featname);
    
        Feats = [Feats ; curFeat.data];
    else
        disp(['Warning, feature ' featname ' not found']);
    end
end

if isfield(FeatStruct, 'melspec')

    Specs = FeatStruct.melspec.data;
else
    Specs = FeatStruct.magspec.data;
end
end