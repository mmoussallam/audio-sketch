function [ Features, Spectrums, n_frames_reached, Data] = load_data( params )
%LOAD_DATA load feature and magnitude spectrum matrices from the given
%location with specified parameters
%   Detailed explanation goes here

% load default parameters
parameters

if isfield(params , 'location')
   audio_file_path =  params.location;
end

% if no number specified, use n_learn_frames
n_frames = getoptions(params, 'n_frames', n_learn_frames);
sigma_noise = getoptions(params, 'sigma', 0.0);
random_seed = getoptions(params, 'shuffle', 1001);
get_data =  getoptions(params, 'get_data', 0);
%TODO other parameters

% apply sub_routine to all the files until a condition is met
n_frames_reached = 0;

all_file_paths = get_filepaths(audio_file_path, random_seed);
file_index = 0;

Spectrums = [];
Features = [];
Data = [];
while n_frames_reached < n_frames
    file_index = file_index+1;
    filepath = all_file_paths{file_index};
    
    % if 
    if get_data
        [loc_magSTFT, loc_Feats, locDatas] = load_data_one_file(filepath, sr, sigma_noise);
        Data = [Data , locDatas'];
    else
        [loc_magSTFT, loc_Feats, ~] = load_data_one_file(filepath, sr, sigma_noise);
    end
    Spectrums = [Spectrums , loc_magSTFT];
    Features = [Features , loc_Feats];
    
    n_frames_reached = min(size(Spectrums,2),size(Features,2)) ;
    
end
n_frames_reached = min(n_frames_reached, n_frames);
Spectrums = Spectrums(:,1:n_frames_reached);
Features = Features(:,1:n_frames_reached);


end

function [magSTFT, Feats, x] = load_data_one_file(filepath, sr, sigma_noise)

parameters;


[siz, fs] = wavread(filepath , 'size');
n_sam = siz(1);

N = max_frame_num_per_file*hoptime*fs;

disp(['Loading from file ' filepath ' size : ' num2str(n_sam)]);
if n_sam>N
    disp(['Cropping at ' num2str(N)]);
    [x, Fs] = wavread(filepath, N);
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

% perform STFT
sig = Signal(x, sr);
sig.windowLength = wintime*sr;
sig.fs = sr;
sig.overlapRatio = 1-hoptime/wintime;
sig.nfft = 2^(ceil(log(wintime*sr)/log(2)));
sig.STFT()

magSTFT = abs(sig.S);

[Feats,~,~] = melfcc(x,sr,'maxfreq',maxfreq, ...
                                'wintime',wintime,'hoptime',hoptime, ...
                                'nbands',nbands,'numcep',numcep);

                            
                            
end





