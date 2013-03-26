function [ Features, Spectrums, n_frames_reached, Data, used_files] = load_yaafedata( params )
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
n_files_used = 0;
while n_frames_reached < n_frames
    file_index = file_index+1;
    filepath = all_file_paths{file_index};
    n_files_used = n_files_used+1;
    % if 
    if get_data
        [loc_magSTFT, loc_Feats, locDatas] = load_data_one_file_melspec(filepath, sr, sigma_noise, params);
        Data = [Data , locDatas'];
    else
        [loc_magSTFT, loc_Feats, ~] = load_data_one_file_melspec(filepath, sr, sigma_noise, params);
    end
    Spectrums = [Spectrums , loc_magSTFT];
    Features = [Features , loc_Feats];
    
    n_frames_reached = min(size(Spectrums,2),size(Features,2)) ;
    
end
n_frames_reached = min(n_frames_reached, n_frames);
Spectrums = Spectrums(:,1:n_frames_reached);
Features = Features(:,1:n_frames_reached);
used_files = all_file_paths(1:n_files_used);

end







