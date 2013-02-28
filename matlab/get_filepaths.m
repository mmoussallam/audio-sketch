function [file_paths] = get_filepaths(audio_path, random_seed)
% retrieves all the wav file names and relative path given the directory
% if random_seed is specified: it applies a random suffling of the files
% paths

shuffle = nargin>1;

file_paths = {};
% root 
dir_list = dir(audio_path);

% recursive searching
for dir_ind =1:length(dir_list)
    
    % remove useless directories
    if strcmp(dir_list(dir_ind).name,'.') || strcmp(dir_list(dir_ind).name,'..')
       continue; 
    end
    
    if dir_list(dir_ind).isdir
        
        sub_files = get_filepaths([audio_path '/' dir_list(dir_ind).name]);
        for s=1:length(sub_files)
            file_paths{length(file_paths) +1} = sub_files{s};
        end
%         file_paths = [file_paths, get_filepaths([audio_path '/' dir_list(dir_ind).name], max_path_length)];
    else        
        
        if strfind(dir_list(dir_ind).name,'.wav');
            file_paths{length(file_paths) +1} = [audio_path '/' dir_list(dir_ind).name];
        end
        
        
    end
    
end

if shuffle
   % use the random_seed to initialize random state 
   rng(random_seed); 
   P = randperm(length(file_paths));
   file_paths = file_paths(P);
end

end