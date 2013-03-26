close all;
clear all;
% parameters
audio_file_path = '/sons/sqam/';
learn_file = '/sons/sqam/voicemale.wav';
test_file = '/sons/sqam/voicefemale.wav';
% loading the learn base
params.n_frames = 2000;
params.sigma = 0.00001;
params.shuffle = 1001;
params.get_data = 1;
params.sr = 32000;
params.wintime = 0.016;
params.hoptime = 0.004;
params.max_frame_num_per_file = 5000;
params.features = {'mfcc','zcr','loudness','lpc','OnsetDet','mfcc_d1'};
% params.features = {'mfcc'};
% [learn_feats, learn_magspecs, n_f_learn, ref_learn_data, learn_files] = load_data(params);
[learn_magspecs, learn_feats, ref_learn_data] = load_data_one_file(learn_file, 32000, 0.001, params);

n_frames_learn = min(size(learn_magspecs,2),size(learn_feats,2));
learn_magspecs = learn_magspecs(:,1:n_frames_learn);
learn_feats = learn_feats(:,1:n_frames_learn);

% loading the test base
params.n_frames = 1800;
params.sigma = 0.00001;
params.shuffle = 3001;
params.get_data = 1;
% params.location = '/sons/rwc/rwc-g-m02/';
% [test_feats, test_magspecs, n_f_test, ref_t_data, test_files] = load_data(params);
[test_magspecs, test_feats, ref_t_data] = load_data_one_file(test_file, 32000, 0.001, params);

n_frames_test = min(size(test_magspecs,2),size(test_feats,2));
test_magspecs = test_magspecs(:,1:n_frames_test);
test_feats = test_feats(:,1:n_frames_test);

% mlearn =  cell2mat(learn_files');
% tlearn =  cell2mat(test_files');
% 
% isinbase = ~isempty(intersect(mlearn,tlearn,'rows'))

%%  Trying a nadaraya-watson

methods_to_try = [1,6];

res_struct = {};
nb_medians = 10;
display = 1;
for methodix=1:length(methods_to_try)
    tic
    disp(['Working on method ' num2str(methodix)]);
    method = methods_to_try(methodix);
    % inverse l2 distance - mapping
    res_struct{methodix}.method = method;
    
    [res_struct{methodix}.m, res_struct{methodix}.err] = knn_freq(learn_feats, learn_magspecs, test_feats, test_magspecs, @covariance,method,display,nb_medians);
    disp(['mean l2 error of  ' num2str(mean(res_struct{methodix}.err))]);
    
    % sliding median filtering ?
    res_struct{methodix}.m_filt = medfilt1(res_struct{methodix}.m',3)';
    
    % reconstruction
    init_vec = randn(length(ref_t_data),1);
    [res_struct{methodix}.x_rec] = gl_reconstruct(res_struct{methodix}.m_filt, init_vec, params.sr, 5, params.wintime*params.sr, params.hoptime*params.sr);

%     % pemo-Q evaluation
%     [res_struct{methodix}.pemoscore,~]= pemo(ref_t_data',res_struct{methodix}.x_rec,params.sr);
%     disp(['Pemo-Q eval:  ' num2str(res_struct{methodix}.pemoscore)]);
    toc
end

% [x_rec_ellis,~,~] = invmelfcc(test_feats, sr,'maxfreq',maxfreq, ...
%                                 'wintime',wintime,'hoptime',hoptime, ...
%                                 'nbands',nbands,'numcep',numcep); 
                            
% pemo(ref_t_data',x_rec_ellis,sr)                    