close all;
clear all;
parameters

% loading the learn base
params.n_frames = 20000;
params.sigma = 0.00001;
params.shuffle = 1001;
params.get_data = 1;
params.features = {'mfcc'};
[learn_feats, learn_magspecs, n_f_learn, ref_learn_data, learn_files] = load_data(params);

% loading the test base
params.n_frames = 1000;
params.sigma = 0.00001;
params.shuffle = 3001;
params.get_data = 1;
% params.location = '/sons/rwc/rwc-g-m02/';
[test_feats, test_magspecs, n_f_test, ref_t_data, test_files] = load_data(params);

mlearn =  cell2mat(learn_files');
tlearn =  cell2mat(test_files');

isinbase = ~isempty(intersect(mlearn,tlearn,'rows'))

%%  Trying a nadaraya-watson

methods_to_try = [1,5];

res_struct = {};

for methodix=1:length(methods_to_try)
    disp(['Working on method ' num2str(methodix)]);
    method = methods_to_try(methodix);
    % inverse l2 distance - mapping
    res_struct{methodix}.method = method;
    
    [res_struct{methodix}.m, res_struct{methodix}.err] = nadaraya_watson(learn_feats, learn_magspecs, test_feats, test_magspecs, @covariance,method,1,5);    
    disp(['mean l2 error of  ' num2str(mean(res_struct{methodix}.err))]);
    
    % sliding median filtering ?
    res_struct{methodix}.m_filt = medfilt1(res_struct{methodix}.m',3)';
    
    % reconstruction
    init_vec = randn(n_f_test*hoptime*sr,1);
    [res_struct{methodix}.x_rec] = gl_reconstruct(res_struct{methodix}.m_filt, init_vec, sr, 5, wintime*sr, hoptime*sr);

    % pemo-Q evaluation
    [res_struct{methodix}.pemoscore,~]= pemo(ref_t_data',res_struct{methodix}.x_rec,sr);
    disp(['Pemo-Q eval:  ' num2str(res_struct{methodix}.pemoscore)]);
    
end

[x_rec_ellis,~,~] = invmelfcc(test_feats, sr,'maxfreq',maxfreq, ...
                                'wintime',wintime,'hoptime',hoptime, ...
                                'nbands',nbands,'numcep',numcep); 
                            
pemo(ref_t_data',x_rec_ellis,sr)                    