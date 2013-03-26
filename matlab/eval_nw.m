function [ res_struct ] = eval_nw( learn_feats, learn_magspecs, test_feats , ...
                                test_magspecs, ref_t_data, ...
                                nb_medians, nb_iter_gl, l_medfilt, method)
%EVAL_NW Summary of this function goes here
%   Detailed explanation goes here

% methods_to_try = [1,5];
parameters
% res_struct = cell(length(methods_to_try));


% inverse l2 distance - mapping
res_struct.method = method;

[res_struct.m, err] = knn_freq(learn_feats, learn_magspecs, test_feats, test_magspecs, @covariance,method,0,nb_medians);                            
res_struct.spec_err = mean(err);
disp(['mean magspec l2 error of  ' num2str(res_struct.spec_err)]);

% sliding median filtering ?
% res_struct.m_filt = medfilt1(res_struct.m',l_medfilt)';

res_struct.m_filt = res_struct.m;

% reconstruction
init_vec = randn(size(test_magspecs,2)*hoptime*sr,1);
[res_struct.x_rec] = gl_reconstruct(res_struct.m_filt, init_vec, sr, nb_iter_gl, wintime*sr, hoptime*sr);

% l2 norm in the time domain? or max xcorr?
% res_struct.wf_err = 20*log10(norm(ref_t_data(1:length(init_vec))' - res_struct.x_rec));
% disp(['mean time l2 error of  ' num2str(res_struct.wf_err)]);

% res_struct.corr = max(xcorr(ref_t_data(1:length(init_vec)), res_struct.x_rec, 200,'coeff'));
% disp(['max correlation of  ' num2str(res_struct.corr)]);

% pemo-Q evaluation
% [res_struct.pemoscore,~]= pemo(ref_t_data(1:length(init_vec))',res_struct.x_rec,sr);
% disp(['Pemo-Q eval:  ' num2str(res_struct.pemoscore)]);




end

