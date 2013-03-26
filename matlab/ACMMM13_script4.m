% Plot the spectrograms for 100 000 frames of learning from different
% speakers and 1000 frames of test for a different sentence with a
% different speaker

% for a variety of parameters, evaluate quality of reconstruction using
% nadaraya-watson scheme on features

% close all;
clear all;

% for all combinations of these parameters
nb_learn = 100000;
nb_median = 10;
nb_features = [7];
nb_trials = 1; %todo
method = 1;

% evaluate using these parameters
nb_iter_gl = 5;
nb_test = 1000;
l_medfilt = 1;

% the evaluations scores are threefold:
% spec_l2 = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
% time_l2 = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
% pemoQ = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
% corr = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
% compute the base once and for all
params.n_frames = nb_learn;
params.sigma = 0.00001;
params.shuffle = 7001;
params.get_data = 1;
params.features = {'zcr','OnsetDet','energy','specstats','mfcc','magspec'};
% params.features = {'mfcc','magspec'};
savematname = ['learnbase_allfeats_' num2str(params.n_frames) '_seed_' num2str(params.shuffle) '.mat'];
if fopen(savematname)<0
    [learn_feats_all, learn_magspecs_all, n_f_learn, ref_learn_data, learn_files] = load_yaafedata(params);
    save(savematname, 'learn_feats_all', 'learn_magspecs_all', 'learn_files');
else
    lstruct = load(savematname);
    learn_feats_all = lstruct.learn_feats_all;
    learn_magspecs_all = lstruct.learn_magspecs_all;
    learn_files = lstruct.learn_files;
end

for trialIdx=1:nb_trials
    isinbase = 1;
    
    
    while isinbase
        % get the test data
        params.n_frames = nb_test;
        params.sigma = 0.00001;
        params.shuffle =  floor(rand(1)*1000);
        params.get_data = 1;
        %     params.features = {'mfcc','zcr','lpc','mfcc_d1','magspec'};
        params.location = '/sons/voxforge/main/Test';
        [test_feats_all, test_magspecs, n_f_test, ref_t_data, test_files] = load_yaafedata(params);
        
        mlearn =  cell2mat(learn_files');
        tlearn =  cell2mat(test_files');
        
        isinbase = ~isempty(intersect(mlearn(:,end-15:end),tlearn(:,end-15:end),'rows'));
        
    end
    
    rescell = cell(length(nb_features));
    for nfi=1:length(nb_features)
        nb_feat = nb_features(nfi);
    learn_feats = learn_feats_all(1:nb_feat, 1:nb_learn);
    learn_magspecs = learn_magspecs_all(:, 1:nb_learn);
    
    test_feats = test_feats_all(1:nb_feat, :);
    
    
    disp(['Trial: ' num2str(trialIdx) 'learn frames:' num2str(nb_learn) ', features :' num2str(nb_feat) ',medians: ' num2str(nb_median) ,', method: ' num2str(method)]);
    
    res_struct = eval_nw( learn_feats, learn_magspecs, test_feats , ...
        test_magspecs, ref_t_data, ...
        nb_median, nb_iter_gl, l_medfilt, method);
    
    
    rescell{nfi} = res_struct;
    end
%     spec_l2(nli, mfi, nmi, methodix, trialIdx) = res_struct.spec_err;
%     %                     time_l2(nli, mfi, nmi, methodix, trialIdx) = res_struct.wf_err;
%     res_struct.trial = trialIdx;
%     dosave = 1;
%     
%     if dosave
%         % result is interesting, keep it
%         save_res_name = ['results/ACMMM13/res_struct_' num2str(nb_learn) '_' num2str(nb_feat) '_' num2str(nb_median) '_' num2str(method) '_trial_' num2str(trialIdx) '.mat'] ;
%         save(save_res_name, 'res_struct');
%     end
    
    
    slice = (1:1000);
    
    newfig = figure(trialIdx);
    subplot(311)
    imagesc(flipud(log(test_magspecs(:,slice))));
    title('Original','Interpreter','latex','FontSize',12.0);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    subplot(312)
    imagesc(flipud(log(rescell{1}.m(:,slice))));
    title('$M=3$','Interpreter','latex','FontSize',12.0);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    subplot(313)
    imagesc(flipud(log(rescell{2}.m(:,slice))));
    title('$M=7$','Interpreter','latex','FontSize',12.0);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);    
    linkaxes
    saveas(newfig,['Figures/ACMMM13/MagSpectros_' num2str(nb_learn) '_' num2str(nb_feat) '_' num2str(nb_median) '_' num2str(method) '_trial_' num2str(trialIdx) '.fig'])
    
end

% parameters
% save(['ACMMM13_results_' num2str(max(nb_learns)) '_' num2str(nb_trials) 'trials.mat'], ...
%     'spec_l2','nb_learns','nb_medians','nb_features','nb_trials','methods')

%%
