% for a variety of parameters, evaluate quality of reconstruction using
% nadaraya-watson scheme on features

% SAME AS BIGEXPE1 but the test sounds are chosen at random this time

close all;
clear all;

% for all combinations of these parameters
nb_learns = [100000];
nb_medians = [5];
nb_features = [1,3,8,21];
nb_trials = 1; %todo
methods = [1];

% evaluate using these parameters
nb_iter_gl = 5;
nb_test = 1000;
l_medfilt = 1;

% the evaluations scores are threefold:
spec_l2 = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
time_l2 = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
pemoQ = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
corr = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);
% compute the base once and for all
params.n_frames = max(nb_learns);
params.sigma = 0.00001;
params.shuffle = 8001;
params.get_data = 1;
params.features = {'zcr','OnsetDet','energy','specstats','specflux','mfcc','magspec'};
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
rescell = {};
maxPemoScore = 0;
bestSpecL2Score = inf;
bestTimeL2Score = inf;
bestCorrScore = 0;

for trialIdx=1:nb_trials
    
    % get the test data
    params.n_frames = nb_test;
    params.sigma = 0.00001;
    params.shuffle =  floor(rand(1)*1000);
    params.get_data = 1;
%     params.features = {'mfcc','zcr','lpc','mfcc_d1','magspec'};
%     params.location = '/sons/voxforge/main/16Khz_16bit/cmu_us_clb_arctic';
    [test_feats_all, test_magspecs, n_f_test, ref_t_data, test_files] = load_yaafedata(params);
    
    mlearn =  cell2mat(learn_files');
    tlearn =  cell2mat(test_files');
    
    isinbase = ~isempty(intersect(mlearn,tlearn,'rows'));
    if isinbase
        warning('test is in base!!!')
        trialIdx = trialIdx-1;
        continue;
    end
    
    for nli=1:length(nb_learns)
        nb_learn = nb_learns(nli);
        
        for mfi=1:length(nb_features)
            nb_feat = nb_features(mfi);
            
            
            
            % draw features and frames at random from the learned base
            featidxs = randperm(size(learn_feats_all,1));
            frameidxs= randperm(size(learn_feats_all,2));
            
            % get corresponding sub matrices for learning
            learn_feats = learn_feats_all(featidxs(1:nb_feat), frameidxs(1:nb_learn));
            learn_magspecs = learn_magspecs_all(:, frameidxs(1:nb_learn));
            
            % also subsample the corresponding feature matrix
            test_feats = test_feats_all(featidxs(1:nb_feat), :);
            
            % finally, loop on number of elements in the median
            for nmi=1:length(nb_medians)
                nb_median = nb_medians(nmi);
                
                
                for methodix=1:length(methods)
                    method = methods(methodix);
                    disp(['learn frames:' num2str(nb_learn) ', features :' num2str(nb_feat) ',medians: ' num2str(nb_median) ,', method: ' num2str(method)]);
                    
                    res_struct = eval_nw( learn_feats, learn_magspecs, test_feats , ...
                        test_magspecs, ref_t_data, ...
                        nb_median, nb_iter_gl, l_medfilt, method);
                    
                    
                    rescell{length(rescell)+1} = res_struct;
                    
                    spec_l2(nli, mfi, nmi, methodix, trialIdx) = res_struct.spec_err;
                    time_l2(nli, mfi, nmi, methodix, trialIdx) = res_struct.wf_err;
%                     pemoQ(nli, mfi, nmi, methodix, trialIdx) = res_struct.pemoscore;
%                     corr(nli, mfi, nmi, methodix, trialIdx) = res_struct.corr;
                    
                    dosave = 0;
                    if res_struct.spec_err < bestSpecL2Score
                        dosave = 1;
                        bestSpecL2Score = res_struct.spec_err;
                        minL2results.nb_learn = nb_learn;
                        minL2results.nb_feat = nb_feat;
                        minL2results.nb_median = nb_median;
                        minL2results.method = res_struct.method;
                        minL2results.x_rec = res_struct.x_rec;
                        disp(['New min Spec L2 score : ' num2str(bestSpecL2Score)]);
                    end
                    
                    if res_struct.wf_err < bestTimeL2Score
                        dosave = 1;
                        bestTimeL2Score = res_struct.wf_err;
                        minTimeL2results.nb_learn = nb_learn;
                        minTimeL2results.nb_feat = nb_feat;
                        minTimeL2results.nb_median = nb_median;
                        minTimeL2results.method = res_struct.method;
                        minTimeL2results.x_rec = res_struct.x_rec;
                        disp(['New min Time L2 score : ' num2str(bestTimeL2Score)]);
                    end
                    
%                     if res_struct.corr > bestCorrScore
%                         dosave = 1;
%                         bestCorrScore = res_struct.corr;
%                         maxCorrScore.nb_learn = nb_learn;
%                         maxCorrScore.nb_feat = nb_feat;
%                         maxCorrScore.nb_median = nb_median;
%                         maxCorrScore.method = res_struct.method;
%                         maxCorrScore.x_rec = res_struct.x_rec;
%                         disp(['New max correlation of : ' num2str(bestCorrScore)]);
%                     end
%                     
%                     if res_struct.pemoscore > maxPemoScore
%                         dosave = 1;
%                         maxPemoScore = res_struct.pemoscore;
%                         maxPemoresults.nb_learn = nb_learn;
%                         maxPemoresults.nb_feat = nb_feat;
%                         maxPemoresults.nb_median = nb_median;
%                         maxPemoresults.method = res_struct.method;
%                         maxPemoresults.x_rec = res_struct.x_rec;
%                         disp(['New best PEMO score : ' num2str(maxPemoScore)]);
%                     end
                    
                    if dosave
                        % result is interesting, keep it
                        save_res_name = ['results/res_struct_' num2str(nb_learn) '_' num2str(nb_feat) '_' num2str(nb_median) '_' num2str(method) '.mat'] ;
                        save(save_res_name, 'res_struct');
                    end
                end
                
            end
        end
    end
    
end

parameters
save(['BigExpe2_results_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.mat'], ...
    'spec_l2','time_l2','pemoQ','nb_learns','nb_medians','nb_features','nb_trials','methods')
% 
% %% dealing with the results
% 
% % what are pure noise performances?
% ref_l2 = 20*log10(norm(test_magspecs - abs(randn(size(test_magspecs)))));
% noise = randn(size(ref_t_data'));
% ref_pemo = pemo(ref_t_data',noise/norm(ref_t_data),sr);
% 
% % influence of number of training samples
% figure
% subplot(121)
% plot(nb_learns, get_mean_along(pemoQ,1));hold on;
% plot(nb_learns,ref_pemo*ones(length(nb_learns),1),'k--');
% title('Influence of training base size');
% xlabel('learning samples');grid on
% ylabel('Pemo-Q Score');
% % ylim([0,1])
% % subplot(222)
% % plot(nb_features, get_mean_along(pemoQ,2));hold on;
% % plot(nb_features,ref_pemo*ones(length(nb_features),1),'k--');
% % title('Influence of feature vec size');
% % xlabel('number of features');grid on
% % ylabel('Pemo-Q Score');ylim([0,1])
% subplot(122)
% plot(nb_medians, get_mean_along(pemoQ,3));hold on;
% plot(nb_medians,ref_pemo*ones(length(nb_medians),1),'k--');
% title('Influence of number of medians');
% xlabel('number of elements combined');grid on
% ylabel('Pemo-Q Score');ylim([0,1])
% % subplot(122)
% % plot(methods, get_mean_along(pemoQ,4))
% % title('Influence of chosen method');ylim([0,1]);hold on;
% % plot(methods,ref_pemo*ones(length(methods),1),'k--');
% % xlabel('method');grid on
% % ylabel('Pemo-Q Score');
% saveas(gcf,['Figures/BigExpe2_PemoQ_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.fig']);
% saveas(gcf,['Figures/BigExpe2_PemoQ_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.png']);
% 
% % influence of number of training samples
% figure
% subplot(121)
% plot(nb_learns, get_mean_along(spec_l2,1))
% title('Influence of training base size');
% xlabel('learning samples');
% ylabel('Spectrogram error');grid on
% % subplot(222)
% % plot(nb_features, get_mean_along(spec_l2,2))
% % title('Influence of feature vec size');
% % xlabel('number of features');
% % ylabel('Spectrogram error');grid on
% subplot(122)
% plot(nb_medians, get_mean_along(spec_l2,3))
% title('Influence of number of medians');
% xlabel('number of elements combined');
% ylabel('Spectrogram error');grid on
% % subplot(122)
% % plot(methods, get_mean_along(spec_l2,4))
% % title('Influence of chosen method');grid on
% % xlabel('method');
% % ylabel('Spectrogram error');
% saveas(gcf,['Figures/BigExpe2_SpecErr_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.fig']);
% saveas(gcf,['Figures/BigExpe2_SpecErr_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.png']);
% 
% % influence of number of training samples
% figure
% subplot(121)
% plot(nb_learns, get_mean_along(time_l2,1))
% title('Influence of training base size');
% xlabel('learning samples');
% ylabel('Time domain error');grid on
% % subplot(222)
% % plot(nb_features, get_mean_along(time_l2,2))
% % title('Influence of feature vec size');
% % xlabel('number of features');grid on
% % ylabel('Time domain error');
% subplot(122)
% plot(nb_medians, get_mean_along(time_l2,3))
% title('Influence of number of medians');
% xlabel('number of elements combined');grid on
% ylabel('Time domain error');
% subplot(122)
% % plot(methods, get_mean_along(time_l2,4))
% % title('Influence of chosen method');grid on
% % xlabel('method');
% % ylabel('Time domain error');
% saveas(gcf,['Figures/BigExpe2_TimeErr_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.fig']);
% saveas(gcf,['Figures/BigExpe2_TimeErr_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.png']);
% 
% 
% figure
% subplot(121)
% plot(nb_learns, get_mean_along(corr,1))
% title('Influence of training base size');
% xlabel('learning samples');
% ylabel('Max of normed corr');grid on
% % subplot(222)
% % plot(nb_features, get_mean_along(corr,2))
% % title('Influence of feature vec size');
% % xlabel('number of features');grid on
% % ylabel('Max of normed corr');
% subplot(122)
% plot(nb_medians, get_mean_along(corr,3))
% title('Influence of number of medians');
% xlabel('number of elements combined');grid on
% ylabel('Max of normed corr');
% subplot(122)
% % plot(methods, get_mean_along(corr,4))
% % title('Influence of chosen method');grid on
% % xlabel('method');
% % ylabel('Max of normed corr');
% saveas(gcf,['Figures/BigExpe2_corr_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.fig']);
% saveas(gcf,['Figures/BigExpe2_corr_' num2str(max(nb_learns)) '_for_' audio_file_path(end-16:end) '_' num2str(nb_trials) 'trials.png']);
% 
