% lecture des resultats et figure
close all;
clear all;

% for all combinations of these parameters
nb_learns = [1,2,5,10,20,50,100]*1000;
nb_medians = [10];
nb_features = [1,2,3,5,7,10,15,20];
nb_trials = 10; %todo
methods = [1];

% evaluate using these parameters
nb_iter_gl = 5;
nb_test = 1000;
l_medfilt = 1;

spec_l2 = zeros(length(nb_learns), length(nb_features), length(nb_medians), length(methods), nb_trials);

params.n_frames = max(nb_learns);
params.sigma = 0.00001;
params.shuffle = 8001;
params.get_data = 1;
params.features = {'zcr','OnsetDet','energy','specstats','specflux','mfcc','magspec'};

for trialIdx=1:nb_trials
    
    for nli=1:length(nb_learns)
        nb_learn = nb_learns(nli);
        
        for mfi=1:length(nb_features)
            nb_feat = nb_features(mfi);
            
            for nmi=1:length(nb_medians)
                nb_median = nb_medians(nmi);
                
                for methodix=1:length(methods)
                    method = methods(methodix);
                    
                    % load the data
                    save_res_name = ['results/ACMMM13/res_struct_' num2str(nb_learn) '_' num2str(nb_feat) '_' num2str(nb_median) '_' num2str(method) '_trial_' num2str(trialIdx) '.mat'] ;
                    
                    loadeddata =load(save_res_name, 'res_struct');
                    spec_l2(nli, mfi, nmi, methodix, trialIdx) = loadeddata.res_struct.spec_err;
                    
                end
            end
        end
    end
end

%%

figure
imagesc(squeeze(mean(spec_l2,5))); axis xy;
ylabel('Training Size');
xlabel('Features');
set(gca,'XTickLabel',nb_features);
set(gca,'YTickLabel',nb_learns);

%% Spectrogram figure
% 
% for trialIdx=1:nb_trials
% figure
% subplot(311)
% imagesc(flipud(log(test_magspecs)));
% title('Original','Interpreter','Latex');
% set(gca,'XTickLabel',[]);
% set(gca,'YTickLabel',[]);
% subplot(312)
% imagesc(flipud(log(rescell{2}.m)));
% title('Features: Zcr, Onset and Energy','Interpreter','Latex');
% set(gca,'XTickLabel',[]);
% set(gca,'YTickLabel',[]);
% subplot(313)
% imagesc(flipud(log(rescell{3}.m)));
% title('Features: Zcr, Onset, Energy and Spectral Stats','Interpreter','Latex');
% set(gca,'XTickLabel',[]);
% set(gca,'YTickLabel',[]);
% xlabel('Time (s)','Interpreter','Latex');
% saveas(newfig,['Figures/ACMMM13/MagSpectros_' num2str(nb_learn) '_' num2str(nb_feat) '_' num2str(nb_median) '_' num2str(method) '_trial_' num2str(trialIdx) '.fig'])


