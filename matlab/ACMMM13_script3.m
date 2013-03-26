% lecture des resultats et figure
close all;
clear all;

% for all combinations of these parameters
nb_learns = [100000];
nb_medians = [10];
nb_features = [7];
nb_trials = 1; %todo
method = 1;

% evaluate using these parameters
nb_iter_gl = 5;
nb_test = 1000;
l_medfilt = 1;

spec_l2 = zeros(length(nb_learns), length(nb_features), length(nb_medians), nb_trials);

params.n_frames = max(nb_learns);
params.sigma = 0.00001;
params.shuffle = 1001;
params.get_data = 1;
params.features = {'zcr','OnsetDet','energy','specstats','magspec'};

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

maxPemoScore = 0;
bestSpecL2Score = inf;
bestTimeL2Score = inf;
bestCorrScore = 0;



for trialIdx=1:nb_trials
    isinbase = 1;
    rescell = {};
    
    while isinbase
        % get the test data
        params.n_frames = nb_test;
        params.sigma = 0.00001;
        params.shuffle =  floor(rand(1)*10000);
        params.get_data = 1;
    %     params.features = {'mfcc','zcr','lpc','mfcc_d1','magspec'};
        params.location = '/sons/voxforge/main/Test';
        

        [test_feats_all, test_magspecs, n_f_test, ref_t_data, test_files] = load_yaafedata(params);

        mlearn =  cell2mat(learn_files');
        tlearn =  cell2mat(test_files');

        isinbase = ~isempty(intersect(mlearn(:,end-15:end),tlearn(:,end-15:end),'rows'));

    end
    
    for nli=1:length(nb_learns)
        nb_learn = nb_learns(nli);
        
        for mfi=1:length(nb_features)
            nb_feat = nb_features(mfi);
            
                        
            % draw features and frames at random from the learned base
%             featidxs = randperm(size(learn_feats_all,1));
%             frameidxs= randperm(size(learn_feats_all,2));
            
            % get corresponding sub matrices for learning
%             learn_feats = learn_feats_all(featidxs(1:nb_feat), frameidxs(1:nb_learn));
%             learn_magspecs = learn_magspecs_all(:, frameidxs(1:nb_learn));
            
            % also subsample the corresponding feature matrix
%             test_feats = test_feats_all(featidxs(1:nb_feat), :);
            
            learn_feats = learn_feats_all(1:nb_feat, 1:nb_learn);
            learn_magspecs = learn_magspecs_all(:, 1:nb_learn);

            test_feats = test_feats_all(1:nb_feat, :);
            
            
            
            Xdev = learn_feats;
            X = test_feats;
            Ydev = learn_magspecs;
            Y = test_magspecs;
            T = size(X,2);
            
            mu = mean(Xdev, 2);
            sigma = std(Xdev, [], 2);
            
            machin = @(x)bsxfun(@times, bsxfun(@minus,x,mu), 1./sigma);
            
%             Ktest_dev = zeros(size(Xdev,2),T);
%             for t =1:T
%                         Ktest_dev(:,t) = covariance(machin(X(:,t)),machin(Xdev), method);
%                 %         weights = bsxfun(@times,Ktest_dev,1./sum(Ktest_dev,2));
%                 %         [~,order] = sort(Ktest_dev(:),'ascend');
%                         
%             end
            Ktest_dev = covariance(machin(X),machin(Xdev), method);
            
            % finally, loop on number of elements in the median
            for nmi=1:length(nb_medians)
                nb_median = nb_medians(nmi);
                

                    disp(['learn frames:' num2str(nb_learn) ', features :' num2str(nb_feat) ',medians: ' num2str(nb_median) ,', method: ' num2str(method)]);
                    
%                     res_struct = eval_nw( learn_feats, learn_magspecs, test_feats , ...
%                         test_magspecs, ref_t_data, ...
%                         nb_median, nb_iter_gl, l_medfilt, method);
%                     
                    

                    % pre-allocating
                    Y_hat = zeros(size(Y));
                    for t =1:T
%                         Ktest_dev = covar(machin(X(:,t)),machin(Xdev), method);
                        
                %         [~,order] = sort(Ktest_dev(:),'ascend');
                        [~,order] = mink(Ktest_dev(:,t),nb_median);
                        
                        Y_hat(:,t) = median(Ydev(:,order),2);
                    end
                    E_forward = Y - Y_hat;
                    error = 20*log10(max(1E-15,(norm(abs(E_forward))))./norm(abs(Y)))                    
%                     rescell{length(rescell)+1} = res_struct;
                    
                    subplot(length(nb_medians),1,nmi)
                    imagesc(log(Y_hat));

                    spec_l2(nli, mfi, nmi,  trialIdx) = error;
%                     time_l2(nli, mfi, nmi, methodix, trialIdx) = res_struct.wf_err;
% %                     res_struct.trial = trialIdx;
%                     dosave = 1;
%                     
%                     if dosave
%                         % result is interesting, keep it
%                         save_res_name = ['results/ACMMM13/res_struct_' num2str(nb_learn) '_' num2str(nb_feat) '_' num2str(nb_median) '_' num2str(method) '_trial_' num2str(trialIdx) '.mat'] ;
%                         save(save_res_name, 'res_struct');
%                     end
       
                
            end
        end
    end
end
%%

figure
semilogx(nb_medians,squeeze(mean(spec_l2,4))); 
grid on;
ylabel('Error (dB)','Interpreter','Latex');
xlabel('P','Interpreter','Latex');
