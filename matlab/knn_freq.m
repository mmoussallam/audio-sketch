function [Y_hat, mean_forward_error] = knn_freq(Xdev,Ydev,X,Y,covar,method,display,K)
Tdev = size(Xdev,2);
T = size(X,2);
seuil = 0.5;
mu = mean(Xdev, 2);
sigma = std(Xdev, [], 2);
%Forward test

machin = @(x)bsxfun(@times, bsxfun(@minus,x,mu), 1./sigma);

% pre-allocating
Y_hat = zeros(size(Y));

if method <= 5
    % Any method based on simple covariance method 
%     Ktest_dev = covar(machin(X),machin(Xdev), method);
% %         weights = bsxfun(@times,Ktest_dev,1./sum(Ktest_dev,2));
%     [~,order] = sort(Ktest_dev,2,'ascend');

    for t =1:T
        Ktest_dev = covar(machin(X(:,t)),machin(Xdev), method);
%         weights = bsxfun(@times,Ktest_dev,1./sum(Ktest_dev,2));
%         [~,order] = sort(Ktest_dev(:),'ascend');
        [~,order] = mink(Ktest_dev,K);
        Y_hat(:,t) = median(Ydev(:,order),2);
    end
   
elseif method==6
    % method using the estimated mahalanobis distance on spectrums to
    % weight the covariance matrix
    W = mahalanobis_estimate(nan,Ydev,Xdev,display);
    if display
        figure
        subplot(211)
        imagesc(covariance(Y,Ydev,3)');
        subplot(212)
        imagesc(abs(Xdev'*W*X));
        
        figure
        imagesc(W)
    end
    Ktest_dev = weight_covariance(X, Xdev, W);
    
%     weights = bsxfun(@times,Ktest_dev,1./sum(Ktest_dev,2));
    weights = Ktest_dev;
    weights(weights<seuil) = -inf;
    [~,order] = sort(weights,2,'descend');

    for t =1:T
        
        Y_hat(:,t) = median(Ydev(:,order(t,1:K)),2);
    end

elseif method==7
    % Same as 6 but distance is calculated separately for each frequency
    % bin
     Ktest_dev_freq = weight_covariance(X, Xdev, mahalanobis_estimate_freq(nan,Ydev,Xdev,display));
     for freq=1:size(Ktest_dev_freq,3)
        weights = bsxfun(@times,Ktest_dev_freq(:,:,freq),1./sum(Ktest_dev_freq(:,:,freq),2));
        [~,order] = sort(weights,2,'descend');

        for t =1:T
            Y_hat(freq,t) = median(Ydev(freq,order(t,1:K)));
        end
     end
   
elseif method>7
    % Same as 7 but distance is calculated separately for several frequency
    % bands (10 so far: parameterize);
    nband = method;
    Lband = floor(size(Ydev,1)/nband);
    Ktest_dev_freq = weight_covariance(X, Xdev, mahalanobis_estimate_freq(nan,Ydev,Xdev,display,nband));
     
     for band=1:size(Ktest_dev_freq,3)
         Ktest_dev_band = squeeze(Ktest_dev_freq(:,:,band));
        weights = bsxfun(@times,Ktest_dev_band,1./sum(Ktest_dev_band,2));
%         weights = Ktest_dev_band;
        [~,order] = sort(weights,2,'descend');

        for t =1:T
            Y_hat((band-1)*Lband+1:band*Lband,t) = median(Ydev((band-1)*Lband+1:band*Lband,order(t,1:K)),2);
        end
     end
          
     
end

%Y_hat = (weights*Ydev.').';
E_forward = Y - Y_hat;
mean_forward_error = mean(20*log10(max(1E-15,(norm(abs(E_forward))))./norm(abs(Y))),2);
mean_forward = mean(mean_forward_error(:));


if display
    figure
    clf
    subplot 211
    imagesc(log(abs(Y)))
    title('Reponse originale')
    subplot 212
    imagesc(log(abs(Y_hat)))
    title('Reponse predite')
%     subplot 313
%     imagesc(abs(E_forward)./abs(Y))
%     title('Erreur normalisee du probleme direct')
end
