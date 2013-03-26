function [Y_hat, mean_forward_error] = nadaraya_watson(Xdev,Ydev,X,Y,covar,method,display,K)
Tdev = size(Xdev,2);
T = size(X,2);

mu = mean(Xdev, 2);
sigma = std(Xdev, [], 2);
%Forward test

machin = @(x)bsxfun(@times, bsxfun(@minus,x,mu), 1./sigma);

if method < 5
    Ktest_dev = covar(machin(X),machin(Xdev), method);
else
    Ktest_dev = weight_covariance(X, Xdev, mahalanobis_estimate(nan,Ydev,Xdev,display));
end
%Ktest_dev_orig = covar(X, Xdev, method);
%Forward test
weights = bsxfun(@times,Ktest_dev,1./sum(Ktest_dev,2));
[~,order] = sort(weights,2,'descend');

for t =1:T
    Y_hat(:,t) = median(Ydev(:,order(t,1:K)),2);
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
