function [Y_hat, mean_forward_error] = reg_thresholded(Xdev,Ydev,X,Y,covar,display, seuil, w_median)

if nargin <6
    display =0;
end

if  nargin<7
    seuil = 0.0;
end

if nargin <8
    w_median = 0;
end

Tdev = size(Xdev,2);
T = size(X,2);

%Forward test
Ktest_dev = covar(X,Xdev);
% weights = bsxfun(@times,Ktest_dev,1./sum(Ktest_dev,2));
weights = bsxfun(@times,Ktest_dev,1./max(Ktest_dev,2)');
%weights = Ktest_dev;
Y_hat = zeros(size(Y));
% seuil = 0.2;


for f=1:size(Ydev,1)/2
    disp(['f=' num2str(f)])
    [x_sort, order] = sort(Ydev(f,:), 'ascend');
    for t =1:T
        w_sort= weights(t,order);
        inds = find(w_sort >= seuil);
        
        if w_median
            % use a weighted median of elements above threshold
            i = inds(find(cumsum(w_sort(inds))/sum(w_sort(inds)) >= 0.5, 1 , 'first'));
            if ~isempty(i) 
                Y_hat(f,t) = x_sort(i);
            else
                Y_hat(f,t) = 0;
            end
        else
            % use the median on all elements above threshold
            Y_hat(f,t) = median(x_sort(inds));
        end

    end
    
end


Y_hat(end:-1:end-size(Ydev,1)/2 +2, :) = Y_hat(2:end/2,:);
%Y_hat = (weights*Ydev.').';
E_forward = Y - Y_hat;
mean_forward_error = mean(20*log10(max(1E-15,abs(E_forward))./abs(Y)),2);
% mean_forward = mean(mean_forward_error(:));


if display
    figure
    clf
    subplot 311
    imagesc(log(abs(Y)))
    title('Reponse originale')
    subplot 312
    imagesc(log(abs(Y_hat)))
    title('Reponse predite')
    subplot 313
    imagesc(abs(E_forward)./abs(Y))
    title('Erreur normalisee du probleme direct')
end
