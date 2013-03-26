function K = covariance(X1,X2, methode)
[N,T1] = size(X1);
[N,T2] = size(X2);

% methode = 1;

switch methode

    case 1
        % inverse norme L2
        K = zeros(T1,T2);
        for t1 = 1:T1
%             K(t1,:) = sum((X2-X1(:,t1)*ones(1,T2)).^2,1);
           K(t1,:) =  sum(bsxfun(@minus,X2,X1(:,t1)).^2,1);
%            K(t1,:) = N./sum(max(eps,abs(difference)).^2,1);
%            K(t1,:) = N./sum(max(eps,abs(difference)).^2,1);
        end
    case 2
        % produit scalaire
        K = X1.'*conj(X2);
    case 3
        % correlation corrcoef
        correls = abs(corrcoef([X1,X2]));
        K = correls(1:T1,T1+1:end);
    case 4
        % correlation corrcoef avec contexte
        X1_bar = [[diff(X1,1,2) X1(:,1)]; X1];
        X2_bar = [[diff(X2,1,2) X2(:,1)]; X2];
        correls = 1-abs(corrcoef([X1_bar,X2_bar]));
        K = correls(1:T1,T1+1:end);
    case 5
        % inverse norme Linf
        K = zeros(T1,T2);
        for t1 = 1:T1
           difference =  bsxfun(@minus,X2,X1(:,t1));
           K(t1,:) = max(max(eps,abs(difference)).^2)/N;
        end        
end