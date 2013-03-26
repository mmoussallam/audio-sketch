function W = mahalanobis_estimate(distance_handle, S,F, display)

if nargin < 4
    display = 0;
end

% dimension
T = size(S,2);
M = size(F,1);
 
% Building ground_truth similarity matrix
A = (corrcoef(S));
% A = zeros(T,T);
% parfor t = 1:T
%     A(t,:) = sum(bsxfun(distance_handle, S,S(:,t)));
% end

%Estimating weights
FFt = F*F'+1e-2*eye(M);
W = FFt\F*A*F'/FFt;

if display
    figure(10)
    clf
    subplot 311
    imagesc(A)
    title('Ground truth similarity between developement spectra')
    subplot 312
    imagesc(F'*F)
    title('Original dot-products between features')
    subplot 313
    imagesc(F'*W*F)
    title('Learned Mahalanobis distance between features')
    
end