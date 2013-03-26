function W = mahalanobis_estimate_freq(distance_handle, S,F, display, nband)
% same as mahalanobis_estimate but one matrix W for each frequency band

if nargin < 4
    display = 0;
end

if nargin < 5
    nband = size(S,1);
end

% dimension
T = size(S,2);
f =  size(S,1);
M = size(F,1);


W = zeros(M,M,nband);

Lband = floor(f/nband);

for band =1:nband    
    A = (corrcoef(S((band-1)*Lband+1:band*Lband,:)));
    FFt = F*F'+0.1*eye(M);
    W(:,:,band) = FFt\F*A*F'/FFt;
end
% Building ground_truth similarity matrix

% A = zeros(T,T);
% parfor t = 1:T
%     A(t,:) = sum(bsxfun(distance_handle, S,S(:,t)));
% end

% %Estimating weights
% FFt = F*F'+1E-5*eye(M);
% W = FFt\F*A*F'/FFt;
% 
% if display
%     figure(10)
%     clf
%     subplot 311
%     imagesc(A)
%     title('Ground truth similarity between developement spectra')
%     subplot 312
%     imagesc(F'*F)
%     title('Original dot-products between features')
%     subplot 313
%     imagesc(F'*W*F)
%     title('Learned Mahalanobis distance between features')
%     
% end