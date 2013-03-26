function K = weight_covariance(X1,X2, W)
% [N,T1] = size(X1);
% [N,T2] = size(X2);

% methode = 1;

if ismatrix(W)
    
    K = abs(X1'*W*X2);
    
elseif ndims(W)==3
    K = zeros(size(X1,2), size(X2,2), size(W,3));
    for freq=1:size(W,3)
        K(:,:,freq) =  X1'*squeeze(W(:,:,freq))*X2;
    end
end
end