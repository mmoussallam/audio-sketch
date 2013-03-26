function [x_rec] = gl_reconstruct(magspec, init_vec, fs, niter, winsize, hopsize, display)
% reconstruct from a magnitude spectrum
% uses Signal class stft
% must be initialized with a random vector or anything closer to the signal

if nargin <7
    display = 0;
end

% initialize signal
x_rec = init_vec;
sig = Signal(x_rec,fs);
sig.fs = fs;
sig.overlapRatio = 1-hopsize/winsize;
% sig.nfft = 2^(ceil(log2(winsize)));
sig.nfft = size(magspec,1);

% sig.windowLength = winsize;

for n=1:niter
        
    % compute stft of candidate
    sig.STFT();
    
    L = min(size(magspec,2), size(sig.S,2));
    % normalize its spectrum by target spectrum
    sig.S = (sig.S(:,1:L) ./ abs(sig.S(:,1:L))).*magspec(:,1:L);
        
    
    % resynthesize using inverse stft
    x_rec = sig.iSTFT();
    
    % plot the current signal and the spectrum
    if display>0
        figure(display)
        subplot(211)
        plot(x_rec)
        subplot(212)
        imagesc(log(abs(sig.S)));
    end
    
end

end