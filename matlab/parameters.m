% Load the parameters
% adding rightful libraries
addpath('/home/manu/workspace/toolboxes/Matlab_code/Signal');
addpath('/home/manu/workspace/toolboxes/Matlab_code/pemo');
addpath('/home/manu/workspace/toolboxes/Matlab_code/pemo/gammatone');
addpath('/home/manu/workspace/toolboxes/Matlab_code/rastamat');
addpath('/home/manu/workspace/toolboxes/Matlab_code/Enhanced_RDIR/rdir_0112');
% audio file path
audio_file_path = '/sons/voxforge/main/16Khz_16bit/';
audio_file_path = '/sons/rwc/rwc-g-m01/';

% signal analysis
wintime = 0.032;
hoptime = 0.008;
sr = 16000;
max_frame_num_per_file = 500;

% learning parameters
n_learn_frames = 2000;   % number of learning frames
n_features = 13;         % number of used features
n_medians = 1;           % number of elements on which median is applied

% mfcc parameters
dither = 0;
nbands = 40;
fbtype ='mel';
minfreq = 0;
maxfreq = 8000;
sumpower = 1;
bwidth = 1.0;
preemph = 0.97;
numcep = 13;
dcttype = 2;
lifterexp = 0.6;
