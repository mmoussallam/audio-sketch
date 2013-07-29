'''
classes.sketches.MiscSketches  -  Created on Jul 25, 2013
@author: M. Moussallam
'''

from base import *

class SWSSketch(AudioSketch):
    """ Sine Wave Speech """

    def __init__(self, orig_sig=None, **kwargs):

        self.params = {'n_formants': 5,
                       'time_step': 0.01,
                       'windowSize': 0.025,
                       'preEmphasis': 50,
                       'script_path': '/home/manu/workspace/audio-sketch/src/tools'}

        for key in kwargs:
            self.params[key] = kwargs[key]
        self.call_str = 'praat %s/getsinewavespeech.praat' % self.params[
            'script_path']
        if orig_sig is not None:
            self.recompute(orig_sig, **kwargs)
        

    def get_sig(self):
        strret = '_nformants-%d_step_%1.3f' % (self.params['n_formants'], self.params['time_step'])
        return strret

    def synthesize(self, sparse=False):
        if not sparse:
            return self.rep
        else:
            return self.sp_rep

    def represent(self, fig=None, sparse=False):
        """ plotting the sinewave speech magnitude spectrogram"""
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()
        if not sparse:
            self.rep.spectrogram(
                2 ** int(np.log2(self.params['windowSize'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['time_step'] * self.rep.fs)),
                order=1, log=True, ax=ax)
        else:
            self.sp_rep.spectrogram(
                2 ** int(np.log2(self.params['windowSize'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['time_step'] * self.rep.fs)),
                order=1, log=True, ax=ax)

    def fgpt(self):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def _extract_sws(self, signal, kwargs):
        """ internal routine to extract the sws"""
        for key in kwargs:
            self.params[key] = kwargs[key]

        if signal is None or not os.path.exists(signal):
            raise ValueError("A valid path to a wave file must be provided")
        self.current_sig = signal        
        os.system('%s %s %1.3f %d %1.3f %d' % (self.call_str, signal,
                                               self.params['time_step'],
                                               self.params['n_formants'],
                                               self.params['windowSize'],
                                               self.params['preEmphasis']))
    # Now retrieve the coefficients and the resulting audio
        self.formants = []
        for forIdx in range(1, 4):
            formant_file = signal[:-4] + '_formant' + str(forIdx) + '.mtxt'
            fid = open(formant_file, 'rU')
            vals = fid.readlines()
            fid.close()  # remove first 3 lines and convert to numpy
            self.formants.append(np.array(vals[3:], dtype='float'))

    def recompute(self, signal=None, **kwargs):
        self._extract_sws(signal, kwargs)

        # save as the original signal if it's not already set
        if self.orig_signal is None:
            self.orig_signal = Signal(signal, normalize=True)
        # and the audio we said
        self.rep = Signal(signal[:-4] + '_sws.wav', normalize=True)

    def sparsify(self, sparsity):
        """ use sparsity to determine the number of formants and window size/steps """

        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= self.rep.length

        new_tstep = float(self.rep.get_duration() * self.params['n_formants'])/float(sparsity)
#            int(sparsity) * self.params['n_formants']) / float(self.rep.fs)
        new_wsize = new_tstep * 2
        print "New time step of %1.3f seconds" % new_tstep
        self._extract_sws(self.current_sig, {'time_step':
                          new_tstep, 'windowSize': new_wsize})
        self.sp_rep = Signal(
            self.current_sig[:-4] + '_sws.wav', normalize=True)


class KNNSketch(AudioSketch):
    """ Reconstruct the target audio from most similar
        ones in a database """

    def __init__(self, original_sig=None, **kwargs):
        """ creating a KNN-based sketchifier """

            # default parameters
        self.params = {
            'n_frames': 100000,
            'shuffle': 0,
            'wintime': 0.032,
            'steptime': 0.008,
            'sr': 16000,
            'frame_num_per_file': 2000,
            'features': ['zcr', 'OnsetDet', 'energy', 'specstats'],  # ,'mfcc','chroma','pcp']
            'location': '',
            'n_neighbs': 3,
            'l_filt': 3}

        for key in kwargs:
            self.params[key] = kwargs[key]

        if original_sig is not None:
            self.orig_signal = original_sig
            self.recompute()
    
    def __del__(self):
        """ some cleanup of instantiated variables """
        try:
            del self.l_feats_all, self.l_magspecs_all, self.learn_files_all
            del self.sp_rep, self.sp_rep
        except:
            pass

    def synthesize(self, sparse=False):
        if not sparse:
            return self.rep
        else:
            return self.sp_rep

    def get_sig(self):
        strret = '_nframes-%d_seed-%d_%dNN' % (self.params['n_frames'],
                                               self.params['shuffle'],
                                               self.params['n_neighbs'])
        return strret

    def represent(self, fig=None, sparse=False):
        """ plotting the sinewave speech magnitude spectrogram"""
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()
        if not sparse:
            self.rep.spectrogram(
                2 ** int(np.log2(self.params['wintime'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['steptime'] * self.rep.fs)),
                order=1, log=True, ax=ax)
        else:
            self.sp_rep.spectrogram(
                2 ** int(np.log2(self.params['wintime'] * self.rep.fs)),
                2 ** int(np.log2(
                         self.params['steptime'] * self.rep.fs)),
                order=1, log=True, ax=ax)

    def fgpt(self):
        raise NotImplementedError(
            "NOT IMPLEMENTED: ABSTRACT CLASS METHOD CALLED")

    def recompute(self, signal=None, **kwargs):
        for key in kwargs:
            self.params[key] = kwargs[key]

        """ recomputing the target """
        if signal is not None:
            self.orig_signal = Signal(signal, normalize=True, mono=True)

        if self.orig_signal is None:
            raise ValueError("No original Sound has been given")

        # Loading the database
        from scipy.io import loadmat
        from feat_invert.features import load_data_one_audio_file
        savematname = '%slearnbase_allfeats_%d_seed_%d.mat' % (self.params['location'],
                                                               self.params[
                                                               'n_frames'],
                                                               self.params['shuffle'])
        lstruct = loadmat(savematname)
        self.l_feats_all = lstruct['learn_feats_all']
        self.l_magspecs_all = lstruct['learn_magspecs_all']
        self.learn_files_all = lstruct['learn_files']

        # Loading the features for the signal
        [self.t_magspecs,
         self.t_feats,
         self.t_data] = load_data_one_audio_file(signal,
                                                self.params[
                                                'sr'],
                                                sigma_noise=0,
                                                wintime=self.params['wintime'],
                                                steptime=self.params['steptime'],
                                                max_frame_num_per_file=3000,
                                                startpoint=0,
                                                features=self.params['features'])

        # Retrieve the best candidates using the routine
        from feat_invert.regression import ann
        from feat_invert.transforms import gl_recons

        n_feats = self.t_feats.shape[1]

        estimated_spectrum, neighbors = ann(self.l_feats_all[:, 0:n_feats].T,
                                            self.l_magspecs_all.T,
                                            self.t_feats.T,
                                            self.t_magspecs.T,
                                            K=self.params['n_neighbs'])

        win_size = int(self.params['wintime'] * self.params['sr'])
        step_size = int(self.params['steptime'] * self.params['sr'])
        # sliding median filtering ?
        if self.params['l_filt'] > 1:
            from scipy.ndimage.filters import median_filter
            estimated_spectrum = median_filter(
                estimated_spectrum, (1, self.params['l_filt']))

        # reconstruction
        init_vec = np.random.randn(step_size * estimated_spectrum.shape[1])
        self.rep = Signal(gl_recons(estimated_spectrum, init_vec, 010,
                                    win_size, step_size, display=False),
                          self.params['sr'], normalize=True)

    def sparsify(self, sparsity):
        """ sparsity determines the number of used neighbors? """
        if sparsity <= 0:
            raise ValueError("Sparsity must be between 0 and 1 if a ratio or greater for a value")
        elif sparsity < 1:
            # interprete as a ratio
            sparsity *= self.rep.length

        # for now just calculate the number of needed coefficients
        self.sp_rep = self.rep
