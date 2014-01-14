'''
speech.wavelets_toy  -  Created on Nov 12, 2013

sandbox to visualize 2D wavelet transform of spectrograms in pyramids

@author: M. Moussallam
'''
import sys, os
from classes.sketches.cochleo import SKETCH_ROOT
sys.path.append(os.environ['SKETCH_ROOT'])
from src.settingup import *
SND_DB_PATH = os.environ['SND_DB_PATH']

import pywt
import pylab as pl
test_file = op.join(SND_DB_PATH, 'sqam/glocs.wav')

lev = 4

def stack(a,clist, level):
#    print level
    if level == 0:
        return a
    ch,cv,cd = clist[level-1]
#    print ch.shape, cv.shape, cd.shape
    sub = stack(a, clist[:-1], level-1)
    print sub.shape, ch.shape
    mat = np.hstack([np.vstack([sub, np.abs(ch)]),
                 np.vstack([np.abs(cv), np.abs(cd)])])
    return mat

def get_wavedec(filename):
    sig = Signal(filename, mono=True)
    sig.downsample(16000)
    spectro = stft(sig.data,512,64) 
    
#    img = sig.spectrogram(512,64,order=0.25,log=False,cbar=False)
#    pl.close()
    img = spectro[0,:,:]
    print img.shape[1]
    if not (img.shape[0]%2 == 0):
        img = img[:-1,:]
    if (img.shape[1]>512):
        print "ok"
        img = img[:,:512]
    print img.shape
    # now compute the wavelet dec
    return pywt.wavedec2(np.abs(img), 'haar','sym',level=lev), spectro

def lp_filter(coeffs):
    """ put everything at zero except lowest frequencies """
    filt_coeffs = []
    filt_coeffs.append(coeffs[0])
    for hf in coeffs[1:]:
        ch,cv,cd = hf
        filt_coeffs.append((np.zeros_like(ch),
                            np.zeros_like(cv),
                            np.zeros_like(cd)))
    return filt_coeffs

def reconstruct(spectro1, rec1):
    torec = spectro1
    torec[0, :rec1.shape[0], :rec1.shape[1]] /= np.abs(torec[0, :rec1.shape[0], :rec1.shape[1]])
    torec[0, :rec1.shape[0], :rec1.shape[1]] *= rec1
    rec_wav = istft(torec, 64)
    return rec_wav



#wtdecs = get_wavedec(test_file)
#mat = stack(wtdecs[0],wtdecs[1:], level=lev)
#pl.imshow(mat, interpolation='nearest',cmap=pl.cm.RdGy)
#pl.show()


# ok now take same sentences from two speaker
index = 10
spk1path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_ksp_arctic')
spk1filename = get_filepaths(spk1path, 0,  ext='wav')[index]
spk2path  = op.join(SND_DB_PATH,'voxforge/main/Learn/cmu_us_jmk_arctic')
spk2filename = get_filepaths(spk2path, 0,  ext='wav')[index]


wtdecs1, spectro1 = get_wavedec(spk1filename)
mat1 = stack(wtdecs1[0],wtdecs1[1:], level=lev)
wtdecs2, spectro2 = get_wavedec(spk2filename)
mat2 = stack(wtdecs2[0],wtdecs2[1:], level=lev)
pl.figure()
plt.subplot(121)
pl.imshow(mat1, interpolation='nearest',cmap=pl.cm.RdGy)
plt.subplot(122)
pl.imshow(mat2, interpolation='nearest',cmap=pl.cm.RdGy)
#pl.show()  

# reconstruction while putting all high-F to zero
rec1 = pywt.waverec2(lp_filter(wtdecs1), 'haar', 'sym')
rec2 = pywt.waverec2(lp_filter(wtdecs2), 'haar', 'sym')


sig_rec1 = Signal(reconstruct(spectro1, rec1), 16000, mono=True, normalize=True)
sig_rec2 = Signal(reconstruct(spectro2, rec2), 16000, mono=True, normalize=True)
sig_orig = Signal(spk1filename, normalize=True)

pl.figure()
plt.subplot(121)
pl.imshow(rec1, interpolation='nearest',cmap=pl.cm.RdGy)
plt.subplot(122)
pl.imshow(np.abs(torec[0,:,:]), interpolation='nearest',cmap=pl.cm.RdGy)
pl.show()
#from pywt import WaveletPacket2D
#
#wp2 = WaveletPacket2D(img, 'db2', 'sym', maxlevel=2)
#
#pl.imshow(img, interpolation="nearest", cmap=pl.cm.gray)
#
#path = ['d', 'v', 'h', 'a']
#
##mod = lambda x: x
##mod = lambda x: abs(x)
#mod = lambda x: np.sqrt(abs(x))
#
#pl.figure()
#for i, p2 in enumerate(path):
#    pl.subplot(2, 2, i + 1)
#    p1p2 = p2
#    pl.imshow(mod(wp2[p1p2].data), origin='image', interpolation="nearest",
#        cmap=pl.cm.gray)
#    pl.title(p1p2)
#
#for p1 in path:
#    pl.figure()
#    for i, p2 in enumerate(path):
#        pl.subplot(2, 2, i + 1)
#        p1p2 = p1 + p2
#        pl.imshow(mod(wp2[p1p2].data), origin='image',
#            interpolation="nearest", cmap=pl.cm.gray)
#        pl.title(p1p2)
#
#pl.figure()
#i = 1
#for row in wp2.get_level(2, 'freq'):
#    for node in row:
#        pl.subplot(len(row), len(row), i)
#        pl.title("%s=(%s row, %s col)" % (
#        (node.path,) + wp2.expand_2d_path(node.path)))
#        pl.imshow(mod(node.data), origin='image', interpolation="nearest",
#            cmap=pl.cm.gray)
#        i += 1
#
#pl.show()