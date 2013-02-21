'''
feat_invert.regression  -  Created on Feb 21, 2013
@author: M. Moussallam
'''
import numpy as np
import matplotlib.pyplot as plt

def innerprod_correl(X1,X2):
    
    (N1,T1) = X1.shape
    (N2,T2) = X2.shape    
    assert(N1==N2)
    correls = np.abs(np.dot(X1.T , X2)) 
    return correls

def corrcoeff_correl(X1,X2):
    """ correlation function based on the covariance matrix """
    
    (N1,T1) = X1.shape
    (N2,T2) = X2.shape    
    assert(N1==N2)
    E1 = np.sum(X1**2,0)
    E2 = np.sum(X2**2,0)
    
    # adding a little noise to any columns that is above threshold
    print "Warning: threshold parameter manually set"
    X1[:,E1<0.01] += 0.01*np.random.randn(*X1[:,E1<0.01].shape)
    X2[:,E2<0.01] += 0.01*np.random.randn(*X2[:,E2<0.01].shape)    
    
    correls = np.abs(np.corrcoef(np.concatenate((X1,X2),axis=1), rowvar=0))
    correls = correls[0:T1,T1:]
    
    return correls;
    
def load_correl(X1,X2):
    """ hack to load a preexisting matrix"""
    from scipy.io import loadmat
    Kdict = loadmat('/home/manu/workspace/toolboxes/Matlab_code/GriffinLim/kdev.mat');
    return Kdict['Kdev_test']

def nadaraya_watson(Xdev,Ydev,X,Y,covar,display=False,
                    K = 1, method='mean'):
    """ implementing regression with a Kernel defined by the covariance 
        handle given"""
        
    if not (Xdev.shape[0] == X.shape[0]):
        raise ValueError("Xdev and X should have same 1rst dimension") 
    
    N = Ydev.shape[0]
    Tdev = Xdev.shape[1]
    T = X.shape[1]
    
    # initalize output
    Y_hat = np.zeros((N,T))
    
    #Forward test    
    Ktest_dev = covar(X,Xdev);
    
    Ktest_dev[np.isnan(Ktest_dev)] = 0
    
    weights = Ktest_dev * np.sum(Ktest_dev,1)[...,None]

#    weights[np.isnan(weights)] = 0;

    order = np.argsort(weights,1)
    
#    plt.figure()
#    plt.imshow(Ktest_dev)
#    print Ktest_dev.dtype
#    plt.show()
    if method =='median':
        for t in range(T):
    #        print order[t,-K:]
            Y_hat[:,t] = np.median(Ydev[:,order[t,-K:]],1)
    elif method =='mean':
        for t in range(T):
    #        print order[t,-K:]
            Y_hat[:,t] = np.mean(Ydev[:,order[t,-K:]],1)
    
#    #Y_hat = (weights*Ydev.').';
#    E_forward = Y - Y_hat;
#    mean_forward_error = np.mean(20.0*np.log10(max(1E-15,abs(E_forward))./abs(Y)),2);
#    mean_forward = mean(mean_forward_error(:));
    
    
    if display:
        plt.figure(1)        
        plt.subplot(211)
        plt.imshow(np.log(np.abs(Y)),
                   interpolation='nearest',
                   origin='lower')
        plt.title('Reponse originale')
        plt.subplot(212)
        plt.imshow(np.log(np.abs(Y_hat)),
                   interpolation='nearest',
                   origin='lower')
        plt.title('Reponse predite')

    return Y_hat, Ktest_dev


