'''VAR and VARMA process

this does not actually do much, trying out a version for a time loop

alternative representation:
* textbook, different blocks in matrices
* Kalman filter
* VAR, VARX and ARX could be calculated with signal.lfilter
  only tried some examples, not implemented

TODO: try minimizing sum of squares of (Y-Yhat)

Note: filter has smallest lag at end of array and largest lag at beginning,
    be careful for asymmetric lags coefficients
    check this again if it is consistently used


changes
2009-09-08 : separated from movstat.py

Author : josefpkt
License : BSD
'''

import numpy as np
from scipy import signal


#NOTE: this just returns that predicted values given the
#B matrix in polynomial form.
#TODO: make sure VAR class returns B/params in this form.
def VAR(x,B, const=0):
    ''' multivariate linear filter

    Parameters
    ----------
    x: (TxK) array
        columns are variables, rows are observations for time period
    B: (PxKxK) array
        b_t-1 is bottom "row", b_t-P is top "row" when printing
        B(:,:,0) is lag polynomial matrix for variable 1
        B(:,:,k) is lag polynomial matrix for variable k
        B(p,:,k) is pth lag for variable k
        B[p,:,:].T corresponds to A_p in Wikipedia
    const : float or array (not tested)
        constant added to autoregression

    Returns
    -------
    xhat: (TxK) array
        filtered, predicted values of x array

    Notes
    -----
    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) }  for all i = 0,K-1, for all t=p..T

    xhat does not include the forecasting observation, xhat(T+1),
    xhat is 1 row shorter than signal.correlate

    References
    ----------
    https://en.wikipedia.org/wiki/Vector_Autoregression
    https://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)
    '''
    p = B.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    for t in range(p,T): #[p+2]:#
##        print(p,T)
##        print(x[t-p:t,:,np.newaxis].shape)
##        print(B.shape)
        #print(x[t-p:t,:,np.newaxis])
        xhat[t,:] = const + (x[t-p:t,:,np.newaxis]*B).sum(axis=1).sum(axis=0)
    return xhat


def VARMA(x,B,C, const=0):
    ''' multivariate linear filter

    x (TxK)
    B (PxKxK)

    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) } +
                sum{_q}sum{_k} { e(t-Q:t,:) .* C(:,:,i) }for all i = 0,K-1

    '''
    P = B.shape[0]
    Q = C.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    e = np.zeros(x.shape)
    start = max(P,Q)
    for t in range(start,T): #[p+2]:#
##        print(p,T
##        print(x[t-p:t,:,np.newaxis].shape
##        print(B.shape
        #print(x[t-p:t,:,np.newaxis]
        xhat[t,:] =  const + (x[t-P:t,:,np.newaxis]*B).sum(axis=1).sum(axis=0) + \
                     (e[t-Q:t,:,np.newaxis]*C).sum(axis=1).sum(axis=0)
        e[t,:] = x[t,:] - xhat[t,:]
    return xhat, e


if __name__ == '__main__':


    T = 20
    K = 2
    P = 3
    #x = np.arange(10).reshape(5,2)
    x = np.column_stack([np.arange(T)]*K)
    B = np.ones((P,K,K))
    #B[:,:,1] = 2
    B[:,:,1] = [[0,0],[0,0],[0,1]]
    xhat = VAR(x,B)
    print(np.all(xhat[P:,0]==np.correlate(x[:-1,0],np.ones(P))*2))
    #print(xhat)


    T = 20
    K = 2
    Q = 2
    P = 3
    const = 1
    #x = np.arange(10).reshape(5,2)
    x = np.column_stack([np.arange(T)]*K)
    B = np.ones((P,K,K))
    #B[:,:,1] = 2
    B[:,:,1] = [[0,0],[0,0],[0,1]]
    C = np.zeros((Q,K,K))
    xhat1 = VAR(x,B, const=const)
    xhat2, err2 = VARMA(x,B,C, const=const)
    print(np.all(xhat2 == xhat1))
    print(np.all(xhat2[P:,0] == np.correlate(x[:-1,0],np.ones(P))*2+const))

    C[1,1,1] = 0.5
    xhat3, err3 = VARMA(x,B,C)

    x = np.r_[np.zeros((P,K)),x]  #prepend initial conditions
    xhat4, err4 = VARMA(x,B,C)

    C[1,1,1] = 1
    B[:,:,1] = [[0,0],[0,0],[0,1]]
    xhat5, err5 = VARMA(x,B,C)
    #print(err5)

    #in differences
    #VARMA(np.diff(x,axis=0),B,C)


    #Note:
    # * signal correlate applies same filter to all columns if kernel.shape[1]<K
    #   e.g. signal.correlate(x0,np.ones((3,1)),'valid')
    # * if kernel.shape[1]==K, then `valid` produces a single column
    #   -> possible to run signal.correlate K times with different filters,
    #      see the following example, which replicates VAR filter
    x0 = np.column_stack([np.arange(T), 2*np.arange(T)])
    B[:,:,0] = np.ones((P,K))
    B[:,:,1] = np.ones((P,K))
    B[1,1,1] = 0
    xhat0 = VAR(x0,B)
    xcorr00 = signal.correlate(x0,B[:,:,0])#[:,0]
    xcorr01 = signal.correlate(x0,B[:,:,1])
    print(np.all(signal.correlate(x0,B[:,:,0],'valid')[:-1,0]==xhat0[P:,0]))
    print(np.all(signal.correlate(x0,B[:,:,1],'valid')[:-1,0]==xhat0[P:,1]))

    #import error
    #from movstat import acovf, acf
    from statsmodels.tsa.stattools import acovf, acf
    aav = acovf(x[:,0])
    print(aav[0] == np.var(x[:,0]))
    aac = acf(x[:,0])
