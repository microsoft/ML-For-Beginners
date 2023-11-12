# -*- coding: utf-8 -*-
"""Principal Component Analysis


Created on Tue Sep 29 20:11:23 2009
Author: josef-pktd

TODO : add class for better reuse of results
"""

import numpy as np


def pca(data, keepdim=0, normalize=0, demean=True):
    '''principal components with eigenvector decomposition
    similar to princomp in matlab

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : int
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    normalize : bool
        if true, then eigenvectors are normalized by sqrt of eigenvalues
    demean : bool
        if true, then the column mean is subtracted from the data

    Returns
    -------
    xreduced : ndarray, 2d, (nobs, nvars)
        projection of the data x on the kept eigenvectors
    factors : ndarray, 2d, (nobs, nfactors)
        factor matrix, given by np.dot(x, evecs)
    evals : ndarray, 2d, (nobs, nfactors)
        eigenvalues
    evecs : ndarray, 2d, (nobs, nfactors)
        eigenvectors, normalized if normalize is true

    Notes
    -----

    See Also
    --------
    pcasvd : principal component analysis using svd

    '''
    x = np.array(data)
    #make copy so original does not change, maybe not necessary anymore
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

    # Covariance matrix
    xcov = np.cov(x, rowvar=0)

    # Compute eigenvalues and sort into descending order
    evals, evecs = np.linalg.eig(xcov)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]

    if keepdim > 0 and keepdim < x.shape[1]:
        evecs = evecs[:,:keepdim]
        evals = evals[:keepdim]

    if normalize:
        #for i in range(shape(evecs)[1]):
        #    evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
        evecs = evecs/np.sqrt(evals) #np.sqrt(np.dot(evecs.T, evecs) * evals)

    # get factor matrix
    #x = np.dot(evecs.T, x.T)
    factors = np.dot(x, evecs)
    # get original data from reduced number of components
    #xreduced = np.dot(evecs.T, factors) + m
    #print x.shape, factors.shape, evecs.shape, m.shape
    xreduced = np.dot(factors, evecs.T) + m
    return xreduced, factors, evals, evecs



def pcasvd(data, keepdim=0, demean=True):
    '''principal components with svd

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : int
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    demean : bool
        if true, then the column mean is subtracted from the data

    Returns
    -------
    xreduced : ndarray, 2d, (nobs, nvars)
        projection of the data x on the kept eigenvectors
    factors : ndarray, 2d, (nobs, nfactors)
        factor matrix, given by np.dot(x, evecs)
    evals : ndarray, 2d, (nobs, nfactors)
        eigenvalues
    evecs : ndarray, 2d, (nobs, nfactors)
        eigenvectors, normalized if normalize is true

    See Also
    --------
    pca : principal component analysis using eigenvector decomposition

    Notes
    -----
    This does not have yet the normalize option of pca.

    '''
    nobs, nvars = data.shape
    #print nobs, nvars, keepdim
    x = np.array(data)
    #make copy so original does not change
    if demean:
        m = x.mean(0)
    else:
        m = 0
##    if keepdim == 0:
##        keepdim = nvars
##        "print reassigning keepdim to max", keepdim
    x -= m
    U, s, v = np.linalg.svd(x.T, full_matrices=1)
    factors = np.dot(U.T, x.T).T #princomps
    if keepdim:
        xreduced = np.dot(factors[:,:keepdim], U[:,:keepdim].T) + m
    else:
        xreduced = data
        keepdim = nvars
        "print reassigning keepdim to max", keepdim

    # s = evals, U = evecs
    # no idea why denominator for s is with minus 1
    evals = s**2/(x.shape[0]-1)
    #print keepdim
    return xreduced, factors[:,:keepdim], evals[:keepdim], U[:,:keepdim] #, v


__all__ = ['pca', 'pcasvd']
