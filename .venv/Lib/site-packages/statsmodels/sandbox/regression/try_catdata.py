'''
Working with categorical data
=============================

use of dummy variables, group statistics, within and between statistics
examples for efficient matrix algebra

dummy versions require that the number of unique groups or categories is not too large
group statistics with scipy.ndimage can handle large number of observations and groups
scipy.ndimage stats is missing count

new: np.bincount can also be used for calculating values per label
'''
from statsmodels.compat.python import lrange
import numpy as np

from scipy import ndimage

#problem: ndimage does not allow axis argument,
#   calculates mean or var corresponding to axis=None in np.mean, np.var
#   useless for multivariate application

def labelmeanfilter(y, x):
    # requires integer labels
    # from mailing list scipy-user 2009-02-11
    labelsunique = np.arange(np.max(y)+1)
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    # returns label means for each original observation
    return labelmeans[y]

#groupcount: i.e. number of observation by group/label
#np.array(ndimage.histogram(yrvs[:,0],0,10,1,labels=yrvs[:,0],index=np.unique(yrvs[:,0])))

def labelmeanfilter_nd(y, x):
    # requires integer labels
    # from mailing list scipy-user 2009-02-11
    # adjusted for 2d x with column variables

    labelsunique = np.arange(np.max(y)+1)
    labmeansdata = []
    labmeans = []

    for xx in x.T:
        labelmeans = np.array(ndimage.mean(xx, labels=y, index=labelsunique))
        labmeansdata.append(labelmeans[y])
        labmeans.append(labelmeans)
    # group count:
    labelcount = np.array(ndimage.histogram(y, labelsunique[0], labelsunique[-1]+1,
                        1, labels=y, index=labelsunique))

    # returns array of lable/group counts and of label/group means
    #         and label/group means for each original observation
    return labelcount, np.array(labmeans), np.array(labmeansdata).T

def labelmeanfilter_str(ys, x):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    unil, unilinv = np.unique(ys, return_index=False, return_inverse=True)
    labelmeans = np.array(ndimage.mean(x, labels=unilinv, index=np.arange(np.max(unil)+1)))
    arr3 = labelmeans[unilinv]
    return arr3

def groupstatsbin(factors, values):
    '''uses np.bincount, assumes factors/labels are integers
    '''
    n = len(factors)
    ix,rind = np.unique(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values)/ (1.0*gcount)
    meanarr = gmean[rind]
    withinvar = np.bincount(rind, weights=(values-meanarr)**2) / (1.0*gcount)
    withinvararr = withinvar[rind]
    return gcount, gmean , meanarr, withinvar, withinvararr


def convertlabels(ys, indices=None):
    '''convert labels based on multiple variables or string labels to unique
    index labels 0,1,2,...,nk-1 where nk is the number of distinct labels
    '''
    if indices is None:
        ylabel = ys
    else:
        idx = np.array(indices)
        if idx.size > 1 and ys.ndim == 2:
            ylabel = np.array(['@%s@' % ii[:2].tostring() for ii in ys])[:,np.newaxis]
            #alternative
    ##        if ys[:,idx].dtype.kind == 'S':
    ##            ylabel = nd.array([' '.join(ii[:2]) for ii in ys])[:,np.newaxis]
        else:
            # there might be a problem here
            ylabel = ys

    unil, unilinv = np.unique(ylabel, return_index=False, return_inverse=True)
    return unilinv, np.arange(len(unil)), unil

def groupsstats_1d(y, x, labelsunique):
    '''use ndimage to get fast mean and variance'''
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    labelvars = np.array(ndimage.var(x, labels=y, index=labelsunique))
    return labelmeans, labelvars

def cat2dummy(y, nonseq=0):
    if nonseq or (y.ndim == 2 and y.shape[1] > 1):
        ycat, uniques, unitransl =  convertlabels(y, lrange(y.shape[1]))
    else:
        ycat = y.copy()
        ymin = y.min()
        uniques = np.arange(ymin,y.max()+1)
    if ycat.ndim == 1:
        ycat = ycat[:,np.newaxis]
    # this builds matrix nobs*ncat
    dummy = (ycat == uniques).astype(int)
    return dummy

def groupsstats_dummy(y, x, nonseq=0):
    if x.ndim == 1:
        # use groupsstats_1d
        x = x[:,np.newaxis]
    dummy = cat2dummy(y, nonseq=nonseq)
    countgr = dummy.sum(0, dtype=float)
    meangr = np.dot(x.T,dummy)/countgr
    meandata = np.dot(dummy,meangr.T) # category/group means as array in shape of x
    xdevmeangr = x - meandata  # deviation from category/group mean
    vargr = np.dot((xdevmeangr * xdevmeangr).T, dummy) / countgr
    return meangr, vargr, xdevmeangr, countgr
