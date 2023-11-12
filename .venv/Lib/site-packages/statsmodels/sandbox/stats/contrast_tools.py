'''functions to work with contrasts for multiple tests

contrast matrices for comparing all pairs, all levels to reference level, ...
extension to 2-way groups in progress

TwoWay: class for bringing two-way analysis together and try out
various helper functions


Idea for second part
- get all transformation matrices to move in between different full rank
  parameterizations
- standardize to one parameterization to get all interesting effects.

- multivariate normal distribution
  - exploit or expand what we have in LikelihoodResults, cov_params, f_test,
    t_test, example: resols_dropf_full.cov_params(C2)
  - connect to new multiple comparison for contrast matrices, based on
    multivariate normal or t distribution (Hothorn, Bretz, Westfall)

'''



from numpy.testing import assert_equal

import numpy as np

#next 3 functions copied from multicomp.py

def contrast_allpairs(nm):
    '''contrast or restriction matrix for all pairs of nm variables

    Parameters
    ----------
    nm : int

    Returns
    -------
    contr : ndarray, 2d, (nm*(nm-1)/2, nm)
       contrast matrix for all pairwise comparisons

    '''
    contr = []
    for i in range(nm):
        for j in range(i+1, nm):
            contr_row = np.zeros(nm)
            contr_row[i] = 1
            contr_row[j] = -1
            contr.append(contr_row)
    return np.array(contr)

def contrast_all_one(nm):
    '''contrast or restriction matrix for all against first comparison

    Parameters
    ----------
    nm : int

    Returns
    -------
    contr : ndarray, 2d, (nm-1, nm)
       contrast matrix for all against first comparisons

    '''
    contr = np.column_stack((np.ones(nm-1), -np.eye(nm-1)))
    return contr

def contrast_diff_mean(nm):
    '''contrast or restriction matrix for all against mean comparison

    Parameters
    ----------
    nm : int

    Returns
    -------
    contr : ndarray, 2d, (nm-1, nm)
       contrast matrix for all against mean comparisons

    '''
    return np.eye(nm) - np.ones((nm,nm))/nm

def signstr(x, noplus=False):
    if x in [-1,0,1]:
        if not noplus:
            return '+' if np.sign(x)>=0 else '-'
        else:
            return '' if np.sign(x)>=0 else '-'
    else:
        return str(x)


def contrast_labels(contrasts, names, reverse=False):
    if reverse:
        sl = slice(None, None, -1)
    else:
        sl = slice(None)
    labels = [''.join(['%s%s' % (signstr(c, noplus=True),v)
                          for c,v in zip(row, names)[sl] if c != 0])
                             for row in contrasts]
    return labels

def contrast_product(names1, names2, intgroup1=None, intgroup2=None, pairs=False):
    '''build contrast matrices for products of two categorical variables

    this is an experimental script and should be converted to a class

    Parameters
    ----------
    names1, names2 : lists of strings
        contains the list of level labels for each categorical variable
    intgroup1, intgroup2 : ndarrays     TODO: this part not tested, finished yet
        categorical variable


    Notes
    -----
    This creates a full rank matrix. It does not do all pairwise comparisons,
    parameterization is using contrast_all_one to get differences with first
    level.

    ? does contrast_all_pairs work as a plugin to get all pairs ?

    '''

    n1 = len(names1)
    n2 = len(names2)
    names_prod = ['%s_%s' % (i,j) for i in names1 for j in names2]
    ee1 = np.zeros((1,n1))
    ee1[0,0] = 1
    if not pairs:
        dd = np.r_[ee1, -contrast_all_one(n1)]
    else:
        dd = np.r_[ee1, -contrast_allpairs(n1)]

    contrast_prod = np.kron(dd[1:], np.eye(n2))
    names_contrast_prod0 = contrast_labels(contrast_prod, names_prod, reverse=True)
    names_contrast_prod = [''.join(['%s%s' % (signstr(c, noplus=True),v)
                              for c,v in zip(row, names_prod)[::-1] if c != 0])
                                 for row in contrast_prod]

    ee2 = np.zeros((1,n2))
    ee2[0,0] = 1
    #dd2 = np.r_[ee2, -contrast_all_one(n2)]
    if not pairs:
        dd2 = np.r_[ee2, -contrast_all_one(n2)]
    else:
        dd2 = np.r_[ee2, -contrast_allpairs(n2)]

    contrast_prod2 = np.kron(np.eye(n1), dd2[1:])
    names_contrast_prod2 = [''.join(['%s%s' % (signstr(c, noplus=True),v)
                              for c,v in zip(row, names_prod)[::-1] if c != 0])
                                 for row in contrast_prod2]

    if (intgroup1 is not None) and (intgroup1 is not None):
        d1, _ = dummy_1d(intgroup1)
        d2, _ = dummy_1d(intgroup2)
        dummy = dummy_product(d1, d2)
    else:
        dummy = None

    return (names_prod, contrast_prod, names_contrast_prod,
                        contrast_prod2, names_contrast_prod2, dummy)





def dummy_1d(x, varname=None):
    '''dummy variable for id integer groups

    Parameters
    ----------
    x : ndarray, 1d
        categorical variable, requires integers if varname is None
    varname : str
        name of the variable used in labels for category levels

    Returns
    -------
    dummy : ndarray, 2d
        array of dummy variables, one column for each level of the
        category (full set)
    labels : list[str]
        labels for the columns, i.e. levels of each category


    Notes
    -----
    use tools.categorical instead for more more options

    See Also
    --------
    statsmodels.tools.categorical

    Examples
    --------
    >>> x = np.array(['F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'],
          dtype='|S1')
    >>> dummy_1d(x, varname='gender')
    (array([[1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1]]), ['gender_F', 'gender_M'])

    '''
    if varname is None:  #assumes integer
        labels = ['level_%d' % i for i in range(x.max() + 1)]
        return (x[:,None]==np.arange(x.max()+1)).astype(int), labels
    else:
        grouplabels = np.unique(x)
        labels = [varname + '_%s' % str(i) for i in grouplabels]
        return (x[:,None]==grouplabels).astype(int), labels


def dummy_product(d1, d2, method='full'):
    '''dummy variable from product of two dummy variables

    Parameters
    ----------
    d1, d2 : ndarray
        two dummy variables, assumes full set for methods 'drop-last'
        and 'drop-first'
    method : {'full', 'drop-last', 'drop-first'}
        'full' returns the full product, encoding of intersection of
        categories.
        The drop methods provide a difference dummy encoding:
        (constant, main effects, interaction effects). The first or last columns
        of the dummy variable (i.e. levels) are dropped to get full rank
        dummy matrix.

    Returns
    -------
    dummy : ndarray
        dummy variable for product, see method

    '''

    if method == 'full':
        dd = (d1[:,:,None]*d2[:,None,:]).reshape(d1.shape[0],-1)
    elif method == 'drop-last':  #same as SAS transreg
        d12rl = dummy_product(d1[:,:-1], d2[:,:-1])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:,:-1], d2[:,:-1],d12rl))
        #Note: dtype int should preserve dtype of d1 and d2
    elif method == 'drop-first':
        d12r = dummy_product(d1[:,1:], d2[:,1:])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:,1:], d2[:,1:],d12r))
    else:
        raise ValueError('method not recognized')

    return dd

def dummy_limits(d):
    '''start and endpoints of groups in a sorted dummy variable array

    helper function for nested categories

    Examples
    --------
    >>> d1 = np.array([[1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> dummy_limits(d1)
    (array([0, 4, 8]), array([ 4,  8, 12]))

    get group slices from an array

    >>> [np.arange(d1.shape[0])[b:e] for b,e in zip(*dummy_limits(d1))]
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
    >>> [np.arange(d1.shape[0])[b:e] for b,e in zip(*dummy_limits(d1))]
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
    '''
    nobs, nvars = d.shape
    start1, col1 = np.nonzero(np.diff(d,axis=0)==1)
    end1, col1_ = np.nonzero(np.diff(d,axis=0)==-1)
    cc = np.arange(nvars)
    #print(cc, np.r_[[0], col1], np.r_[col1_, [nvars-1]]
    if ((not (np.r_[[0], col1] == cc).all())
            or (not (np.r_[col1_, [nvars-1]] == cc).all())):
        raise ValueError('dummy variable is not sorted')

    start = np.r_[[0], start1+1]
    end = np.r_[end1+1, [nobs]]
    return start, end



def dummy_nested(d1, d2, method='full'):
    '''unfinished and incomplete mainly copy past dummy_product
    dummy variable from product of two dummy variables

    Parameters
    ----------
    d1, d2 : ndarray
        two dummy variables, d2 is assumed to be nested in d1
        Assumes full set for methods 'drop-last' and 'drop-first'.
    method : {'full', 'drop-last', 'drop-first'}
        'full' returns the full product, which in this case is d2.
        The drop methods provide an effects encoding:
        (constant, main effects, subgroup effects). The first or last columns
        of the dummy variable (i.e. levels) are dropped to get full rank
        encoding.

    Returns
    -------
    dummy : ndarray
        dummy variable for product, see method

    '''
    if method == 'full':
        return d2

    start1, end1 = dummy_limits(d1)
    start2, end2 = dummy_limits(d2)
    first = np.in1d(start2, start1)
    last = np.in1d(end2, end1)
    equal = (first == last)
    col_dropf = ~first*~equal
    col_dropl = ~last*~equal


    if method == 'drop-last':
        d12rl = dummy_product(d1[:,:-1], d2[:,:-1])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:,:-1], d2[:,col_dropl]))
        #Note: dtype int should preserve dtype of d1 and d2
    elif method == 'drop-first':
        d12r = dummy_product(d1[:,1:], d2[:,1:])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:,1:], d2[:,col_dropf]))
    else:
        raise ValueError('method not recognized')

    return dd, col_dropf, col_dropl


class DummyTransform:
    '''Conversion between full rank dummy encodings


    y = X b + u
    b = C a
    a = C^{-1} b

    y = X C a + u

    define Z = X C, then

    y = Z a + u

    contrasts:

    R_b b = r

    R_a a = R_b C a = r

    where R_a = R_b C

    Here C is the transform matrix, with dot_left and dot_right as the main
    methods, and the same for the inverse transform matrix, C^{-1}

    Note:
     - The class was mainly written to keep left and right straight.
     - No checking is done.
     - not sure yet if method names make sense


    '''

    def __init__(self, d1, d2):
        '''C such that d1 C = d2, with d1 = X, d2 = Z

        should be (x, z) in arguments ?
        '''
        self.transf_matrix = np.linalg.lstsq(d1, d2, rcond=-1)[0]
        self.invtransf_matrix = np.linalg.lstsq(d2, d1, rcond=-1)[0]

    def dot_left(self, a):
        ''' b = C a
        '''
        return np.dot(self.transf_matrix, a)

    def dot_right(self, x):
        ''' z = x C
        '''
        return np.dot(x, self.transf_matrix)

    def inv_dot_left(self, b):
        ''' a = C^{-1} b
        '''
        return np.dot(self.invtransf_matrix, b)

    def inv_dot_right(self, z):
        ''' x = z C^{-1}
        '''
        return np.dot(z, self.invtransf_matrix)





def groupmean_d(x, d):
    '''groupmeans using dummy variables

    Parameters
    ----------
    x : array_like, ndim
        data array, tested for 1,2 and 3 dimensions
    d : ndarray, 1d
        dummy variable, needs to have the same length
        as x in axis 0.

    Returns
    -------
    groupmeans : ndarray, ndim-1
        means for each group along axis 0, the levels
        of the groups are the last axis

    Notes
    -----
    This will be memory intensive if there are many levels
    in the categorical variable, i.e. many columns in the
    dummy variable. In this case it is recommended to use
    a more efficient version.

    '''
    x = np.asarray(x)
##    if x.ndim == 1:
##        nvars = 1
##    else:
    nvars = x.ndim + 1
    sli = [slice(None)] + [None]*(nvars-2) + [slice(None)]
    return (x[...,None] * d[sli]).sum(0)*1./d.sum(0)



class TwoWay:
    '''a wrapper class for two way anova type of analysis with OLS


    currently mainly to bring things together

    Notes
    -----
    unclear: adding multiple test might assume block design or orthogonality

    This estimates the full dummy version with OLS.
    The drop first dummy representation can be recovered through the
    transform method.

    TODO: add more methods, tests, pairwise, multiple, marginal effects
    try out what can be added for userfriendly access.

    missing: ANOVA table

    '''
    def __init__(self, endog, factor1, factor2, varnames=None):
        self.nobs = factor1.shape[0]
        if varnames is None:
            vname1 = 'a'
            vname2 = 'b'
        else:
            vname1, vname1 = varnames

        self.d1, self.d1_labels = d1, d1_labels = dummy_1d(factor1, vname1)
        self.d2, self.d2_labels = d2, d2_labels = dummy_1d(factor2, vname2)
        self.nlevel1 = nlevel1 = d1.shape[1]
        self.nlevel2 = nlevel2 = d2.shape[1]


        #get product dummies
        res = contrast_product(d1_labels, d2_labels)
        prodlab, C1, C1lab, C2, C2lab, _ = res
        self.prod_label, self.C1, self.C1_label, self.C2, self.C2_label, _ = res
        dp_full = dummy_product(d1, d2, method='full')
        dp_dropf = dummy_product(d1, d2, method='drop-first')
        self.transform = DummyTransform(dp_full, dp_dropf)

        #estimate the model
        self.nvars = dp_full.shape[1]
        self.exog = dp_full
        self.resols = sm.OLS(endog, dp_full).fit()
        self.params = self.resols.params

        #get transformed parameters, (constant, main, interaction effect)
        self.params_dropf = self.transform.inv_dot_left(self.params)
        self.start_interaction = 1 + (nlevel1 - 1) + (nlevel2 - 1)
        self.n_interaction = self.nvars - self.start_interaction

    #convert to cached property
    def r_nointer(self):
        '''contrast/restriction matrix for no interaction
        '''
        nia = self.n_interaction
        R_nointer = np.hstack((np.zeros((nia, self.nvars-nia)), np.eye(nia)))
        #inter_direct = resols_full_dropf.tval[-nia:]
        R_nointer_transf = self.transform.inv_dot_right(R_nointer)
        self.R_nointer_transf = R_nointer_transf
        return R_nointer_transf

    def ttest_interaction(self):
        '''ttests for no-interaction terms are zero
        '''
        #use self.r_nointer instead
        nia = self.n_interaction
        R_nointer = np.hstack((np.zeros((nia, self.nvars-nia)), np.eye(nia)))
        #inter_direct = resols_full_dropf.tval[-nia:]
        R_nointer_transf = self.transform.inv_dot_right(R_nointer)
        self.R_nointer_transf = R_nointer_transf
        t_res = self.resols.t_test(R_nointer_transf)
        return t_res

    def ftest_interaction(self):
        '''ttests for no-interaction terms are zero
        '''
        R_nointer_transf = self.r_nointer()
        return self.resols.f_test(R_nointer_transf)

    def ttest_conditional_effect(self, factorind):
        if factorind == 1:
            return self.resols.t_test(self.C1), self.C1_label
        else:
            return self.resols.t_test(self.C2), self.C2_label

    def summary_coeff(self):
        from statsmodels.iolib import SimpleTable
        params_arr = self.params.reshape(self.nlevel1, self.nlevel2)
        stubs = self.d1_labels
        headers = self.d2_labels
        title = 'Estimated Coefficients by factors'
        table_fmt = dict(
            data_fmts = ["%#10.4g"]*self.nlevel2)
        return SimpleTable(params_arr, headers, stubs, title=title,
                           txt_fmt=table_fmt)


# --------------- tests
# TODO: several tests still missing, several are in the example with print

class TestContrastTools:

    def __init__(self):
        self.v1name = ['a0', 'a1', 'a2']
        self.v2name = ['b0', 'b1']
        self.d1 = np.array([[1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1]])

    def test_dummy_1d(self):
        x = np.array(['F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'],
              dtype='|S1')
        d, labels = (np.array([[1, 0],
                               [1, 0],
                               [0, 1],
                               [0, 1],
                               [1, 0],
                               [1, 0],
                               [0, 1],
                               [0, 1],
                               [1, 0],
                               [1, 0],
                               [0, 1],
                               [0, 1]]), ['gender_F', 'gender_M'])
        res_d, res_labels = dummy_1d(x, varname='gender')
        assert_equal(res_d, d)
        assert_equal(res_labels, labels)

    def test_contrast_product(self):
        res_cp = contrast_product(self.v1name, self.v2name)
        res_t = [0]*6
        res_t[0] = ['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1', 'a2_b0', 'a2_b1']
        res_t[1] = np.array([[-1.,  0.,  1.,  0.,  0.,  0.],
                           [ 0., -1.,  0.,  1.,  0.,  0.],
                           [-1.,  0.,  0.,  0.,  1.,  0.],
                           [ 0., -1.,  0.,  0.,  0.,  1.]])
        res_t[2] = ['a1_b0-a0_b0', 'a1_b1-a0_b1', 'a2_b0-a0_b0', 'a2_b1-a0_b1']
        res_t[3] =  np.array([[-1.,  1.,  0.,  0.,  0.,  0.],
                           [ 0.,  0., -1.,  1.,  0.,  0.],
                           [ 0.,  0.,  0.,  0., -1.,  1.]])
        res_t[4] = ['a0_b1-a0_b0', 'a1_b1-a1_b0', 'a2_b1-a2_b0']
        for ii in range(5):
            np.testing.assert_equal(res_cp[ii], res_t[ii], err_msg=str(ii))

    def test_dummy_limits(self):
        b,e = dummy_limits(self.d1)
        assert_equal(b, np.array([0, 4, 8]))
        assert_equal(e, np.array([ 4,  8, 12]))




if __name__ == '__main__':
    tt = TestContrastTools()
    tt.test_contrast_product()
    tt.test_dummy_1d()
    tt.test_dummy_limits()

    import statsmodels.api as sm

    examples = ['small', 'large', None][1]

    v1name = ['a0', 'a1', 'a2']
    v2name = ['b0', 'b1']
    res_cp = contrast_product(v1name, v2name)
    print(res_cp)

    y = np.arange(12)
    x1 = np.arange(12)//4
    x2 = np.arange(12)//2 % 2

    if 'small' in examples:
        d1, d1_labels = dummy_1d(x1)
        d2, d2_labels = dummy_1d(x2)


    if 'large' in examples:
        x1 = np.repeat(x1, 5, axis=0)
        x2 = np.repeat(x2, 5, axis=0)

    nobs = x1.shape[0]
    d1, d1_labels = dummy_1d(x1)
    d2, d2_labels = dummy_1d(x2)

    dd_full = dummy_product(d1, d2, method='full')
    dd_dropl = dummy_product(d1, d2, method='drop-last')
    dd_dropf = dummy_product(d1, d2, method='drop-first')

    #Note: full parameterization of dummies is orthogonal
    #np.eye(6)*10 in "large" example
    print((np.dot(dd_full.T, dd_full) == np.diag(dd_full.sum(0))).all())

    #check that transforms work
    #generate 3 data sets with the 3 different parameterizations

    effect_size = [1., 0.01][1]
    noise_scale = [0.001, 0.1][0]
    noise = noise_scale * np.random.randn(nobs)
    beta = effect_size * np.arange(1,7)
    ydata_full = (dd_full * beta).sum(1) + noise
    ydata_dropl = (dd_dropl * beta).sum(1) + noise
    ydata_dropf = (dd_dropf * beta).sum(1) + noise

    resols_full_full = sm.OLS(ydata_full, dd_full).fit()
    resols_full_dropf = sm.OLS(ydata_full, dd_dropf).fit()
    params_f_f = resols_full_full.params
    params_f_df = resols_full_dropf.params

    resols_dropf_full = sm.OLS(ydata_dropf, dd_full).fit()
    resols_dropf_dropf = sm.OLS(ydata_dropf, dd_dropf).fit()
    params_df_f = resols_dropf_full.params
    params_df_df = resols_dropf_dropf.params


    tr_of = np.linalg.lstsq(dd_dropf, dd_full, rcond=-1)[0]
    tr_fo = np.linalg.lstsq(dd_full, dd_dropf, rcond=-1)[0]
    print(np.dot(tr_fo, params_df_df) - params_df_f)
    print(np.dot(tr_of, params_f_f) - params_f_df)

    transf_f_df = DummyTransform(dd_full, dd_dropf)
    print(np.max(np.abs((dd_full - transf_f_df.inv_dot_right(dd_dropf)))))
    print(np.max(np.abs((dd_dropf - transf_f_df.dot_right(dd_full)))))
    print(np.max(np.abs((params_df_df
                         - transf_f_df.inv_dot_left(params_df_f)))))
    np.max(np.abs((params_f_df
                         - transf_f_df.inv_dot_left(params_f_f))))

    prodlab, C1, C1lab, C2, C2lab,_ = contrast_product(v1name, v2name)

    print('\ntvalues for no effect of factor 1')
    print('each test is conditional on a level of factor 2')
    print(C1lab)
    print(resols_dropf_full.t_test(C1).tvalue)

    print('\ntvalues for no effect of factor 2')
    print('each test is conditional on a level of factor 1')
    print(C2lab)
    print(resols_dropf_full.t_test(C2).tvalue)

    #covariance matrix of restrictions C2, note: orthogonal
    resols_dropf_full.cov_params(C2)

    #testing for no interaction effect
    R_noint = np.hstack((np.zeros((2,4)), np.eye(2)))
    inter_direct = resols_full_dropf.tvalues[-2:]
    inter_transf = resols_full_full.t_test(transf_f_df.inv_dot_right(R_noint)).tvalue
    print(np.max(np.abs((inter_direct - inter_transf))))

    #now with class version
    tw = TwoWay(ydata_dropf, x1, x2)
    print(tw.ttest_interaction().tvalue)
    print(tw.ttest_interaction().pvalue)
    print(tw.ftest_interaction().fvalue)
    print(tw.ftest_interaction().pvalue)
    print(tw.ttest_conditional_effect(1)[0].tvalue)
    print(tw.ttest_conditional_effect(2)[0].tvalue)
    print(tw.summary_coeff())

















''' documentation for early examples while developing - some have changed already
>>> y = np.arange(12)
>>> y
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> x1 = np.arange(12)//4
>>> x1
array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
>>> x2 = np.arange(12)//2%2
>>> x2
array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

>>> d1 = dummy_1d(x1)
>>> d1
array([[1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1]])

>>> d2 = dummy_1d(x2)
>>> d2
array([[1, 0],
       [1, 0],
       [0, 1],
       [0, 1],
       [1, 0],
       [1, 0],
       [0, 1],
       [0, 1],
       [1, 0],
       [1, 0],
       [0, 1],
       [0, 1]])

>>> d12 = dummy_product(d1, d2)
>>> d12
array([[1, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 1]])


>>> d12rl = dummy_product(d1[:,:-1], d2[:,:-1])
>>> np.column_stack((np.ones(d1.shape[0]), d1[:,:-1], d2[:,:-1],d12rl))
array([[ 1.,  1.,  0.,  1.,  1.,  0.],
       [ 1.,  1.,  0.,  1.,  1.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  1.,  1.,  0.,  1.],
       [ 1.,  0.,  1.,  1.,  0.,  1.],
       [ 1.,  0.,  1.,  0.,  0.,  0.],
       [ 1.,  0.,  1.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.]])
'''




#nprod = ['%s_%s' % (i,j) for i in ['a0', 'a1', 'a2'] for j in ['b0', 'b1']]
#>>> [''.join(['%s%s' % (signstr(c),v) for c,v in zip(row, nprod) if c != 0])
#     for row in np.kron(dd[1:], np.eye(2))]



'''
>>> nprod = ['%s_%s' % (i,j) for i in ['a0', 'a1', 'a2'] for j in ['b0', 'b1']]
>>> nprod
['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1', 'a2_b0', 'a2_b1']
>>> [''.join(['%s%s' % (signstr(c),v) for c,v in zip(row, nprod) if c != 0]) for row in np.kron(dd[1:], np.eye(2))]
['-a0b0+a1b0', '-a0b1+a1b1', '-a0b0+a2b0', '-a0b1+a2b1']
>>> [''.join(['%s%s' % (signstr(c),v) for c,v in zip(row, nprod)[::-1] if c != 0]) for row in np.kron(dd[1:], np.eye(2))]
['+a1_b0-a0_b0', '+a1_b1-a0_b1', '+a2_b0-a0_b0', '+a2_b1-a0_b1']

>>> np.r_[[[1,0,0,0,0]],contrast_all_one(5)]
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 1., -1.,  0.,  0.,  0.],
       [ 1.,  0., -1.,  0.,  0.],
       [ 1.,  0.,  0., -1.,  0.],
       [ 1.,  0.,  0.,  0., -1.]])

>>> idxprod = [(i,j) for i in range(3) for j in range(2)]
>>> idxprod
[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
>>> np.array(idxprod).reshape(2,3,2,order='F')[:,:,0]
array([[0, 1, 2],
       [0, 1, 2]])
>>> np.array(idxprod).reshape(2,3,2,order='F')[:,:,1]
array([[0, 0, 0],
       [1, 1, 1]])
>>> dd3_ = np.r_[[[0,0,0]],contrast_all_one(3)]



pairwise contrasts and reparameterization

dd = np.r_[[[1,0,0,0,0]],-contrast_all_one(5)]
>>> dd
array([[ 1.,  0.,  0.,  0.,  0.],
       [-1.,  1.,  0.,  0.,  0.],
       [-1.,  0.,  1.,  0.,  0.],
       [-1.,  0.,  0.,  1.,  0.],
       [-1.,  0.,  0.,  0.,  1.]])
>>> np.dot(dd.T, np.arange(5))
array([-10.,   1.,   2.,   3.,   4.])
>>> np.round(np.linalg.inv(dd.T)).astype(int)
array([[1, 1, 1, 1, 1],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1]])
>>> np.round(np.linalg.inv(dd)).astype(int)
array([[1, 0, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [1, 0, 1, 0, 0],
       [1, 0, 0, 1, 0],
       [1, 0, 0, 0, 1]])
>>> dd
array([[ 1.,  0.,  0.,  0.,  0.],
       [-1.,  1.,  0.,  0.,  0.],
       [-1.,  0.,  1.,  0.,  0.],
       [-1.,  0.,  0.,  1.,  0.],
       [-1.,  0.,  0.,  0.,  1.]])
>>> ddinv=np.round(np.linalg.inv(dd.T)).astype(int)
>>> np.dot(ddinv, np.arange(5))
array([10,  1,  2,  3,  4])
>>> np.dot(dd, np.arange(5))
array([ 0.,  1.,  2.,  3.,  4.])
>>> np.dot(dd, 5+np.arange(5))
array([ 5.,  1.,  2.,  3.,  4.])
>>> ddinv2 = np.round(np.linalg.inv(dd)).astype(int)
>>> np.dot(ddinv2, np.arange(5))
array([0, 1, 2, 3, 4])
>>> np.dot(ddinv2, 5+np.arange(5))
array([ 5, 11, 12, 13, 14])
>>> np.dot(ddinv2, [5, 0, 0 , 1, 2])
array([5, 5, 5, 6, 7])
>>> np.dot(ddinv2, dd)
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])



>>> dd3 = -np.r_[[[1,0,0]],contrast_all_one(3)]
>>> dd2 = -np.r_[[[1,0]],contrast_all_one(2)]
>>> np.kron(np.eye(3), dd2)
array([[-1.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0., -1.,  1.]])
>>> dd2
array([[-1.,  0.],
       [-1.,  1.]])
>>> np.kron(np.eye(3), dd2[1:])
array([[-1.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0., -1.,  1.]])
>>> np.kron(dd[1:], np.eye(2))
array([[-1.,  0.,  1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  1.,  0.,  0.],
       [-1.,  0.,  0.,  0.,  1.,  0.],
       [ 0., -1.,  0.,  0.,  0.,  1.]])



d_ = np.r_[[[1,0,0,0,0]],contrast_all_one(5)]
>>> d_
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 1., -1.,  0.,  0.,  0.],
       [ 1.,  0., -1.,  0.,  0.],
       [ 1.,  0.,  0., -1.,  0.],
       [ 1.,  0.,  0.,  0., -1.]])
>>> np.round(np.linalg.pinv(d_)).astype(int)
array([[ 1,  0,  0,  0,  0],
       [ 1, -1,  0,  0,  0],
       [ 1,  0, -1,  0,  0],
       [ 1,  0,  0, -1,  0],
       [ 1,  0,  0,  0, -1]])
>>> np.linalg.inv(d_).astype(int)
array([[ 1,  0,  0,  0,  0],
       [ 1, -1,  0,  0,  0],
       [ 1,  0, -1,  0,  0],
       [ 1,  0,  0, -1,  0],
       [ 1,  0,  0,  0, -1]])


group means

>>> sli = [slice(None)] + [None]*(3-2) + [slice(None)]
>>> (np.column_stack((y, x1, x2))[...,None] * d1[sli]).sum(0)*1./d1.sum(0)
array([[ 1.5,  5.5,  9.5],
       [ 0. ,  1. ,  2. ],
       [ 0.5,  0.5,  0.5]])

>>> [(z[:,None] * d1).sum(0)*1./d1.sum(0) for z in np.column_stack((y, x1, x2)).T]
[array([ 1.5,  5.5,  9.5]), array([ 0.,  1.,  2.]), array([ 0.5,  0.5,  0.5])]
>>>

'''
