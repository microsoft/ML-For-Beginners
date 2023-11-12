''' convenience functions for ANOVA type analysis with OLS

Note: statistical results of ANOVA are not checked, OLS is
checked but not whether the reported results are the ones used
in ANOVA

'''

import numpy as np
import numpy.lib.recfunctions

from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS


dt_b = np.dtype([('breed', int), ('sex', int), ('litter', int),
               ('pen', int), ('pig', int), ('age', float),
               ('bage', float), ('y', float)])
''' too much work using structured masked arrays
dta = np.mafromtxt('dftest3.data', dtype=dt_b)

dta_use = np.ma.column_stack[[dta[col] for col in 'y sex age'.split()]]
'''


dta = np.genfromtxt('dftest3.data')
print(dta.shape)
mask = np.isnan(dta)
print("rows with missing values", mask.any(1).sum())
vars = dict((v[0], (idx, v[1])) for idx, v in enumerate((('breed', int),
                                                         ('sex', int),
                                                         ('litter', int),
                                                         ('pen', int),
                                                         ('pig', int),
                                                         ('age', float),
                                                         ('bage', float),
                                                         ('y', float))))

datavarnames = 'y sex age'.split()
#possible to avoid temporary array ?
dta_use = dta[:, [vars[col][0] for col in datavarnames]]
keeprows = ~np.isnan(dta_use).any(1)
print('number of complete observations', keeprows.sum())
dta_used = dta_use[keeprows,:]

varsused = dict((k, [dta_used[:,idx], idx, vars[k][1]]) for idx, k in enumerate(datavarnames))

# use function for dummy
#sexgroups = np.unique(dta_used[:,1])
#sexdummy = (dta_used[:,1][:, None] == sexgroups).astype(int)

def data2dummy(x, returnall=False):
    '''convert array of categories to dummy variables
    by default drops dummy variable for last category
    uses ravel, 1d only'''
    x = x.ravel()
    groups = np.unique(x)
    if returnall:
        return (x[:, None] == groups).astype(int)
    else:
        return (x[:, None] == groups).astype(int)[:,:-1]

def data2proddummy(x):
    '''creates product dummy variables from 2 columns of 2d array

    drops last dummy variable, but not from each category
    singular with simple dummy variable but not with constant

    quickly written, no safeguards

    '''
    #brute force, assumes x is 2d
    #replace with encoding if possible
    groups = np.unique(lmap(tuple, x.tolist()))
    #includes singularity with additive factors
    return (x==groups[:,None,:]).all(-1).T.astype(int)[:,:-1]

def data2groupcont(x1,x2):
    '''create dummy continuous variable

    Parameters
    ----------
    x1 : 1d array
        label or group array
    x2 : 1d array (float)
        continuous variable

    Notes
    -----
    useful for group specific slope coefficients in regression
    '''
    if x2.ndim == 1:
        x2 = x2[:,None]
    dummy = data2dummy(x1, returnall=True)
    return dummy * x2

sexdummy = data2dummy(dta_used[:,1])
factors = ['sex']
for k in factors:
    varsused[k][0] = data2dummy(varsused[k][0])

products = [('sex', 'age')]
for k in products:
    varsused[''.join(k)] = data2proddummy(np.c_[varsused[k[0]][0],varsused[k[1]][0]])

# make dictionary of variables with dummies as one variable
#vars_to_use = {name: data or dummy variables}

X_b0 = np.c_[sexdummy, dta_used[:,2], np.ones((dta_used.shape[0],1))]
y_b0 = dta_used[:,0]
res_b0 = OLS(y_b0, X_b0).results
print(res_b0.params)
print(res_b0.ssr)

anova_str0 = '''
ANOVA statistics (model sum of squares excludes constant)
Source    DF  Sum Squares   Mean Square    F Value    Pr > F
Model     %(df_model)i        %(ess)f       %(mse_model)f   %(fvalue)f %(f_pvalue)f
Error     %(df_resid)i     %(ssr)f       %(mse_resid)f
CTotal    %(nobs)i    %(uncentered_tss)f     %(mse_total)f

R squared  %(rsquared)f
'''

anova_str = '''
ANOVA statistics (model sum of squares includes constant)
Source    DF  Sum Squares   Mean Square    F Value    Pr > F
Model     %(df_model)i      %(ssmwithmean)f       %(mse_model)f   %(fvalue)f %(f_pvalue)f
Error     %(df_resid)i     %(ssr)f       %(mse_resid)f
CTotal    %(nobs)i    %(uncentered_tss)f     %(mse_total)f

R squared  %(rsquared)f
'''

#print(anova_str % dict([('df_model', res.df_model)])
#anovares = ['df_model' , 'df_resid'

def anovadict(res):
    '''update regression results dictionary with ANOVA specific statistics

    not checked for completeness
    '''
    ad = {}
    ad.update(res.__dict__)
    anova_attr = ['df_model', 'df_resid', 'ess', 'ssr','uncentered_tss',
                 'mse_model', 'mse_resid', 'mse_total', 'fvalue', 'f_pvalue',
                  'rsquared']
    for key in anova_attr:
        ad[key] = getattr(res, key)
    ad['nobs'] = res.model.nobs
    ad['ssmwithmean'] = res.uncentered_tss - res.ssr
    return ad


print(anova_str0 % anovadict(res_b0))
#the following leaves the constant in, not with NIST regression
#but something fishy with res.ess negative in examples
print(anova_str % anovadict(res_b0))

print('using sex only')
X2 = np.c_[sexdummy, np.ones((dta_used.shape[0],1))]
res2 = OLS(y_b0, X2).results
print(res2.params)
print(res2.ssr)
print(anova_str % anovadict(res2))

print('using age only')
X3 = np.c_[ dta_used[:,2], np.ones((dta_used.shape[0],1))]
res3 = OLS(y_b0, X3).results
print(res3.params)
print(res3.ssr)
print(anova_str % anovadict(res3))


def form2design(ss, data):
    '''convert string formula to data dictionary

    ss : str
     * I : add constant
     * varname : for simple varnames data is used as is
     * F:varname : create dummy variables for factor varname
     * P:varname1*varname2 : create product dummy variables for
       varnames
     * G:varname1*varname2 : create product between factor and
       continuous variable
    data : dict or structured array
       data set, access of variables by name as in dictionaries

    Returns
    -------
    vars : dictionary
        dictionary of variables with converted dummy variables
    names : list
        list of names, product (P:) and grouped continuous
        variables (G:) have name by joining individual names
        sorted according to input

    Examples
    --------
    >>> xx, n = form2design('I a F:b P:c*d G:c*f', testdata)
    >>> xx.keys()
    ['a', 'b', 'const', 'cf', 'cd']
    >>> n
    ['const', 'a', 'b', 'cd', 'cf']

    Notes
    -----

    with sorted dict, separate name list would not be necessary
    '''
    vars = {}
    names = []
    for item in ss.split():
        if item == 'I':
            vars['const'] = np.ones(data.shape[0])
            names.append('const')
        elif ':' not in item:
            vars[item] = data[item]
            names.append(item)
        elif item[:2] == 'F:':
            v = item.split(':')[1]
            vars[v] = data2dummy(data[v])
            names.append(v)
        elif item[:2] == 'P:':
            v = item.split(':')[1].split('*')
            vars[''.join(v)] = data2proddummy(np.c_[data[v[0]],data[v[1]]])
            names.append(''.join(v))
        elif item[:2] == 'G:':
            v = item.split(':')[1].split('*')
            vars[''.join(v)] = data2groupcont(data[v[0]], data[v[1]])
            names.append(''.join(v))
        else:
            raise ValueError('unknown expression in formula')
    return vars, names

nobs = 1000
testdataint = np.random.randint(3, size=(nobs,4)).view([('a',int),('b',int),('c',int),('d',int)])
testdatacont = np.random.normal( size=(nobs,2)).view([('e',float), ('f',float)])
dt2 = numpy.lib.recfunctions.zip_descr((testdataint, testdatacont),flatten=True)
# concatenate structured arrays
testdata = np.empty((nobs,1), dt2)
for name in testdataint.dtype.names:
    testdata[name] = testdataint[name]
for name in testdatacont.dtype.names:
    testdata[name] = testdatacont[name]


#print(form2design('a',testdata))

if 0:
    xx, n = form2design('F:a',testdata)
    print(xx)
    print(form2design('P:a*b',testdata))
    print(data2proddummy((np.c_[testdata['a'],testdata['b']])))

    xx, names = form2design('a F:b P:c*d',testdata)

#xx, names = form2design('I a F:b F:c F:d P:c*d',testdata)
xx, names = form2design('I a F:b P:c*d', testdata)
xx, names = form2design('I a F:b P:c*d G:a*e f', testdata)


X = np.column_stack([xx[nn] for nn in names])
# simple test version: all coefficients equal to one
y = X.sum(1) + 0.01*np.random.normal(size=(nobs))
rest1 = OLS(y,X).results
print(rest1.params)
print(anova_str % anovadict(rest1))

def dropname(ss, li):
    '''drop names from a list of strings,
    names to drop are in space delimited list
    does not change original list
    '''
    newli = li[:]
    for item in ss.split():
        newli.remove(item)
    return newli

X = np.column_stack([xx[nn] for nn in dropname('ae f', names)])
# simple test version: all coefficients equal to one
y = X.sum(1) + 0.01*np.random.normal(size=(nobs))
rest1 = OLS(y,X).results
print(rest1.params)
print(anova_str % anovadict(rest1))


# Example: from Bruce
# -------------------

# read data set and drop rows with missing data
dta = np.genfromtxt('dftest3.data', dt_b,missing='.', usemask=True)
print('missing', [dta.mask[k].sum() for k in dta.dtype.names])
m = dta.mask.view(bool)
droprows = m.reshape(-1,len(dta.dtype.names)).any(1)
# get complete data as plain structured array
# maybe does not work with masked arrays
dta_use_b1 = dta[~droprows,:].data
print(dta_use_b1.shape)
print(dta_use_b1.dtype)

#Example b1: variables from Bruce's glm

# prepare data and dummy variables
xx_b1, names_b1 = form2design('I F:sex age', dta_use_b1)
# create design matrix
X_b1 = np.column_stack([xx_b1[nn] for nn in dropname('', names_b1)])
y_b1 = dta_use_b1['y']
# estimate using OLS
rest_b1 = OLS(y_b1, X_b1).results
# print(results)
print(rest_b1.params)
print(anova_str % anovadict(rest_b1))
#compare with original version only in original version
print(anova_str % anovadict(res_b0))

# Example: use all variables except pig identifier

allexog = ' '.join(dta.dtype.names[:-1])
#'breed sex litter pen pig age bage'

xx_b1a, names_b1a = form2design('I F:breed F:sex F:litter F:pen age bage', dta_use_b1)
X_b1a = np.column_stack([xx_b1a[nn] for nn in dropname('', names_b1a)])
y_b1a = dta_use_b1['y']
rest_b1a = OLS(y_b1a, X_b1a).results
print(rest_b1a.params)
print(anova_str % anovadict(rest_b1a))

for dropn in names_b1a:
    print('\nResults dropping', dropn)
    X_b1a_ = np.column_stack([xx_b1a[nn] for nn in dropname(dropn, names_b1a)])
    y_b1a_ = dta_use_b1['y']
    rest_b1a_ = OLS(y_b1a_, X_b1a_).results
    #print(rest_b1a_.params
    print(anova_str % anovadict(rest_b1a_))
