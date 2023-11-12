'''conditional logit and nested conditional logit

nested conditional logit is supposed to be the random utility version
(RU2 and maybe RU1)

References:
-----------
currently based on:
Greene, Econometric Analysis, 5th edition and draft (?)
Hess, Florian, 2002, Structural Choice analysis with nested logit models,
    The Stats Journal 2(3) pp 227-252

not yet used:
Silberhorn Nadja, Yasemin Boztug, Lutz Hildebrandt, 2008, Estimation with the
    nested logit model: specifications and software particularities,
    OR Spectrum
Koppelman, Frank S., and Chandra Bhat with technical support from Vaneet Sethi,
    Sriram Subramanian, Vincent Bernardin and Jian Zhang, 2006,
    A Self Instructing Course in Mode Choice Modeling: Multinomial and
    Nested Logit Models

Author: josef-pktd
License: BSD (simplified)
'''
import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize


class TryCLogit:
    '''
    Conditional Logit, data handling test

    Parameters
    ----------

    endog : array (nobs,nchoices)
        dummy encoding of realized choices
    exog_bychoices : list of arrays
        explanatory variables, one array of exog for each choice. Variables
        with common coefficients have to be first in each array
    ncommon : int
        number of explanatory variables with common coefficients

    Notes
    -----

    Utility for choice j is given by

        $V_j = X_j * beta + Z * gamma_j$

    where X_j contains generic variables (terminology Hess) that have the same
    coefficient across choices, and Z are variables, like individual-specific
    variables that have different coefficients across variables.

    If there are choice specific constants, then they should be contained in Z.
    For identification, the constant of one choice should be dropped.


    '''

    def __init__(self, endog, exog_bychoices, ncommon):
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        self.nobs, self.nchoices = endog.shape
        self.nchoices = len(exog_bychoices)

        #TODO: rename beta to params and include inclusive values for nested CL
        betaind = [exog_bychoices[ii].shape[1]-ncommon for ii in range(4)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        beta_indices = [np.r_[np.array([0, 1]),z[zi[ii]:zi[ii+1]]]
                       for ii in range(len(zi)-1)]
        self.beta_indices = beta_indices

        #for testing only
        beta = np.arange(7)
        betaidx_bychoices = [beta[idx] for idx in beta_indices]


    def xbetas(self, params):
        '''these are the V_i
        '''

        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:,choiceind] = np.dot(self.exog_bychoices[choiceind],
                                      params[self.beta_indices[choiceind]])
        return res

    def loglike(self, params):
        #normalization ?
        xb = self.xbetas(params)
        expxb = np.exp(xb)
        sumexpxb = expxb.sum(1)#[:,None]
        probs = expxb/expxb.sum(1)[:,None]  #we do not really need this for all
        loglike = (self.endog * np.log(probs)).sum(1)
        #is this the same: YES
        #self.logliketest = (self.endog * xb).sum(1) - np.log(sumexpxb)
        #if self.endog where index then xb[self.endog]
        return -loglike.sum()   #return sum for now not for each observation

    def fit(self, start_params=None):
        if start_params is None:
            start_params = np.zeros(6)  # need better np.zeros(6)
        return optimize.fmin(self.loglike, start_params, maxfun=10000)


class TryNCLogit:
    '''
    Nested Conditional Logit (RUNMNL), data handling test

    unfinished, does not do anything yet

    '''

    def __init__(self, endog, exog_bychoices, ncommon):
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        self.nobs, self.nchoices = endog.shape
        self.nchoices = len(exog_bychoices)


        #TODO rename beta to params and include inclusive values for nested CL
        betaind = [exog_bychoices[ii].shape[1]-ncommon for ii in range(4)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        beta_indices = [np.r_[np.array([0, 1]),z[zi[ii]:zi[ii+1]]]
                       for ii in range(len(zi)-1)]
        self.beta_indices = beta_indices

        #for testing only
        beta = np.arange(7)
        betaidx_bychoices = [beta[idx] for idx in beta_indices]


    def xbetas(self, params):
        '''these are the V_i
        '''

        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:,choiceind] = np.dot(self.exog_bychoices[choiceind],
                                      params[self.beta_indices[choiceind]])
        return res

    def loglike_leafbranch(self, params, tau):
        #normalization ?
        #check/change naming for tau
        xb = self.xbetas(params)
        expxb = np.exp(xb/tau)
        sumexpxb = expxb.sum(1)#[:,None]
        logsumexpxb = np.log(sumexpxb)
        #loglike = (self.endog * xb).sum(1) - logsumexpxb
        probs = expxb/sumexpxb[:,None]
        return probs, logsumexpxp  # noqa:F821  See GH#5756
        #if self.endog where index then xb[self.endog]
        #return -loglike.sum()   #return sum for now not for each observation

    def loglike_branch(self, params, tau):
        #not yet sure how to keep track of branches during walking of tree
        ivs = []
        for b in branches:  # noqa:F821  See GH#5756
            probs, iv = self.loglike_leafbranch(params, tau)
            ivs.append(iv)

        #ivs = np.array(ivs)   #note ivs is (nobs,nbranchchoices)
        ivs = np.column_stack(ivs) # this way ?
        exptiv = np.exp(tau*ivs)
        sumexptiv = exptiv.sum(1)
        logsumexpxb = np.log(sumexpxb)  # noqa:F821  See GH#5756
        probs = exptiv/sumexptiv[:,None]


####### obsolete version to try out attaching data,
####### new in treewalkerclass.py, copy new version to replace this
####### problem with bzr I will disconnect history when copying
testxb = 0 #global to class
class RU2NMNL:
    '''Nested Multinomial Logit with Random Utility 2 parameterization

    '''

    def __init__(self, endog, exog, tree, paramsind):
        self.endog = endog
        self.datadict = exog
        self.tree = tree
        self.paramsind = paramsind

        self.branchsum = ''
        self.probs = {}


    def calc_prob(self, tree, keys=None):
        '''walking a tree bottom-up based on dictionary
        '''
        endog = self.endog
        datadict = self.datadict
        paramsind = self.paramsind
        branchsum = self.branchsum


        if isinstance(tree, tuple):   #assumes leaves are int for choice index
            name, subtree = tree
            print(name, datadict[name])
            print('subtree', subtree)
            keys = []
            if testxb:
                branchsum = datadict[name]
            else:
                branchsum = name  #0
            for b in subtree:
                print(b)
                #branchsum += branch2(b)
                branchsum = branchsum + self.calc_prob(b, keys)
            print('branchsum', branchsum, keys)
            for k in keys:
                self.probs[k] = self.probs[k] + ['*' + name + '-prob']

        else:
            keys.append(tree)
            self.probs[tree] = [tree + '-prob' +
                                '(%s)' % ', '.join(self.paramsind[tree])]
            if testxb:
                leavessum = sum((datadict[bi] for bi in tree))
                print('final branch with', tree, ''.join(tree), leavessum) #sum(tree)
                return leavessum  #sum(xb[tree])
            else:
                return ''.join(tree) #sum(tree)

        print('working on branch', tree, branchsum)
        return branchsum



#Trying out ways to handle data
#------------------------------

#travel data from Greene
dta = np.genfromtxt('TableF23-2.txt', skip_header=1,
                    names='Mode   Ttme   Invc    Invt      GC     Hinc    PSize'.split())

endog = dta['Mode'].reshape(-1,4).copy() #I do not want a view
nobs, nchoices = endog.shape
datafloat = dta.view(float).reshape(-1,7)
exog = datafloat[:,1:].reshape(-1,6*nchoices).copy() #I do not want a view

print(endog.sum(0))
varnames = dta.dtype.names
print(varnames[1:])
modes = ['Air', 'Train', 'Bus', 'Car']
print(exog.mean(0).reshape(nchoices, -1)) # Greene Table 23.23




#try dummy encoding for individual-specific variables
exog_choice_names = ['GC', 'Ttme']
exog_choice = np.column_stack([dta[name] for name in exog_choice_names])
exog_choice = exog_choice.reshape(-1,len(exog_choice_names)*nchoices)
exog_choice = np.c_[endog, exog_choice] # add constant dummy

exog_individual = dta['Hinc'][:,None]

#exog2 = np.c_[exog_choice, exog_individual*endog]

# we can also overwrite and select in original datafloat
# e.g. Hinc*endog{choice)

choice_index = np.arange(dta.shape[0]) % nchoices
hinca = dta['Hinc']*(choice_index==0)
dta2=recf.append_fields(dta, ['Hinca'],[hinca], usemask=False)


#another version

xi = []
for ii in range(4):
    xi.append(datafloat[choice_index==ii])

#one more
dta1 = recf.append_fields(dta, ['Const'],[np.ones(dta.shape[0])], usemask=False)

xivar = [['GC', 'Ttme', 'Const', 'Hinc'],
         ['GC', 'Ttme', 'Const'],
         ['GC', 'Ttme', 'Const'],
         ['GC', 'Ttme']]    #need to drop one constant

xi = []
for ii in range(4):
    xi.append(dta1[xivar[ii]][choice_index==ii])
    #this does not change sequence of columns, bug report by Skipper I think

ncommon = 2
betaind = [len(xi[ii].dtype.names)-ncommon for ii in range(4)]
zi=np.r_[[ncommon], ncommon+np.array(betaind).cumsum()]
z=np.arange(7)  #what is n?
betaindices = [np.r_[np.array([0, 1]),z[zi[ii]:zi[ii+1]]]
               for ii in range(len(zi)-1)]

beta = np.arange(7)
betai = [beta[idx] for idx in betaindices]




#examples for TryCLogit
#----------------------


#get exogs as float
xifloat = [xx.view(float).reshape(nobs,-1) for xx in xi]
clogit = TryCLogit(endog, xifloat, 2)

debug = 0
if debug:
    res = optimize.fmin(clogit.loglike, np.ones(6))
#estimated parameters from Greene:
tab2324 = [-0.15501, -0.09612, 0.01329, 5.2074, 3.8690, 3.1632]
if debug:
    res2 = optimize.fmin(clogit.loglike, tab2324)

res3 = optimize.fmin(clogit.loglike, np.zeros(6),maxfun=10000)
#this has same numbers as Greene table 23.24, but different sequence
#coefficient on GC is exactly 10% of Greene's
#TODO: get better starting values
'''
Optimization terminated successfully.
         Current function value: 199.128369
         Iterations: 957
         Function evaluations: 1456
array([-0.0961246 , -0.0155019 ,  0.01328757,  5.20741244,  3.86905293,
        3.16319074])
'''
res3corr = res3[[1, 0, 2, 3, 4, 5]]
res3corr[0] *= 10
print(res3corr - tab2324)  # diff 1e-5 to 1e-6
#199.128369 - 199.1284  #llf same up to print(precision of Greene

print(clogit.fit())


tree0 = ('top',
            [('Fly',['Air']),
             ('Ground', ['Train', 'Car', 'Bus'])
             ])

datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                    [xifloat[i]for i in range(4)]))

#for testing only (mock that returns it's own name
datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                    ['Airdata', 'Traindata', 'Busdata', 'Cardata']))

datadict.update({'top' :   [],
                 'Fly' :   [],
                 'Ground': []})

paramsind = {'top' :   [],
             'Fly' :   [],
             'Ground': [],
             'Air' :   ['GC', 'Ttme', 'ConstA', 'Hinc'],
             'Train' : ['GC', 'Ttme', 'ConstT'],
             'Bus' :   ['GC', 'Ttme', 'ConstB'],
             'Car' :   ['GC', 'Ttme']
             }

modru = RU2NMNL(endog, datadict, tree0, paramsind)
print(modru.calc_prob(modru.tree))
print('\nmodru.probs')
print(modru.probs)
