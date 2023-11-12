'''

Formulas
--------

This follows mostly Greene notation (in slides)
partially ignoring factors tau or mu for now, ADDED
(if all tau==1, then runmnl==clogit)

leaf k probability :

Prob(k|j) = exp(b_k * X_k / mu_j)/ sum_{i in L(j)} (exp(b_i * X_i / mu_j)

branch j probabilities :

Prob(j) = exp(b_j * X_j + mu*IV_j )/ sum_{i in NB(j)} (exp(b_i * X_i + mu_i*IV_i)

inclusive value of branch j :

IV_j = log( sum_{i in L(j)} (exp(b_i * X_i / mu_j) )

this is the log of the denominator of the leaf probabilities

L(j) : leaves at branch j, where k is child of j
NB(j) : set of j and it's siblings

Design
------

* splitting calculation transmission between returns and changes to
  instance.probs
  - probability for each leaf is in instance.probs
  - inclusive values and contribution of exog on branch level need to be
    added separately. handed up the tree through returns
* question: should params array be accessed directly through
  `self.recursionparams[self.parinddict[name]]` or should the dictionary
  return the values of the params, e.g. `self.params_node_dict[name]`.
  The second would be easier for fixing tau=1 for degenerate branches.
  The easiest might be to do the latter only for the taus and default to 1 if
  the key ('tau_'+branchname) is not found. I also need to exclude tau for
  degenerate branches from params, but then I cannot change them from the
  outside for testing and experimentation. (?)
* SAS manual describes restrictions on tau (though their model is a bit
  different), e.g. equal tau across sibling branches, fixed tau. The also
  allow linear and non-linear (? not sure) restriction on params, the
  regression coefficients. Related to previous issue, callback without access
  to the underlying array, where params_node_dict returns the actual params
  value would provide more flexibility to impose different kinds of restrictions.



bugs/problems
-------------

* singleton branches return zero to `top`, not a value
  I'm not sure what they are supposed to return, given the split between returns
  and instance.probs DONE
* Why does 'Air' (singleton branch) get probability exactly 0.5 ? DONE

TODO
----
* add tau, normalization for nested logit, currently tau is 1 (clogit)
  taus also needs to become part of params MOSTLY DONE
* add effect of branch level explanatory variables DONE
* write a generic multinomial logit that takes arbitrary probabilities, this
  would be the same for MNL, clogit and runmnl,
  delegate calculation of probabilities
* test on actual data,
  - tau=1 replicate clogit numbers,
  - transport example from Greene tests 1-level tree and degenerate sub-trees
  - test example for multi-level trees ???
* starting values: Greene mentiones that the starting values for the nested
  version come from the (non-nested) MNL version. SPSS uses constant equal
  (? check transformation) to sample frequencies and zeros for slope
  coefficient as starting values for (non-nested) MNL
* associated test statistics
  - (I do not think I will fight with the gradient or hessian of the log-like.)
  - basic MLE statistics can be generic
  - tests specific to the model (?)
* nice printouts since I'm currently collecting a lot of information in the tree
  recursion and everything has names

The only parts that are really necessary to get a functional nested logit are
adding the taus (DONE) and the MLE wrapper class. The rest are enhancements.

I added fake tau, one fixed tau for all branches. (OBSOLETE)
It's not clear where the tau for leaf should be added either at
original assignment of self.probs, or as part of the one-step-down
probability correction in the bottom branches. The second would be
cleaner (would make treatment of leaves and branches more symmetric,
but requires that initial assignment in the leaf only does
initialization. e.g self.probs = 1.  ???

DONE added taus

still todo:
- tau for degenerate branches are not identified, set to 1 for MLE
- rename parinddict to paramsinddict


Author: Josef Perktold
License : BSD (3-clause)
'''
from statsmodels.compat.python import lrange

from pprint import pprint

import numpy as np


def randintw(w, size=1):
    '''generate integer random variables given probabilties

    useful because it can be used as index into any array or sequence type

    Parameters
    ----------
    w : 1d array_like
        sequence of weights, probabilities. The weights are normalized to add
        to one.
    size : int or tuple of ints
        shape of output array

    Returns
    -------
    rvs : array of shape given by size
        random variables each distributed according to the same discrete
        distribution defined by (normalized) w.

    Examples
    --------
    >>> np.random.seed(0)
    >>> randintw([0.4, 0.4, 0.2], size=(2,6))
    array([[1, 1, 1, 1, 1, 1],
           [1, 2, 2, 0, 1, 1]])

    >>> np.bincount(randintw([0.6, 0.4, 0.0], size=3000))/3000.
    array([ 0.59566667,  0.40433333])

    '''
    #from Charles Harris, numpy mailing list
    from numpy.random import random
    p = np.cumsum(w)/np.sum(w)
    rvs = p.searchsorted(random(np.prod(size))).reshape(size)
    return rvs

def getbranches(tree):
    '''
    walk tree to get list of branches

    Parameters
    ----------
    tree : list of tuples
        tree as defined for RU2NMNL

    Returns
    -------
    branch : list
        list of all branch names

    '''
    if isinstance(tree, tuple):
        name, subtree = tree
        a = [name]
        for st in subtree:
            a.extend(getbranches(st))
        return a
    return []

def getnodes(tree):
    '''
    walk tree to get list of branches and list of leaves

    Parameters
    ----------
    tree : list of tuples
        tree as defined for RU2NMNL

    Returns
    -------
    branch : list
        list of all branch names
    leaves : list
        list of all leaves names

    '''
    if isinstance(tree, tuple):
        name, subtree = tree
        ab = [name]
        al = []
        #degenerate branches
        if len(subtree) == 1:
            adeg = [name]
        else:
            adeg = []

        for st in subtree:
            b, l, d = getnodes(st)
            ab.extend(b)
            al.extend(l)
            adeg.extend(d)
        return ab, al, adeg
    return [], [tree], []


testxb = 2 #global to class to return strings instead of numbers

class RU2NMNL:
    '''Nested Multinomial Logit with Random Utility 2 parameterization


    Parameters
    ----------
    endog : ndarray
        not used in this part
    exog : dict_like
        dictionary access to data where keys correspond to branch and leaf
        names. The values are the data arrays for the exog in that node.
    tree : nested tuples and lists
        each branch, tree or subtree, is defined by a tuple
        (branch_name, [subtree1, subtree2, ..., subtreek])
        Bottom branches have as subtrees the list of leaf names.
    paramsind : dictionary
        dictionary that maps branch and leaf names to the names of parameters,
        the coefficients for exogs)

    Methods
    -------
    get_probs

    Attributes
    ----------
    branches
    leaves
    paramsnames
    parinddict

    Notes
    -----
    endog needs to be encoded so it is consistent with self.leaves, which
    defines the columns for the probability array. The ordering in leaves is
    determined by the ordering of the tree.
    In the dummy encoding of endog, the columns of endog need to have the
    same order as self.leaves. In the integer encoding, the integer for a
    choice has to correspond to the index in self.leaves.
    (This could be made more robust, by handling the endog encoding internally
    by leaf names, if endog is defined as categorical variable with
    associated category level names.)

    '''

    def __init__(self, endog, exog, tree, paramsind):
        self.endog = endog
        self.datadict = exog
        self.tree = tree
        self.paramsind = paramsind

        self.branchsum = ''
        self.probs = {}
        self.probstxt = {}
        self.branchleaves = {}
        self.branchvalues = {}  #just to keep track of returns by branches
        self.branchsums = {}
        self.bprobs = {}
        self.branches, self.leaves, self.branches_degenerate  = getnodes(tree)
        self.nbranches = len(self.branches)

        #copied over but not quite sure yet
        #unique, parameter array names,
        #sorted alphabetically, order is/should be only internal

        self.paramsnames = (sorted(set([i for j in paramsind.values()
                                       for i in j])) +
                            ['tau_%s' % bname for bname in self.branches])

        self.nparams = len(self.paramsnames)

        #mapping coefficient names to indices to unique/parameter array
        self.paramsidx = dict((name, idx) for (idx,name) in
                              enumerate(self.paramsnames))

        #mapping branch and leaf names to index in parameter array
        self.parinddict = dict((k, [self.paramsidx[j] for j in v])
                               for k,v in self.paramsind.items())

        self.recursionparams = 1. + np.arange(len(self.paramsnames))
        #for testing that individual parameters are used in the right place
        self.recursionparams = np.zeros(len(self.paramsnames))
        #self.recursionparams[2] = 1
        self.recursionparams[-self.nbranches:] = 1  #values for tau's
        #self.recursionparams[-2] = 2


    def get_probs(self, params):
        '''
        obtain the probability array given an array of parameters

        This is the function that can be called by loglike or other methods
        that need the probabilities as function of the params.

        Parameters
        ----------
        params : 1d array, (nparams,)
            coefficients and tau that parameterize the model. The required
            length can be obtained by nparams. (and will depend on the number
            of degenerate leaves - not yet)

        Returns
        -------
        probs : ndarray, (nobs, nchoices)
            probabilities for all choices for each observation. The order
            is available by attribute leaves. See note in docstring of class



        '''
        self.recursionparams = params
        self.calc_prob(self.tree)
        probs_array = np.array([self.probs[leaf] for leaf in self.leaves])
        return probs_array
        #what's the ordering? Should be the same as sequence in tree.
        #TODO: need a check/assert that this sequence is the same as the
        #      encoding in endog


    def calc_prob(self, tree, parent=None):
        '''walking a tree bottom-up based on dictionary
        '''

        #0.5#2 #placeholder for now
        #should be tau=self.taus[name] but as part of params for optimization
        endog = self.endog
        datadict = self.datadict
        paramsind = self.paramsind
        branchsum = self.branchsum


        if isinstance(tree, tuple):   #assumes leaves are int for choice index

            name, subtree = tree
            self.branchleaves[name] = []  #register branch in dictionary

            tau = self.recursionparams[self.paramsidx['tau_'+name]]
            if DEBUG:
                print('----------- starting next branch-----------')
                print(name, datadict[name], 'tau=', tau)
                print('subtree', subtree)
            branchvalue = []
            if testxb == 2:
                branchsum = 0
            elif testxb == 1:
                branchsum = datadict[name]
            else:
                branchsum = name
            for b in subtree:
                if DEBUG:
                    print(b)
                bv = self.calc_prob(b, name)
                bv = np.exp(bv/tau)  #this should not be here, when adding branch data
                branchvalue.append(bv)
                branchsum = branchsum + bv
            self.branchvalues[name] = branchvalue #keep track what was returned

            if DEBUG:
                print('----------- returning to branch-----------')
                print(name)
                print('branchsum in branch', name, branchsum)

            if parent:
                if DEBUG:
                    print('parent', parent)
                self.branchleaves[parent].extend(self.branchleaves[name])
            if 0:  #not name == 'top':  # not used anymore !!! ???
            #if not name == 'top':
                #TODO: do I need this only on the lowest branches ?
                tmpsum = 0
                for k in self.branchleaves[name]:
                    #similar to this is now also in return branch values
                    #depends on what will be returned
                    tmpsum += self.probs[k]
                    iv = np.log(tmpsum)

                for k in self.branchleaves[name]:
                    self.probstxt[k] = self.probstxt[k] + ['*' + name + '-prob' +
                                    '(%s)' % ', '.join(self.paramsind[name])]

                    #TODO: does this use the denominator twice now
                    self.probs[k] = self.probs[k] / tmpsum
                    if np.size(self.datadict[name])>0:
                        #not used yet, might have to move one indentation level
                        #self.probs[k] = self.probs[k] / tmpsum
##                            np.exp(-self.datadict[name] *
##                             np.sum(self.recursionparams[self.parinddict[name]]))
                        if DEBUG:
                            print('self.datadict[name], self.probs[k]')
                            print(self.datadict[name], self.probs[k])
                    #if not name == 'top':
                    #    self.probs[k] = self.probs[k] * np.exp( iv)

            #walk one level down again to add branch probs to instance.probs
            self.bprobs[name] = []
            for bidx, b in enumerate(subtree):
                if DEBUG:
                    print('repr(b)', repr(b), bidx)
                #if len(b) == 1: #TODO: skip leaves, check this
                if not isinstance(b,  tuple): # isinstance(b, str):
                    #TODO: replace this with a check for branch (tuple) instead
                    #this implies name is a bottom branch,
                    #possible to add special things here
                    self.bprobs[name].append(self.probs[b])
                    #TODO: need tau possibly here
                    self.probs[b] = self.probs[b] / branchsum
                    if DEBUG:
                        print('*********** branchsum at bottom branch', branchsum)
                    #self.bprobs[name].append(self.probs[b])
                else:
                    bname = b[0]
                    branchsum2 = sum(self.branchvalues[name])
                    assert np.abs(branchsum - branchsum2).sum() < 1e-8
                    bprob = branchvalue[bidx]/branchsum
                    self.bprobs[name].append(bprob)

                    for k in self.branchleaves[bname]:

                        if DEBUG:
                            print('branchprob', bname, k, bprob, branchsum)
                        #temporary hack with maximum to avoid zeros
                        self.probs[k] = self.probs[k] * np.maximum(bprob, 1e-4)


            if DEBUG:
                print('working on branch', tree, branchsum)
            if testxb<2:
                return branchsum
            else: #this is the relevant part
                self.branchsums[name] = branchsum
                if np.size(self.datadict[name])>0:
                    branchxb = np.sum(self.datadict[name] *
                                  self.recursionparams[self.parinddict[name]])
                else:
                    branchxb = 0
                if not name=='top':
                    tau = self.recursionparams[self.paramsidx['tau_'+name]]
                else:
                    tau = 1
                iv = branchxb + tau * branchsum #which tau: name or parent???
                return branchxb + tau * np.log(branchsum) #iv
                #branchsum is now IV, TODO: add effect of branch variables

        else:
            tau = self.recursionparams[self.paramsidx['tau_'+parent]]
            if DEBUG:
                print('parent', parent)
            self.branchleaves[parent].append(tree) # register leave with parent
            self.probstxt[tree] = [tree + '-prob' +
                                '(%s)' % ', '.join(self.paramsind[tree])]
            #this is not yet a prob, not normalized to 1, it is exp(x*b)
            leafprob = np.exp(np.sum(self.datadict[tree] *
                                  self.recursionparams[self.parinddict[tree]])
                              / tau)   # fake tau for now, wrong spot ???
            #it seems I get the same answer with and without tau here
            self.probs[tree] = leafprob  #= 1 #try initialization only
            #TODO: where  should I add tau in the leaves

            if testxb == 2:
                return np.log(leafprob)
            elif testxb == 1:
                leavessum = np.array(datadict[tree]) # sum((datadict[bi] for bi in datadict[tree]))
                if DEBUG:
                    print('final branch with', tree, ''.join(tree), leavessum) #sum(tree)
                return leavessum  #sum(xb[tree])
            elif testxb == 0:
                return ''.join(tree) #sum(tree)



if __name__ == '__main__':
    DEBUG = 0

    endog = 5 # dummy place holder


    ##############  Example similar to Greene

    #get pickled data
    #endog3, xifloat3 = pickle.load(open('xifloat2.pickle','rb'))


    tree0 = ('top',
                [('Fly',['Air']),
                 ('Ground', ['Train', 'Car', 'Bus'])
                 ])

    ''' this is with real data from Greene's clogit example
    datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                        [xifloat[i]for i in range(4)]))
    '''

    #for testing only (mock that returns it's own name
    datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                        ['Airdata', 'Traindata', 'Busdata', 'Cardata']))

    if testxb:
        datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                        np.arange(4)))

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
    modru.recursionparams[-1] = 2
    modru.recursionparams[1] = 1

    print('Example 1')
    print('---------\n')
    print(modru.calc_prob(modru.tree))

    print('Tree')
    pprint(modru.tree)
    print('\nmodru.probs')
    pprint(modru.probs)



    ##############  example with many layers

    tree2 = ('top',
                [('B1',['a','b']),
                 ('B2',
                       [('B21',['c', 'd']),
                        ('B22',['e', 'f', 'g'])
                        ]
                  ),
                 ('B3',['h'])])

    #Note: dict looses ordering
    paramsind2 = {
     'B1': [],
     'a': ['consta', 'p'],
     'b': ['constb', 'p'],
     'B2': ['const2', 'x2'],
     'B21': [],
     'c': ['constc', 'p', 'time'],
     'd': ['constd', 'p', 'time'],
     'B22': ['x22'],
     'e': ['conste', 'p', 'hince'],
     'f': ['constf', 'p', 'hincf'],
     'g': [          'p', 'hincg'],
     'B3': [],
     'h': ['consth', 'p', 'h'],
     'top': []}


    datadict2 = dict([i for i in zip('abcdefgh',lrange(8))])
    datadict2.update({'top':1000, 'B1':100, 'B2':200, 'B21':21,'B22':22, 'B3':300})
    '''
    >>> pprint(datadict2)
    {'B1': 100,
     'B2': 200,
     'B21': 21,
     'B22': 22,
     'B3': 300,
     'a': 0.5,
     'b': 1,
     'c': 2,
     'd': 3,
     'e': 4,
     'f': 5,
     'g': 6,
     'h': 7,
     'top': 1000}
    '''


    modru2 = RU2NMNL(endog, datadict2, tree2, paramsind2)
    modru2.recursionparams[-3] = 2
    modru2.recursionparams[3] = 1
    print('\n\nExample 2')
    print('---------\n')
    print(modru2.calc_prob(modru2.tree))
    print('Tree')
    pprint(modru2.tree)
    print('\nmodru.probs')
    pprint(modru2.probs)


    print('sum of probs', sum(list(modru2.probs.values())))
    print('branchvalues')
    print(modru2.branchvalues)
    print(modru.branchvalues)

    print('branch probabilities')
    print(modru.bprobs)

    print('degenerate branches')
    print(modru.branches_degenerate)

    '''
    >>> modru.bprobs
    {'Fly': [], 'top': [0.0016714179077931082, 0.99832858209220687], 'Ground': []}
    >>> modru2.bprobs
    {'top': [0.25000000000000006, 0.62499999999999989, 0.12500000000000003], 'B22': [], 'B21': [], 'B1': [], 'B2': [0.40000000000000008, 0.59999999999999998], 'B3': []}
    '''

    params1 = np.array([ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  2.])
    print(modru.get_probs(params1))
    params2 = np.array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  1.,  1.])
    print(modru2.get_probs(params2)) #raises IndexError
