'''Trying out tree structure for nested logit

sum is standing for likelihood calculations

should collect and aggregate likelihood contributions bottom up

'''
from statsmodels.compat.python import lrange
import numpy as np

tree = [[0,1],[[2,3],[4,5,6]],[7]]
#singleton/degenerate branch needs to be list

xb = 2*np.arange(8)
testxb = 1 #0

def branch(tree):
    '''walking a tree bottom-up
    '''

    if not isinstance(tree[0], int):   #assumes leaves are int for choice index
        branchsum = 0
        for b in tree:
            branchsum += branch(b)
    else:
        print(tree)
        print('final branch with', tree, sum(tree))
        if testxb:
            return sum(xb[tree])
        else:
            return sum(tree)

    print('working on branch', tree, branchsum)
    return branchsum

print(branch(tree))



#new version that also keeps track of branch name and allows V_j for a branch
#   as in Greene, V_j + lamda * IV does not look the same as including the
#   explanatory variables in leaf X_j, V_j is linear in X, IV is logsumexp of X,


testxb = 0#1#0
def branch2(tree):
    '''walking a tree bottom-up based on dictionary
    '''


    if isinstance(tree,  tuple):   #assumes leaves are int for choice index
        name, subtree = tree
        print(name, data2[name])
        print('subtree', subtree)
        if testxb:
            branchsum = data2[name]
        else:
            branchsum = name  #0
        for b in subtree:
            #branchsum += branch2(b)
            branchsum = branchsum + branch2(b)
    else:
        leavessum = sum((data2[bi] for bi in tree))
        print('final branch with', tree, ''.join(tree), leavessum) #sum(tree)
        if testxb:
            return leavessum  #sum(xb[tree])
        else:
            return ''.join(tree) #sum(tree)

    print('working on branch', tree, branchsum)
    return branchsum

tree = [[0,1],[[2,3],[4,5,6]],[7]]
tree2 = ('top',
            [('B1',['a','b']),
             ('B2',
                   [('B21',['c', 'd']),
                    ('B22',['e', 'f', 'g'])
                    ]
              ),
             ('B3',['h'])]
         )

data2 = dict([i for i in zip('abcdefgh',lrange(8))])
#data2.update({'top':1000, 'B1':100, 'B2':200, 'B21':300,'B22':400, 'B3':400})
data2.update({'top':1000, 'B1':100, 'B2':200, 'B21':21,'B22':22, 'B3':300})

#data2
#{'a': 0, 'c': 2, 'b': 1, 'e': 4, 'd': 3, 'g': 6, 'f': 5, 'h': 7,
#'top': 1000, 'B22': 22, 'B21': 21, 'B1': 100, 'B2': 200, 'B3': 300}

print('\n tree with dictionary data')
print(branch2(tree2))  # results look correct for testxb=0 and 1


#parameters/coefficients map coefficient names to indices, list of indices into
#a 1d params one for each leave and branch

#Note: dict looses ordering
paramsind = {
 'B1': [],
 'a': ['consta', 'p'],
 'b': ['constb', 'p'],
 'B2': ['const2', 'x2'],
 'B21': [],
 'c': ['consta', 'p', 'time'],
 'd': ['consta', 'p', 'time'],
 'B22': ['x22'],
 'e': ['conste', 'p', 'hince'],
 'f': ['constt', 'p', 'hincf'],
 'g': [          'p', 'hincg'],
 'B3': [],
 'h': ['consth', 'p', 'h'],
 'top': []}

#unique, parameter array names,
#sorted alphabetically, order is/should be only internal

paramsnames = sorted(set([i for j in paramsind.values() for i in j]))

#mapping coefficient names to indices to unique/parameter array
paramsidx = dict((name, idx) for (idx,name) in enumerate(paramsnames))

#mapping branch and leaf names to index in parameter array
inddict = dict((k,[paramsidx[j] for j in v]) for k,v in paramsind.items())

'''
>>> paramsnames
['const2', 'consta', 'constb', 'conste', 'consth', 'constt', 'h', 'hince',
 'hincf', 'hincg', 'p', 'time', 'x2', 'x22']
>>> parmasidx
{'conste': 3, 'consta': 1, 'constb': 2, 'h': 6, 'time': 11, 'consth': 4,
 'p': 10, 'constt': 5, 'const2': 0, 'x2': 12, 'x22': 13, 'hince': 7,
 'hincg': 9, 'hincf': 8}
>>> inddict
{'a': [1, 10], 'c': [1, 10, 11], 'b': [2, 10], 'e': [3, 10, 7],
 'd': [1, 10, 11], 'g': [10, 9], 'f': [5, 10, 8], 'h': [4, 10, 6],
 'top': [], 'B22': [13], 'B21': [], 'B1': [], 'B2': [0, 12], 'B3': []}
>>> paramsind
{'a': ['consta', 'p'], 'c': ['consta', 'p', 'time'], 'b': ['constb', 'p'],
 'e': ['conste', 'p', 'hince'], 'd': ['consta', 'p', 'time'],
 'g': ['p', 'hincg'], 'f': ['constt', 'p', 'hincf'], 'h': ['consth', 'p', 'h'],
 'top': [], 'B22': ['x22'], 'B21': [], 'B1': [], 'B2': ['const2', 'x2'],
 'B3': []}
'''
