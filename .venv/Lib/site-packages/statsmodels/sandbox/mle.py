'''What's the origin of this file? It is not ours.
Does not run because of missing mtx files, now included

changes: JP corrections to imports so it runs, comment out print
'''
import numpy as np
from numpy import dot,  outer, random
from scipy import io, linalg, optimize
from scipy.sparse import eye as speye
import matplotlib.pyplot as plt

def R(v):
    rq = dot(v.T,A*v)/dot(v.T,B*v)
    res = (A*v-rq*B*v)/linalg.norm(B*v)
    data.append(linalg.norm(res))
    return rq

def Rp(v):
    """ Gradient """
    result = 2*(A*v-R(v)*B*v)/dot(v.T,B*v)
    #print "Rp: ", result
    return result

def Rpp(v):
    """ Hessian """
    result = 2*(A-R(v)*B-outer(B*v,Rp(v))-outer(Rp(v),B*v))/dot(v.T,B*v)
    #print "Rpp: ", result
    return result


A = io.mmread('nos4.mtx') # clustered eigenvalues
#B = io.mmread('bcsstm02.mtx.gz')
#A = io.mmread('bcsstk06.mtx.gz') # clustered eigenvalues
#B = io.mmread('bcsstm06.mtx.gz')
n = A.shape[0]
B = speye(n,n)
random.seed(1)
v_0=random.rand(n)

print("try fmin_bfgs")
full_output = 1
data=[]
v,fopt, gopt, Hopt, func_calls, grad_calls, warnflag, allvecs = \
        optimize.fmin_bfgs(R,v_0,fprime=Rp,full_output=full_output,retall=1)
if warnflag == 0:
    plt.semilogy(np.arange(0,len(data)),data)
    print('Rayleigh quotient BFGS',R(v))


print("fmin_bfgs OK")

print("try fmin_ncg")

#
# WARNING: the program may hangs if fmin_ncg is used
#
data=[]
v,fopt, fcalls, gcalls, hcalls, warnflag, allvecs = \
        optimize.fmin_ncg(R,v_0,fprime=Rp,fhess=Rpp,full_output=full_output,retall=1)
if warnflag==0:
    plt.figure()
    plt.semilogy(np.arange(0,len(data)),data)
    print('Rayleigh quotient NCG',R(v))
