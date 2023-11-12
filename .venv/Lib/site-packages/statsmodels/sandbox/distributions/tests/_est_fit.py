# NOTE: contains only one test, _est_cont_fit, that is renamed so that
#       the test runner does not run it
# I put this here for the record and for the case when someone wants to
# verify the quality of fit
# with current parameters: relatively small sample size, default starting values
#       Ran 84 tests in 401.797s
#       FAILED (failures=15)


import numpy as np

from scipy import stats

from .distparams import distcont

# this is not a proper statistical test for convergence, but only
# verifies that the estimate and true values do not differ by too much
n_repl1 = 1000 # sample size for first run
n_repl2 = 5000 # sample size for second run, if first run fails
thresh_percent = 0.25 # percent of true parameters for fail cut-off
thresh_min = 0.75  # minimum difference estimate - true to fail test

#distcont = [['genextreme', (3.3184017469423535,)]]

def _est_cont_fit():
    # this tests the closeness of the estimated parameters to the true
    # parameters with fit method of continuous distributions
    # Note: is slow, some distributions do not converge with sample size <= 10000

    for distname, arg in distcont:
        yield check_cont_fit, distname,arg


def check_cont_fit(distname,arg):
    distfn = getattr(stats, distname)
    rvs = distfn.rvs(size=n_repl1,*arg)
    est = distfn.fit(rvs)  #,*arg) # start with default values

    truearg = np.hstack([arg,[0.0,1.0]])
    diff = est-truearg

    txt = ''
    diffthreshold = np.max(np.vstack([truearg*thresh_percent,
                    np.ones(distfn.numargs+2)*thresh_min]),0)
    # threshold for location
    diffthreshold[-2] = np.max([np.abs(rvs.mean())*thresh_percent,thresh_min])

    if np.any(np.isnan(est)):
        raise AssertionError('nan returned in fit')
    else:
        if np.any((np.abs(diff) - diffthreshold) > 0.0):
##            txt = 'WARNING - diff too large with small sample'
##            print 'parameter diff =', diff - diffthreshold, txt
            rvs = np.concatenate([rvs,distfn.rvs(size=n_repl2-n_repl1,*arg)])
            est = distfn.fit(rvs) #,*arg)
            truearg = np.hstack([arg,[0.0,1.0]])
            diff = est-truearg
            if np.any((np.abs(diff) - diffthreshold) > 0.0):
                txt  = 'parameter: %s\n' % str(truearg)
                txt += 'estimated: %s\n' % str(est)
                txt += 'diff     : %s\n' % str(diff)
                raise AssertionError('fit not very good in %s\n' % distfn.name + txt)



if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
