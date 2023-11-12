'''given a 1D sample of observation, find a matching distribution

* estimate maximum likelihood parameter for each distribution
* rank estimated distribution by Kolmogorov-Smirnov and Anderson-Darling
  test statistics

Author: Josef Pktd
License: Simplified BSD
original December 2008

TODO:

* refactor to result class
* split estimation by support, add option and choose automatically
*

'''
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

#stats.distributions.beta_gen._fitstart = lambda self, data : (5,5,0,1)

def plothist(x,distfn, args, loc, scale, right=1):

    plt.figure()
    # the histogram of the data
    n, bins, patches = plt.hist(x, 25, normed=1, facecolor='green', alpha=0.75)
    maxheight = max([p.get_height() for p in patches])
    print(maxheight)
    axlim = list(plt.axis())
    #print(axlim)
    axlim[-1] = maxheight*1.05
    #plt.axis(tuple(axlim))
##    print(bins)
##    print('args in plothist', args)
    # add a 'best fit' line
    #yt = stats.norm.pdf( bins, loc=loc, scale=scale)
    yt = distfn.pdf( bins, loc=loc, scale=scale, *args)
    yt[yt>maxheight]=maxheight
    lt = plt.plot(bins, yt, 'r--', linewidth=1)
    ys = stats.t.pdf( bins, 10,scale=10,)*right
    ls = plt.plot(bins, ys, 'b-', linewidth=1)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Testing: %s :}\ \mu=%f,\ \sigma=%f$' % (distfn.name,loc,scale))

    #plt.axis([bins[0], bins[-1], 0, 0.134+0.05])

    plt.grid(True)
    plt.draw()
    #plt.show()
    #plt.close()





#targetdist = ['norm','t','truncnorm','johnsonsu','johnsonsb',
targetdist = ['norm','alpha', 'anglit', 'arcsine',
           'beta', 'betaprime', 'bradford', 'burr', 'fisk', 'cauchy',
           'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang',
           'expon', 'exponweib', 'exponpow', 'fatiguelife', 'foldcauchy',
           'f', 'foldnorm', 'frechet_r', 'weibull_min', 'frechet_l',
           'weibull_max', 'genlogistic', 'genpareto', 'genexpon', 'genextreme',
           'gamma', 'gengamma', 'genhalflogistic', 'gompertz', 'gumbel_r',
           'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant',
           'gausshyper', 'invgamma', 'invnorm', 'invweibull', 'johnsonsb',
           'johnsonsu', 'laplace', 'levy', 'levy_l',
           'logistic', 'loggamma', 'loglaplace', 'lognorm', 'gilbrat',
           'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 't',
           'nct', 'pareto', 'lomax', 'powerlaw', 'powerlognorm', 'powernorm',
           'rdist', 'rayleigh', 'reciprocal', 'rice', 'recipinvgauss',
           'semicircular', 'triang', 'truncexpon', 'truncnorm',
           'tukeylambda', 'uniform', 'vonmises', 'wald', 'wrapcauchy',

           'binom', 'bernoulli', 'nbinom', 'geom', 'hypergeom', 'logser',
           'poisson', 'planck', 'boltzmann', 'randint', 'zipf', 'dlaplace']

left = []
right = []
finite = []
unbound = []
other = []
contdist = []
discrete = []

categ = {('open','open'):'unbound', ('0','open'):'right',('open','0',):'left',
             ('finite','finite'):'finite',('oth','oth'):'other'}
categ = {('open','open'):unbound, ('0','open'):right,('open','0',):left,
             ('finite','finite'):finite,('oth','oth'):other}

categ2 = {
    ('open', '0') : ['frechet_l', 'weibull_max', 'levy_l'],
    ('finite', 'finite') : ['anglit', 'cosine', 'rdist', 'semicircular'],
    ('0', 'open') : ['alpha', 'burr', 'fisk', 'chi', 'chi2', 'erlang',
                'expon', 'exponweib', 'exponpow', 'fatiguelife', 'foldcauchy', 'f',
                'foldnorm', 'frechet_r', 'weibull_min', 'genpareto', 'genexpon',
                'gamma', 'gengamma', 'genhalflogistic', 'gompertz', 'halfcauchy',
                'halflogistic', 'halfnorm', 'invgamma', 'invnorm', 'invweibull',
                'levy', 'loglaplace', 'lognorm', 'gilbrat', 'maxwell', 'mielke',
                'nakagami', 'ncx2', 'ncf', 'lomax', 'powerlognorm', 'rayleigh',
                'rice', 'recipinvgauss', 'truncexpon', 'wald'],
    ('open', 'open') : ['cauchy', 'dgamma', 'dweibull', 'genlogistic', 'genextreme',
                'gumbel_r', 'gumbel_l', 'hypsecant', 'johnsonsu', 'laplace',
                'logistic', 'loggamma', 't', 'nct', 'powernorm', 'reciprocal',
                'truncnorm', 'tukeylambda', 'vonmises'],
    ('0', 'finite') : ['arcsine', 'beta', 'betaprime', 'bradford', 'gausshyper',
                'johnsonsb', 'powerlaw', 'triang', 'uniform', 'wrapcauchy'],
    ('finite', 'open') : ['pareto']
    }

#Note: weibull_max == frechet_l

right_incorrect = ['genextreme']

right_all = categ2[('0', 'open')] + categ2[('0', 'finite')] + categ2[('finite', 'open')]\
            + right_incorrect

for distname in targetdist:
    distfn = getattr(stats,distname)
    if hasattr(distfn,'_pdf'):
        if np.isinf(distfn.a):
            low = 'open'
        elif distfn.a == 0:
            low = '0'
        else:
            low = 'finite'
        if np.isinf(distfn.b):
            high = 'open'
        elif distfn.b == 0:
            high = '0'
        else:
            high = 'finite'
        contdist.append(distname)
        categ.setdefault((low,high),[]).append(distname)

not_good = ['genextreme', 'reciprocal', 'vonmises']
# 'genextreme' is right (or left?), 'reciprocal' requires 0<a<b, 'vonmises' no a,b
targetdist = [f for f in categ[('open', 'open')] if f not in not_good]
not_good = ['wrapcauchy']
not_good = ['vonmises']
not_good = ['genexpon','vonmises']
#'wrapcauchy' requires additional parameter (scale) in argcheck
targetdist = [f for f in contdist if f not in not_good]
#targetdist = contdist
#targetdist = not_good
#targetdist = ['t', 'f']
#targetdist = ['norm','burr']

if __name__ == '__main__':

    #TODO: calculate correct tail probability for mixture
    prefix = 'run_conv500_1_'
    convol = 0.75
    n = 500
    dgp_arg = 10
    dgp_scale = 10
    results = []
    for i in range(1):
        rvs_orig = stats.t.rvs(dgp_arg,scale=dgp_scale,size=n*convol)
        rvs_orig = np.hstack((rvs_orig,stats.halflogistic.rvs(loc=0.4, scale=5.0,size =n*(1-convol))))
        rvs_abs = np.absolute(rvs_orig)
        rvs_pos = rvs_orig[rvs_orig>0]
        rightfactor = 1
        rvs_right = rvs_pos
        print('='*50)
        print('samplesize = ', n)
        for distname in targetdist:
            distfn = getattr(stats,distname)
            if distname in right_all:
                rvs = rvs_right
                rind = rightfactor

            else:
                rvs = rvs_orig
                rind = 1
            print('-'*30)
            print('target = %s' % distname)
            sm = rvs.mean()
            sstd = np.sqrt(rvs.var())
            ssupp = (rvs.min(), rvs.max())
            if distname in ['truncnorm','betaprime','reciprocal']:

                par0 = (sm-2*sstd,sm+2*sstd)
                par_est = tuple(distfn.fit(rvs,loc=sm,scale=sstd,*par0))
            elif distname == 'norm':
                par_est = tuple(distfn.fit(rvs,loc=sm,scale=sstd))
            elif distname == 'genextreme':
                par_est = tuple(distfn.fit(rvs,-5,loc=sm,scale=sstd))
            elif distname == 'wrapcauchy':
                par_est = tuple(distfn.fit(rvs,0.5,loc=0,scale=sstd))
            elif distname == 'f':
                par_est = tuple(distfn.fit(rvs,10,15,loc=0,scale=1))

            elif distname in right:
                sm = rvs.mean()
                sstd = np.sqrt(rvs.var())
                par_est = tuple(distfn.fit(rvs,loc=0,scale=1))
            else:
                sm = rvs.mean()
                sstd = np.sqrt(rvs.var())
                par_est = tuple(distfn.fit(rvs,loc=sm,scale=sstd))


            print('fit', par_est)
            arg_est = par_est[:-2]
            loc_est = par_est[-2]
            scale_est = par_est[-1]
            rvs_normed = (rvs-loc_est)/scale_est
            ks_stat, ks_pval = stats.kstest(rvs_normed,distname, arg_est)
            print('kstest', ks_stat, ks_pval)
            quant = 0.1
            crit = distfn.ppf(1-quant*float(rind), loc=loc_est, scale=scale_est,*par_est)
            tail_prob = stats.t.sf(crit,dgp_arg,scale=dgp_scale)
            print('crit, prob', quant, crit, tail_prob)
            #if distname == 'norm':
                #plothist(rvs,loc_est,scale_est)
                #args = tuple()
            results.append([distname,ks_stat, ks_pval,arg_est,loc_est,scale_est,crit,tail_prob ])
            #plothist(rvs,distfn,arg_est,loc_est,scale_est)

    #plothist(rvs,distfn,arg_est,loc_est,scale_est)
    #plt.show()
    #plt.close()
    #TODO: collect results and compare tail quantiles


    from operator import itemgetter

    res_sort = sorted(results, key = itemgetter(2))

    res_sort.reverse()  #kstest statistic: smaller is better, pval larger is better

    print('number of distributions', len(res_sort))
    imagedir = 'matchresults'
    import os
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)

    for ii,di in enumerate(res_sort):
        distname,ks_stat, ks_pval,arg_est,loc_est,scale_est,crit,tail_prob = di[:]
        distfn = getattr(stats,distname)
        if distname in right_all:
            rvs = rvs_right
            rind = rightfactor
            ri = 'r'
        else:
            rvs = rvs_orig
            ri = ''
            rind = 1
        print('%s ks-stat = %f, ks-pval = %f tail_prob = %f)' % \
              (distname, ks_stat, ks_pval, tail_prob))
    ##    print('arg_est = %s, loc_est = %f scale_est = %f)' % \
    ##          (repr(arg_est),loc_est,scale_est))
        plothist(rvs,distfn,arg_est,loc_est,scale_est,right = rind)
        plt.savefig(os.path.join(imagedir,'%s%s%02d_%s.png'% (prefix, ri,ii, distname)))
    ##plt.show()
    ##plt.close()
