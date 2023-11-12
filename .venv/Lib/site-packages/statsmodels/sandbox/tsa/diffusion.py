'''getting started with diffusions, continuous time stochastic processes

Author: josef-pktd
License: BSD


References
----------

An Algorithmic Introduction to Numerical Simulation of Stochastic Differential
Equations
Author(s): Desmond J. Higham
Source: SIAM Review, Vol. 43, No. 3 (Sep., 2001), pp. 525-546
Published by: Society for Industrial and Applied Mathematics
Stable URL: http://www.jstor.org/stable/3649798

http://www.sitmo.com/  especially the formula collection


Notes
-----

OU process: use same trick for ARMA with constant (non-zero mean) and drift
some of the processes have easy multivariate extensions

*Open Issues*

include xzero in returned sample or not? currently not

*TODOS*

* Milstein from Higham paper, for which processes does it apply
* Maximum Likelihood estimation
* more statistical properties (useful for tests)
* helper functions for display and MonteCarlo summaries (also for testing/checking)
* more processes for the menagerie (e.g. from empirical papers)
* characteristic functions
* transformations, non-linear e.g. log
* special estimators, e.g. Ait Sahalia, empirical characteristic functions
* fft examples
* check naming of methods, "simulate", "sample", "simexact", ... ?



stochastic volatility models: estimation unclear

finance applications ? option pricing, interest rate models


'''
import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt

#np.random.seed(987656789)

class Diffusion:
    '''Wiener Process, Brownian Motion with mu=0 and sigma=1
    '''
    def __init__(self):
        pass

    def simulateW(self, nobs=100, T=1, dt=None, nrepl=1):
        '''generate sample of Wiener Process
        '''
        dt = T*1.0/nobs
        t = np.linspace(dt, 1, nobs)
        dW = np.sqrt(dt)*np.random.normal(size=(nrepl, nobs))
        W = np.cumsum(dW,1)
        self.dW = dW
        return W, t

    def expectedsim(self, func, nobs=100, T=1, dt=None, nrepl=1):
        '''get expectation of a function of a Wiener Process by simulation

        initially test example from
        '''
        W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
        U = func(t, W)
        Umean = U.mean(0)
        return U, Umean, t

class AffineDiffusion(Diffusion):
    r'''

    differential equation:

    :math::
    dx_t = f(t,x)dt + \sigma(t,x)dW_t

    integral:

    :math::
    x_T = x_0 + \int_{0}^{T}f(t,S)dt + \int_0^T  \sigma(t,S)dW_t

    TODO: check definition, affine, what about jump diffusion?

    '''

    def __init__(self):
        pass

    def sim(self, nobs=100, T=1, dt=None, nrepl=1):
        # this does not look correct if drift or sig depend on x
        # see arithmetic BM
        W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
        dx =  self._drift() + self._sig() * W
        x  = np.cumsum(dx,1)
        xmean = x.mean(0)
        return x, xmean, t

    def simEM(self, xzero=None, nobs=100, T=1, dt=None, nrepl=1, Tratio=4):
        '''

        from Higham 2001

        TODO: reverse parameterization to start with final nobs and DT
        TODO: check if I can skip the loop using my way from exactprocess
              problem might be Winc (reshape into 3d and sum)
        TODO: (later) check memory efficiency for large simulations
        '''
        #TODO: reverse parameterization to start with final nobs and DT
        nobs = nobs * Tratio  # simple way to change parameter
        # maybe wrong parameterization,
        # drift too large, variance too small ? which dt/Dt
        # _drift, _sig independent of dt is wrong
        if xzero is None:
            xzero = self.xzero
        if dt is None:
            dt = T*1.0/nobs
        W, t = self.simulateW(nobs=nobs, T=T, dt=dt, nrepl=nrepl)
        dW = self.dW
        t = np.linspace(dt, 1, nobs)
        Dt = Tratio*dt
        L = nobs/Tratio        # L EM steps of size Dt = R*dt
        Xem = np.zeros((nrepl,L))    # preallocate for efficiency
        Xtemp = xzero
        Xem[:,0] = xzero
        for j in np.arange(1,L):
            #Winc = np.sum(dW[:,Tratio*(j-1)+1:Tratio*j],1)
            Winc = np.sum(dW[:,np.arange(Tratio*(j-1)+1,Tratio*j)],1)
            #Xtemp = Xtemp + Dt*lamda*Xtemp + mu*Xtemp*Winc;
            Xtemp = Xtemp + self._drift(x=Xtemp) + self._sig(x=Xtemp) * Winc
            #Dt*lamda*Xtemp + mu*Xtemp*Winc;
            Xem[:,j] = Xtemp
        return Xem

'''
    R = 4; Dt = R*dt; L = N/R;        % L EM steps of size Dt = R*dt
    Xem = zeros(1,L);                 % preallocate for efficiency
    Xtemp = Xzero;
    for j = 1:L
       Winc = sum(dW(R*(j-1)+1:R*j));
       Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp*Winc;
       Xem(j) = Xtemp;
    end
'''

class ExactDiffusion(AffineDiffusion):
    '''Diffusion that has an exact integral representation

    this is currently mainly for geometric, log processes

    '''

    def __init__(self):
        pass

    def exactprocess(self, xzero, nobs, ddt=1., nrepl=2):
        '''ddt : discrete delta t



        should be the same as an AR(1)
        not tested yet
        '''
        t = np.linspace(ddt, nobs*ddt, nobs)
        #expnt = np.exp(-self.lambd * t)
        expddt = np.exp(-self.lambd * ddt)
        normrvs = np.random.normal(size=(nrepl,nobs))
        #do I need lfilter here AR(1) ? if mean reverting lag-coeff<1
        #lfilter does not handle 2d arrays, it does?
        inc = self._exactconst(expddt) + self._exactstd(expddt) * normrvs
        return signal.lfilter([1.], [1.,-expddt], inc)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        meant = xzero * expnt + self._exactconst(expnt)
        stdt = self._exactstd(expnt)
        return stats.norm(loc=meant, scale=stdt)

class ArithmeticBrownian(AffineDiffusion):
    '''
    :math::
    dx_t &= \\mu dt + \\sigma dW_t
    '''

    def __init__(self, xzero, mu, sigma):
        self.xzero = xzero
        self.mu = mu
        self.sigma = sigma

    def _drift(self, *args, **kwds):
        return self.mu
    def _sig(self, *args, **kwds):
        return self.sigma
    def exactprocess(self, nobs, xzero=None, ddt=1., nrepl=2):
        '''ddt : discrete delta t

        not tested yet
        '''
        if xzero is None:
            xzero = self.xzero
        t = np.linspace(ddt, nobs*ddt, nobs)
        normrvs = np.random.normal(size=(nrepl,nobs))
        inc = self._drift + self._sigma * np.sqrt(ddt) * normrvs
        #return signal.lfilter([1.], [1.,-1], inc)
        return xzero + np.cumsum(inc,1)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        meant = self._drift * t
        stdt = self._sigma * np.sqrt(t)
        return stats.norm(loc=meant, scale=stdt)


class GeometricBrownian(AffineDiffusion):
    '''Geometric Brownian Motion

    :math::
    dx_t &= \\mu x_t dt + \\sigma x_t dW_t

    $x_t $ stochastic process of Geometric Brownian motion,
    $\\mu $ is the drift,
    $\\sigma $ is the Volatility,
    $W$ is the Wiener process (Brownian motion).

    '''
    def __init__(self, xzero, mu, sigma):
        self.xzero = xzero
        self.mu = mu
        self.sigma = sigma

    def _drift(self, *args, **kwds):
        x = kwds['x']
        return self.mu * x
    def _sig(self, *args, **kwds):
        x = kwds['x']
        return self.sigma * x


class OUprocess(AffineDiffusion):
    '''Ornstein-Uhlenbeck

    :math::
      dx_t&=\\lambda(\\mu - x_t)dt+\\sigma dW_t

    mean reverting process



    TODO: move exact higher up in class hierarchy
    '''
    def __init__(self, xzero, mu, lambd, sigma):
        self.xzero = xzero
        self.lambd = lambd
        self.mu = mu
        self.sigma = sigma

    def _drift(self, *args, **kwds):
        x = kwds['x']
        return self.lambd * (self.mu - x)
    def _sig(self, *args, **kwds):
        x = kwds['x']
        return self.sigma * x
    def exact(self, xzero, t, normrvs):
        #TODO: aggregate over time for process with observations for all t
        #      i.e. exact conditional distribution for discrete time increment
        #      -> exactprocess
        #TODO: for single t, return stats.norm -> exactdist
        expnt = np.exp(-self.lambd * t)
        return (xzero * expnt + self.mu * (1-expnt) +
                self.sigma * np.sqrt((1-expnt*expnt)/2./self.lambd) * normrvs)

    def exactprocess(self, xzero, nobs, ddt=1., nrepl=2):
        '''ddt : discrete delta t

        should be the same as an AR(1)
        not tested yet
        # after writing this I saw the same use of lfilter in sitmo
        '''
        t = np.linspace(ddt, nobs*ddt, nobs)
        expnt = np.exp(-self.lambd * t)
        expddt = np.exp(-self.lambd * ddt)
        normrvs = np.random.normal(size=(nrepl,nobs))
        #do I need lfilter here AR(1) ? lfilter does not handle 2d arrays, it does?
        from scipy import signal
        #xzero * expnt
        inc = ( self.mu * (1-expddt) +
                self.sigma * np.sqrt((1-expddt*expddt)/2./self.lambd) * normrvs )

        return signal.lfilter([1.], [1.,-expddt], inc)


    def exactdist(self, xzero, t):
        #TODO: aggregate over time for process with observations for all t
        #TODO: for single t, return stats.norm
        expnt = np.exp(-self.lambd * t)
        meant = xzero * expnt + self.mu * (1-expnt)
        stdt = self.sigma * np.sqrt((1-expnt*expnt)/2./self.lambd)
        from scipy import stats
        return stats.norm(loc=meant, scale=stdt)

    def fitls(self, data, dt):
        '''assumes data is 1d, univariate time series
        formula from sitmo
        '''
        # brute force, no parameter estimation errors
        nobs = len(data)-1
        exog = np.column_stack((np.ones(nobs), data[:-1]))
        parest, res, rank, sing = np.linalg.lstsq(exog, data[1:], rcond=-1)
        const, slope = parest
        errvar = res/(nobs-2.)
        lambd = -np.log(slope)/dt
        sigma = np.sqrt(-errvar * 2.*np.log(slope)/ (1-slope**2)/dt)
        mu = const / (1-slope)
        return mu, lambd, sigma


class SchwartzOne(ExactDiffusion):
    '''the Schwartz type 1 stochastic process

    :math::
    dx_t = \\kappa (\\mu - \\ln x_t) x_t dt + \\sigma x_tdW \\

    The Schwartz type 1 process is a log of the Ornstein-Uhlenbeck stochastic
    process.

    '''

    def __init__(self, xzero, mu, kappa, sigma):
        self.xzero = xzero
        self.mu = mu
        self.kappa = kappa
        self.lambd = kappa #alias until I fix exact
        self.sigma = sigma

    def _exactconst(self, expnt):
        return (1-expnt) * (self.mu - self.sigma**2 / 2. /self.kappa)

    def _exactstd(self, expnt):
        return self.sigma * np.sqrt((1-expnt*expnt)/2./self.kappa)

    def exactprocess(self, xzero, nobs, ddt=1., nrepl=2):
        '''uses exact solution for log of process
        '''
        lnxzero = np.log(xzero)
        lnx = super(self.__class__, self).exactprocess(xzero, nobs, ddt=ddt, nrepl=nrepl)
        return np.exp(lnx)

    def exactdist(self, xzero, t):
        expnt = np.exp(-self.lambd * t)
        #TODO: check this is still wrong, just guessing
        meant = np.log(xzero) * expnt + self._exactconst(expnt)
        stdt = self._exactstd(expnt)
        return stats.lognorm(loc=meant, scale=stdt)

    def fitls(self, data, dt):
        '''assumes data is 1d, univariate time series
        formula from sitmo
        '''
        # brute force, no parameter estimation errors
        nobs = len(data)-1
        exog = np.column_stack((np.ones(nobs),np.log(data[:-1])))
        parest, res, rank, sing = np.linalg.lstsq(exog, np.log(data[1:]), rcond=-1)
        const, slope = parest
        errvar = res/(nobs-2.)  #check denominator estimate, of sigma too low
        kappa = -np.log(slope)/dt
        sigma = np.sqrt(errvar * kappa / (1-np.exp(-2*kappa*dt)))
        mu = const / (1-np.exp(-kappa*dt)) + sigma**2/2./kappa
        if np.shape(mu)== (1,):
            mu = mu[0]   # TODO: how to remove scalar array ?
        if np.shape(sigma)== (1,):
            sigma = sigma[0]
        #mu, kappa are good, sigma too small
        return mu, kappa, sigma



class BrownianBridge:
    def __init__(self):
        pass

    def simulate(self, x0, x1, nobs, nrepl=1, ddt=1., sigma=1.):
        nobs=nobs+1
        dt = ddt*1./nobs
        t = np.linspace(dt, ddt-dt, nobs)
        t = np.linspace(dt, ddt, nobs)
        wm = [t/ddt, 1-t/ddt]
        #wmi = wm[1]
        #wm1 = x1*wm[0]
        wmi = 1-dt/(ddt-t)
        wm1 = x1*(dt/(ddt-t))
        su = sigma* np.sqrt(t*(1-t)/ddt)
        s = sigma* np.sqrt(dt*(ddt-t-dt)/(ddt-t))
        x = np.zeros((nrepl, nobs))
        x[:,0] = x0
        rvs = s*np.random.normal(size=(nrepl,nobs))
        for i in range(1,nobs):
            x[:,i] = x[:,i-1]*wmi[i] + wm1[i] + rvs[:,i]
        return x, t, su


class CompoundPoisson:
    '''nobs iid compound poisson distributions, not a process in time
    '''
    def __init__(self, lambd, randfn=np.random.normal):
        if len(lambd) != len(randfn):
            raise ValueError('lambd and randfn need to have the same number of elements')

        self.nobj = len(lambd)
        self.randfn = randfn
        self.lambd = np.asarray(lambd)

    def simulate(self, nobs, nrepl=1):
        nobj = self.nobj
        x = np.zeros((nrepl, nobs, nobj))
        N = np.random.poisson(self.lambd[None,None,:], size=(nrepl,nobs,nobj))
        for io in range(nobj):
            randfnc = self.randfn[io]

            nc = N[:,:,io]
            #print nrepl,nobs,nc
            #xio = randfnc(size=(nrepl,nobs,np.max(nc))).cumsum(-1)[np.arange(nrepl)[:,None],np.arange(nobs),nc-1]
            rvs = randfnc(size=(nrepl,nobs,np.max(nc)))
            print('rvs.sum()', rvs.sum(), rvs.shape)
            xio = rvs.cumsum(-1)[np.arange(nrepl)[:,None],np.arange(nobs),nc-1]
            #print xio.shape
            x[:,:,io] = xio
        x[N==0] = 0
        return x, N









'''
randn('state',100)                                % set the state of randn
T = 1; N = 500; dt = T/N; t = [dt:dt:1];

M = 1000;                                         % M paths simultaneously
dW = sqrt(dt)*randn(M,N);                         % increments
W = cumsum(dW,2);                                 % cumulative sum
U = exp(repmat(t,[M 1]) + 0.5*W);
Umean = mean(U);
plot([0,t],[1,Umean],'b-'), hold on               % plot mean over M paths
plot([0,t],[ones(5,1),U(1:5,:)],'r--'), hold off  % plot 5 individual paths
xlabel('t','FontSize',16)
ylabel('U(t)','FontSize',16,'Rotation',0,'HorizontalAlignment','right')
legend('mean of 1000 paths','5 individual paths',2)

averr = norm((Umean - exp(9*t/8)),'inf')          % sample error
'''

if __name__ == '__main__':
    doplot = 1
    nrepl = 1000
    examples = []#['all']

    if 'all' in examples:
        w = Diffusion()

        # Wiener Process
        # ^^^^^^^^^^^^^^

        ws = w.simulateW(1000, nrepl=nrepl)
        if doplot:
            plt.figure()
            tmp = plt.plot(ws[0].T)
            tmp = plt.plot(ws[0].mean(0), linewidth=2)
            plt.title('Standard Brownian Motion (Wiener Process)')

        func = lambda t, W: np.exp(t + 0.5*W)
        us = w.expectedsim(func, nobs=500, nrepl=nrepl)
        if doplot:
            plt.figure()
            tmp = plt.plot(us[0].T)
            tmp = plt.plot(us[1], linewidth=2)
            plt.title('Brownian Motion - exp')
        #plt.show()
        averr = np.linalg.norm(us[1] - np.exp(9*us[2]/8.), np.inf)
        print(averr)
        #print us[1][:10]
        #print np.exp(9.*us[2][:10]/8.)

        # Geometric Brownian
        # ^^^^^^^^^^^^^^^^^^

        gb = GeometricBrownian(xzero=1., mu=0.01, sigma=0.5)
        gbs = gb.simEM(nobs=100, nrepl=100)
        if doplot:
            plt.figure()
            tmp = plt.plot(gbs.T)
            tmp = plt.plot(gbs.mean(0), linewidth=2)
            plt.title('Geometric Brownian')
            plt.figure()
            tmp = plt.plot(np.log(gbs).T)
            tmp = plt.plot(np.log(gbs.mean(0)), linewidth=2)
            plt.title('Geometric Brownian - log-transformed')

        ab = ArithmeticBrownian(xzero=1, mu=0.05, sigma=1)
        abs = ab.simEM(nobs=100, nrepl=100)
        if doplot:
            plt.figure()
            tmp = plt.plot(abs.T)
            tmp = plt.plot(abs.mean(0), linewidth=2)
            plt.title('Arithmetic Brownian')

        # Ornstein-Uhlenbeck
        # ^^^^^^^^^^^^^^^^^^

        ou = OUprocess(xzero=2, mu=1, lambd=0.5, sigma=0.1)
        ous = ou.simEM()
        oue = ou.exact(1, 1, np.random.normal(size=(5,10)))
        ou.exact(0, np.linspace(0,10,10/0.1), 0)
        ou.exactprocess(0,10)
        print(ou.exactprocess(0,10, ddt=0.1,nrepl=10).mean(0))
        #the following looks good, approaches mu
        oues = ou.exactprocess(0,100, ddt=0.1,nrepl=100)
        if doplot:
            plt.figure()
            tmp = plt.plot(oues.T)
            tmp = plt.plot(oues.mean(0), linewidth=2)
            plt.title('Ornstein-Uhlenbeck')

        # SchwartsOne
        # ^^^^^^^^^^^

        so = SchwartzOne(xzero=0, mu=1, kappa=0.5, sigma=0.1)
        sos = so.exactprocess(0,50, ddt=0.1,nrepl=100)
        print(sos.mean(0))
        print(np.log(sos.mean(0)))
        doplot = 1
        if doplot:
            plt.figure()
            tmp = plt.plot(sos.T)
            tmp = plt.plot(sos.mean(0), linewidth=2)
            plt.title('Schwartz One')
        print(so.fitls(sos[0,:],dt=0.1))
        sos2 = so.exactprocess(0,500, ddt=0.1,nrepl=5)
        print('true: mu=1, kappa=0.5, sigma=0.1')
        for i in range(5):
            print(so.fitls(sos2[i],dt=0.1))



        # Brownian Bridge
        # ^^^^^^^^^^^^^^^

        bb = BrownianBridge()
        #bbs = bb.sample(x0, x1, nobs, nrepl=1, ddt=1., sigma=1.)
        bbs, t, wm = bb.simulate(0, 0.5, 99, nrepl=500, ddt=1., sigma=0.1)
        if doplot:
            plt.figure()
            tmp = plt.plot(bbs.T)
            tmp = plt.plot(bbs.mean(0), linewidth=2)
            plt.title('Brownian Bridge')
            plt.figure()
            plt.plot(wm,'r', label='theoretical')
            plt.plot(bbs.std(0), label='simulated')
            plt.title('Brownian Bridge - Variance')
            plt.legend()

    # Compound Poisson
    # ^^^^^^^^^^^^^^^^
    cp = CompoundPoisson([1,1], [np.random.normal,np.random.normal])
    cps = cp.simulate(nobs=20000,nrepl=3)
    print(cps[0].sum(-1).sum(-1))
    print(cps[0].sum())
    print(cps[0].mean(-1).mean(-1))
    print(cps[0].mean())
    print(cps[1].size)
    print(cps[1].sum())
    #Note Y = sum^{N} X is compound poisson of iid x, then
    #E(Y) = E(N)*E(X)   eg. eq. (6.37) page 385 in http://ee.stanford.edu/~gray/sp.html


    #plt.show()
