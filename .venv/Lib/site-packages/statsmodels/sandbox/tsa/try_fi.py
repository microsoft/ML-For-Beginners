
'''
using lfilter to get fractional integration polynomial (1-L)^d, d<1
`ri` is (1-L)^(-d), d<1

second part in here is ar2arma

only examples left

'''
import numpy as np
from scipy.special import gamma
from scipy import signal

from statsmodels.tsa.arima_process import (lpol_fiar, lpol_fima,
                                           ar2arma, arma_impulse_response)



if __name__ == '__main__':
    d = 0.4
    n = 1000
    j = np.arange(n*10)
    ri0 = gamma(d+j)/(gamma(j+1)*gamma(d))
    ri = lpol_fima(d, n=n)  # get_ficoefs(d, n=n) old naming?
    riinv = signal.lfilter([1], ri, [1]+[0]*(n-1))#[[5,10,20,25]]
    '''
    array([-0.029952  , -0.01100641, -0.00410998, -0.00299859])
    >>> d=0.4; j=np.arange(1000);ri=gamma(d+j)/(gamma(j+1)*gamma(d))
    >>> # (1-L)^d, d<1 is
    >>> lfilter([1], ri, [1]+[0]*30)
    array([ 1.        , -0.4       , -0.12      , -0.064     , -0.0416    ,
          -0.029952  , -0.0229632 , -0.01837056, -0.01515571, -0.01279816,
          -0.01100641, -0.0096056 , -0.00848495, -0.00757118, -0.00681406,
          -0.00617808, -0.0056375 , -0.00517324, -0.00477087, -0.00441934,
          -0.00410998, -0.00383598, -0.00359188, -0.00337324, -0.00317647,
          -0.00299859, -0.00283712, -0.00269001, -0.00255551, -0.00243214,
          -0.00231864])
    >>> # verified for points [[5,10,20,25]] at 4 decimals with Bhardwaj, Swanson, Journal of Eonometrics 2006
    '''
    print(lpol_fiar(0.4, n=20))
    print(lpol_fima(-0.4, n=20))
    print(np.sum((lpol_fima(-0.4, n=n)[1:] + riinv[1:])**2)) #different signs
    print(np.sum((lpol_fiar(0.4, n=n)[1:] - riinv[1:])**2)) #corrected signs

    #test is now in statsmodels.tsa.tests.test_arima_process
    from statsmodels.tsa.tests.test_arima_process import test_fi
    test_fi()

    ar_true = [1, -0.4]
    ma_true = [1, 0.5]


    ar_desired = arma_impulse_response(ma_true, ar_true)
    ar_app, ma_app, res = ar2arma(ar_desired, 2,1, n=100, mse='ar', start=[0.1])
    print(ar_app, ma_app)
    ar_app, ma_app, res = ar2arma(ar_desired, 2,2, n=100, mse='ar', start=[-0.1, 0.1])
    print(ar_app, ma_app)
    ar_app, ma_app, res = ar2arma(ar_desired, 2,3, n=100, mse='ar')#, start = [-0.1, 0.1])
    print(ar_app, ma_app)

    slow = 1
    if slow:
        ar_desired = lpol_fiar(0.4, n=100)
        ar_app, ma_app, res = ar2arma(ar_desired, 3, 1, n=100, mse='ar')#, start = [-0.1, 0.1])
        print(ar_app, ma_app)
        ar_app, ma_app, res = ar2arma(ar_desired, 10, 10, n=100, mse='ar')#, start = [-0.1, 0.1])
        print(ar_app, ma_app)
