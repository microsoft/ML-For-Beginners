# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 12:26:38 2012
Author: Josef Perktold


function jc =  c_sja(n,p)
% PURPOSE: find critical values for Johansen maximum eigenvalue statistic
% ------------------------------------------------------------
% USAGE:  jc = c_sja(n,p)
% where:    n = dimension of the VAR system
%           p = order of time polynomial in the null-hypothesis
%                 p = -1, no deterministic part
%                 p =  0, for constant term
%                 p =  1, for constant plus time-trend
%                 p >  1  returns no critical values
% ------------------------------------------------------------
% RETURNS: a (3x1) vector of percentiles for the maximum eigenvalue
%          statistic for: [90% 95% 99%]
% ------------------------------------------------------------
% NOTES: for n > 12, the function returns a (3x1) vector of zeros.
%        The values returned by the function were generated using
%        a method described in MacKinnon (1996), using his FORTRAN
%        program johdist.f
% ------------------------------------------------------------
% SEE ALSO: johansen()
% ------------------------------------------------------------
% References: MacKinnon, Haug, Michelis (1996) 'Numerical distribution
% functions of likelihood ratio tests for cointegration',
% Queen's University Institute for Economic Research Discussion paper.
% -------------------------------------------------------
% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

"""

import numpy as np

ss_ejcp0 = '''\
         2.9762  4.1296  6.9406
         9.4748 11.2246 15.0923
        15.7175 17.7961 22.2519
        21.8370 24.1592 29.0609
        27.9160 30.4428 35.7359
        33.9271 36.6301 42.2333
        39.9085 42.7679 48.6606
        45.8930 48.8795 55.0335
        51.8528 54.9629 61.3449
        57.7954 61.0404 67.6415
        63.7248 67.0756 73.8856
        69.6513 73.0946 80.0937'''

ss_ejcp1 = '''\
         2.7055   3.8415   6.6349
        12.2971  14.2639  18.5200
        18.8928  21.1314  25.8650
        25.1236  27.5858  32.7172
        31.2379  33.8777  39.3693
        37.2786  40.0763  45.8662
        43.2947  46.2299  52.3069
        49.2855  52.3622  58.6634
        55.2412  58.4332  64.9960
        61.2041  64.5040  71.2525
        67.1307  70.5392  77.4877
        73.0563  76.5734  83.7105'''

ss_ejcp2 = '''\
         2.7055   3.8415   6.6349
        15.0006  17.1481  21.7465
        21.8731  24.2522  29.2631
        28.2398  30.8151  36.1930
        34.4202  37.1646  42.8612
        40.5244  43.4183  49.4095
        46.5583  49.5875  55.8171
        52.5858  55.7302  62.1741
        58.5316  61.8051  68.5030
        64.5292  67.9040  74.7434
        70.4630  73.9355  81.0678
        76.4081  79.9878  87.2395'''

ejcp0 = np.array(ss_ejcp0.split(),float).reshape(-1,3)
ejcp1 = np.array(ss_ejcp1.split(),float).reshape(-1,3)
ejcp2 = np.array(ss_ejcp2.split(),float).reshape(-1,3)


def c_sja(n, p):
    if ((p > 1) or (p < -1)):
        jc = np.full(3, np.nan)
    elif ((n > 12) or (n < 1)):
        jc = np.full(3, np.nan)
    elif p == -1:
        jc = ejcp0[n-1,:]
    elif p == 0:
        jc = ejcp1[n-1,:]
    elif p == 1:
        jc = ejcp2[n-1,:]

    return jc


'''
function jc = c_sjt(n,p)
% PURPOSE: find critical values for Johansen trace statistic
% ------------------------------------------------------------
% USAGE:  jc = c_sjt(n,p)
% where:    n = dimension of the VAR system
%               NOTE: routine does not work for n > 12
%           p = order of time polynomial in the null-hypothesis
%                 p = -1, no deterministic part
%                 p =  0, for constant term
%                 p =  1, for constant plus time-trend
%                 p >  1  returns no critical values
% ------------------------------------------------------------
% RETURNS: a (3x1) vector of percentiles for the trace
%          statistic for [90% 95% 99%]
% ------------------------------------------------------------
% NOTES: for n > 12, the function returns a (3x1) vector of zeros.
%        The values returned by the function were generated using
%        a method described in MacKinnon (1996), using his FORTRAN
%        program johdist.f
% ------------------------------------------------------------
% SEE ALSO: johansen()
% ------------------------------------------------------------
% % References: MacKinnon, Haug, Michelis (1996) 'Numerical distribution
% functions of likelihood ratio tests for cointegration',
% Queen's University Institute for Economic Research Discussion paper.
% -------------------------------------------------------
% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com
% these are the values from Johansen's 1995 book
% for comparison to the MacKinnon values
%jcp0 = [ 2.98   4.14   7.02
%        10.35  12.21  16.16
%        21.58  24.08  29.19
%        36.58  39.71  46.00
%        55.54  59.24  66.71
%        78.30  86.36  91.12
%       104.93 109.93 119.58
%       135.16 140.74 151.70
%       169.30 175.47 187.82
%       207.21 214.07 226.95
%       248.77 256.23 270.47
%       293.83 301.95 318.14];
%
'''


ss_tjcp0 = '''\
         2.9762   4.1296   6.9406
        10.4741  12.3212  16.3640
        21.7781  24.2761  29.5147
        37.0339  40.1749  46.5716
        56.2839  60.0627  67.6367
        79.5329  83.9383  92.7136
       106.7351 111.7797 121.7375
       137.9954 143.6691 154.7977
       173.2292 179.5199 191.8122
       212.4721 219.4051 232.8291
       255.6732 263.2603 277.9962
       302.9054 311.1288 326.9716'''


ss_tjcp1 = '''\
          2.7055   3.8415   6.6349
         13.4294  15.4943  19.9349
         27.0669  29.7961  35.4628
         44.4929  47.8545  54.6815
         65.8202  69.8189  77.8202
         91.1090  95.7542 104.9637
        120.3673 125.6185 135.9825
        153.6341 159.5290 171.0905
        190.8714 197.3772 210.0366
        232.1030 239.2468 253.2526
        277.3740 285.1402 300.2821
        326.5354 334.9795 351.2150'''

ss_tjcp2 = '''\
           2.7055   3.8415   6.6349
          16.1619  18.3985  23.1485
          32.0645  35.0116  41.0815
          51.6492  55.2459  62.5202
          75.1027  79.3422  87.7748
         102.4674 107.3429 116.9829
         133.7852 139.2780 150.0778
         169.0618 175.1584 187.1891
         208.3582 215.1268 228.2226
         251.6293 259.0267 273.3838
         298.8836 306.8988 322.4264
         350.1125 358.7190 375.3203'''

tjcp0 = np.array(ss_tjcp0.split(),float).reshape(-1,3)
tjcp1 = np.array(ss_tjcp1.split(),float).reshape(-1,3)
tjcp2 = np.array(ss_tjcp2.split(),float).reshape(-1,3)


def c_sjt(n, p):
    if ((p > 1) or (p < -1)):
        jc = np.full(3, np.nan)
    elif ((n > 12) or (n < 1)):
        jc = np.full(3, np.nan)
    elif p == -1:
        jc = tjcp0[n-1,:]
    elif p == 0:
        jc = tjcp1[n-1,:]
    elif p == 1:
        jc = tjcp2[n-1,:]
    else:
        raise ValueError('invalid p')

    return jc


if __name__ == '__main__':
    for p in range(-2, 3, 1):
        for n in range(12):
            print(n, p)
            print(c_sja(n, p))
            print(c_sjt(n, p))
