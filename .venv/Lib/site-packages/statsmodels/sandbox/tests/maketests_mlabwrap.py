'''generate py modules with test cases and results from mlabwrap

currently matlab: princomp, garchar, garchma
'''

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import array



xo = array([[ -419,  -731, -1306, -1294],
       [    6,   529,  -200,  -437],
       [  -27,  -833,    -6,  -564],
       [ -304,  -273,  -502,  -739],
       [ 1377,  -912,   927,   280],
       [ -375,  -517,  -514,    49],
       [  247,  -504,   123,  -259],
       [  712,   534,  -773,   286],
       [  195, -1080,  3256,  -178],
       [ -854,    75,  -706, -1084],
       [-1219,  -612,   -15,  -203],
       [  550,  -628,  -483, -2686],
       [ -365,  1376, -1266,   317],
       [ -489,   544,  -195,   431],
       [ -656,   854,   840,  -723],
       [   16, -1385,  -880,  -460],
       [  258, -2252,    96,    54],
       [ 2049,  -750, -1115,   381],
       [  -65,   280,  -777,   416],
       [  755,    82,  -806,  1027],
       [  -39,  -170, -2134,   743],
       [ -859,   780,   746,  -133],
       [  762,   252,  -450,  -459],
       [ -941,  -202,    49,  -202],
       [  -54,   115,   455,   388],
       [-1348,  1246,  1430,  -480],
       [  229,  -535, -1831,  1524],
       [ -651,  -167,  2116,   483],
       [-1249, -1373,   888, -1092],
       [  -75, -2162,   486,  -496],
       [ 2436, -1627, -1069,   162],
       [  -63,   560,  -601,   587],
       [  -60,  1051,  -277,  1323],
       [ 1329, -1294,    68,     5],
       [ 1532,  -633,  -923,   696],
       [  669,   895, -1762,  -375],
       [ 1129,  -548,  2064,   609],
       [ 1320,   573,  2119,   270],
       [ -213,  -412, -2517,  1685],
       [   73,  -979,  1312, -1220],
       [-1360, -2107,  -237,  1522],
       [ -645,   205,  -543,  -169],
       [ -212,  1072,   543,  -128],
       [ -352,  -129,  -605,  -904],
       [  511,    85,   167, -1914],
       [ 1515,  1862,   942,  1622],
       [ -465,   623,  -495,   -89],
       [-1396,  -979,  1758,   128],
       [ -255,   -47,   980,   501],
       [-1282,   -58,   -49,  -610],
       [ -889, -1177,  -492,   494],
       [ 1415,  1146,   696,  -722],
       [ 1237,  -224, -1609,   -64],
       [ -528, -1625,   231,   883],
       [ -327,  1636,  -476,  -361],
       [ -781,   793,  1882,   234],
       [ -506,  -561,  1988,  -810],
       [-1233,  1467,  -261,  2164],
       [   53,  1069,   824,  2123],
       [-1200,  -441,  -321,   339],
       [ 1606,   298,  -995,  1292],
       [-1740,  -672, -1628,  -129],
       [-1450,  -354,   224,  -657],
       [-2556,  1006,  -706, -1453],
       [ -717,  -463,   345, -1821],
       [ 1056,   -38,  -420,  -455],
       [ -523,   565,   425,  1138],
       [-1030,  -187,   683,    78],
       [ -214,  -312, -1171,  -528],
       [  819,   736,  -265,   423],
       [ 1339,   351,  1142,   579],
       [ -387,  -126, -1573,  2346],
       [  969,     2,   327,  -134],
       [  163,   227,    90,  2021],
       [ 1022, -1076,   174,   304],
       [ 1042,  1317,   311,   880],
       [ 2018,  -840,   295,  2651],
       [ -277,   566,  1147,  -189],
       [   20,   467,  1262,   263],
       [ -663,  1061, -1552, -1159],
       [ 1830,   391,  2534,  -199],
       [ -487,   752, -1061,   351],
       [-2138,  -556,  -367,  -457],
       [ -868,  -411,  -559,   726],
       [ 1770,   819,  -892,  -363],
       [  553,  -736,  -169,  -490],
       [  388,  -503,   809,  -821],
       [ -516, -1452,  -192,   483],
       [  493,  2904,  1318,  2591],
       [  175,   584, -1001,  1675],
       [ 1316, -1596,  -460,  1500],
       [ 1212,   214,  -644,  -696],
       [ -501,   338,  1197,  -841],
       [ -587,  -469, -1101,    24],
       [-1205,  1910,   659,  1232],
       [ -150,   398,   594,   394],
       [   34,  -663,   235,  -334],
       [-1580,   647,   239,  -351],
       [-2177,  -345,  1215, -1494],
       [ 1923,   329,  -152,  1128]])

x = xo/1000.

class HoldIt:
    def __init__(self, name):
        self.name = name
    def save(self, what=None, filename=None, header=True, useinstant=True,
             comment=None):
        if what is None:
            what = (i for i in self.__dict__ if i[0] != '_')
        if header:
            txt = ['import numpy as np\nfrom numpy import array\n\n']
            if useinstant:
                txt.append('from statsmodels.tools.testing import Holder\n\n')
        else:
            txt = []

        if useinstant:
            txt.append('%s = Holder()' % self.name)
            prefix = '%s.' % self.name
        else:
            prefix = ''

        if comment is not None:
            txt.append("%scomment = '%s'" % (prefix, comment))

        for x in what:
            txt.append('%s%s = %s' % (prefix, x, repr(getattr(self,x))))
        txt.extend(['','']) #add empty lines at end
        if filename is not None:
            with open(filename, 'a+', encoding="utf-8") as fd:
                fd.write('\n'.join(txt))
        return txt

def generate_princomp(xo, filen='testsave.py'):
    # import mlabwrap only when run as script
    from mlabwrap import mlab
    np.set_printoptions(precision=14, linewidth=100)
    data =  HoldIt('data')
    data.xo = xo
    data.save(filename='testsave.py', comment='generated data, divide by 1000')

    res_princomp =  HoldIt('princomp1')
    res_princomp.coef, res_princomp.factors, res_princomp.values = \
                       mlab.princomp(x, nout=3)
    res_princomp.save(filename=filen, header=False,
                      comment='mlab.princomp(x, nout=3)')

    res_princomp =  HoldIt('princomp2')
    res_princomp.coef, res_princomp.factors, res_princomp.values = \
                       mlab.princomp(x[:20,], nout=3)
    np.set_printoptions(precision=14, linewidth=100)
    res_princomp.save(filename=filen, header=False,
                      comment='mlab.princomp(x[:20,], nout=3)')

    res_princomp =  HoldIt('princomp3')
    res_princomp.coef, res_princomp.factors, res_princomp.values = \
                       mlab.princomp(x[:20,]-x[:20,].mean(0), nout=3)
    np.set_printoptions(precision=14, linewidth=100)
    res_princomp.save(filename=filen, header=False,
                      comment='mlab.princomp(x[:20,]-x[:20,].mean(0), nout=3)')

def generate_armarep(filen='testsave.py'):
    # import mlabwrap only when run as script
    from mlabwrap import mlab
    res_armarep =  HoldIt('armarep')
    res_armarep.ar = np.array([1.,  -0.5, +0.8])
    res_armarep.ma = np.array([1., -0.6,  0.08])

    res_armarep.marep = mlab.garchma(-res_armarep.ar[1:], res_armarep.ma[1:], 20)
    res_armarep.arrep = mlab.garchar(-res_armarep.ar[1:], res_armarep.ma[1:], 20)
    res_armarep.save(filename=filen, header=False,
            comment=("''mlab.garchma(-res_armarep.ar[1:], res_armarep.ma[1:], 20)\n" +
                     "mlab.garchar(-res_armarep.ar[1:], res_armarep.ma[1:], 20)''"))





def exampletest(res_armarep):
    from statsmodels.sandbox import tsa
    arrep = tsa.arma_impulse_response(res_armarep.ma, res_armarep.ar, nobs=21)[1:]
    marep = tsa.arma_impulse_response(res_armarep.ar, res_armarep.ma, nobs=21)[1:]
    assert_array_almost_equal(res_armarep.marep.ravel(), marep, 14)
    #difference in sign convention to matlab for AR term
    assert_array_almost_equal(-res_armarep.arrep.ravel(), arrep, 14)


if __name__ == '__main__':
    from mlabwrap import mlab

    import savedrvs
    xo = savedrvs.rvsdata.xar2
    x100 = xo[-100:]/1000.
    x1000 = xo/1000.

    filen = 'testsavetls.py'
    res_pacf =  HoldIt('mlpacf')
    res_pacf.comment = 'mlab.parcorr(x, [], 2, nout=3)'
    res_pacf.pacf100, res_pacf.lags100, res_pacf.bounds100 = \
                      mlab.parcorr(x100, [], 2, nout=3)
    res_pacf.pacf1000, res_pacf.lags1000, res_pacf.bounds1000 = \
                      mlab.parcorr(x1000, [], 2, nout=3)
    res_pacf.save(filename=filen, header=True)

    res_acf =  HoldIt('mlacf')
    res_acf.comment = 'mlab.autocorr(x, [], 2, nout=3)'
    res_acf.acf100, res_acf.lags100, res_acf.bounds100 = \
                    mlab.autocorr(x100, [], 2, nout=3)
    res_acf.acf1000, res_acf.lags1000, res_acf.bounds1000 = \
                    mlab.autocorr(x1000, [], 2, nout=3)
    res_acf.save(filename=filen, header=False)


    res_ccf =  HoldIt('mlccf')
    res_ccf.comment = 'mlab.crosscorr(x[4:], x[:-4], [], 2, nout=3)'
    res_ccf.ccf100, res_ccf.lags100, res_ccf.bounds100 = \
                 mlab.crosscorr(x100[4:], x100[:-4], [], 2, nout=3)
    res_ccf.ccf1000, res_ccf.lags1000, res_ccf.bounds1000 = \
                 mlab.crosscorr(x1000[4:], x1000[:-4], [], 2, nout=3)
    res_ccf.save(filename=filen, header=False)


    res_ywar =  HoldIt('mlywar')
    res_ywar.comment = "mlab.ar(x100-x100.mean(), 10, 'yw').a.ravel()"
    mbaryw = mlab.ar(x100-x100.mean(), 10, 'yw')
    res_ywar.arcoef100 = np.array(mbaryw.a.ravel())
    mbaryw = mlab.ar(x1000-x1000.mean(), 20, 'yw')
    res_ywar.arcoef1000 = np.array(mbaryw.a.ravel())
    res_ywar.save(filename=filen, header=False)
