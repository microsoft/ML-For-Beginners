# -*- coding: utf-8 -*-
"""

Created on Thu Feb 28 13:23:09 2013

Author: Josef Perktold
"""

import collections

from statsmodels.tools.testing import Holder


# numbers from R package `pwr` pwr.chisq.test
pwr_chisquare = collections.defaultdict(Holder)
pwr_chisquare[0].w = 1e-04
pwr_chisquare[0].N = 5
pwr_chisquare[0].df = 4
pwr_chisquare[0].sig_level = 0.05
pwr_chisquare[0].power = 0.05000000244872708
pwr_chisquare[0].method = 'Chi squared power calculation'
pwr_chisquare[0].note = 'N is the number of observations'
pwr_chisquare[1].w = 0.005
pwr_chisquare[1].N = 5
pwr_chisquare[1].df = 4
pwr_chisquare[1].sig_level = 0.05
pwr_chisquare[1].power = 0.05000612192891004
pwr_chisquare[1].method = 'Chi squared power calculation'
pwr_chisquare[1].note = 'N is the number of observations'
pwr_chisquare[2].w = 0.1
pwr_chisquare[2].N = 5
pwr_chisquare[2].df = 4
pwr_chisquare[2].sig_level = 0.05
pwr_chisquare[2].power = 0.05246644635810126
pwr_chisquare[2].method = 'Chi squared power calculation'
pwr_chisquare[2].note = 'N is the number of observations'
pwr_chisquare[3].w = 1
pwr_chisquare[3].N = 5
pwr_chisquare[3].df = 4
pwr_chisquare[3].sig_level = 0.05
pwr_chisquare[3].power = 0.396188517504065
pwr_chisquare[3].method = 'Chi squared power calculation'
pwr_chisquare[3].note = 'N is the number of observations'
pwr_chisquare[4].w = 1e-04
pwr_chisquare[4].N = 100
pwr_chisquare[4].df = 4
pwr_chisquare[4].sig_level = 0.05
pwr_chisquare[4].power = 0.05000004897454883
pwr_chisquare[4].method = 'Chi squared power calculation'
pwr_chisquare[4].note = 'N is the number of observations'
pwr_chisquare[5].w = 0.005
pwr_chisquare[5].N = 100
pwr_chisquare[5].df = 4
pwr_chisquare[5].sig_level = 0.05
pwr_chisquare[5].power = 0.05012248082672883
pwr_chisquare[5].method = 'Chi squared power calculation'
pwr_chisquare[5].note = 'N is the number of observations'
pwr_chisquare[6].w = 0.1
pwr_chisquare[6].N = 100
pwr_chisquare[6].df = 4
pwr_chisquare[6].sig_level = 0.05
pwr_chisquare[6].power = 0.1054845044462312
pwr_chisquare[6].method = 'Chi squared power calculation'
pwr_chisquare[6].note = 'N is the number of observations'
pwr_chisquare[7].w = 1
pwr_chisquare[7].N = 100
pwr_chisquare[7].df = 4
pwr_chisquare[7].sig_level = 0.05
pwr_chisquare[7].power = 0.999999999999644
pwr_chisquare[7].method = 'Chi squared power calculation'
pwr_chisquare[7].note = 'N is the number of observations'
pwr_chisquare[8].w = 1e-04
pwr_chisquare[8].N = 1000
pwr_chisquare[8].df = 4
pwr_chisquare[8].sig_level = 0.05
pwr_chisquare[8].power = 0.0500004897461283
pwr_chisquare[8].method = 'Chi squared power calculation'
pwr_chisquare[8].note = 'N is the number of observations'
pwr_chisquare[9].w = 0.005
pwr_chisquare[9].N = 1000
pwr_chisquare[9].df = 4
pwr_chisquare[9].sig_level = 0.05
pwr_chisquare[9].power = 0.0512288025485101
pwr_chisquare[9].method = 'Chi squared power calculation'
pwr_chisquare[9].note = 'N is the number of observations'
pwr_chisquare[10].w = 0.1
pwr_chisquare[10].N = 1000
pwr_chisquare[10].df = 4
pwr_chisquare[10].sig_level = 0.05
pwr_chisquare[10].power = 0.715986350467412
pwr_chisquare[10].method = 'Chi squared power calculation'
pwr_chisquare[10].note = 'N is the number of observations'
pwr_chisquare[11].w = 1
pwr_chisquare[11].N = 1000
pwr_chisquare[11].df = 4
pwr_chisquare[11].sig_level = 0.05
pwr_chisquare[11].power = 1
pwr_chisquare[11].method = 'Chi squared power calculation'
pwr_chisquare[11].note = 'N is the number of observations'
pwr_chisquare[12].w = 1e-04
pwr_chisquare[12].N = 30000
pwr_chisquare[12].df = 4
pwr_chisquare[12].sig_level = 0.05
pwr_chisquare[12].power = 0.05001469300301765
pwr_chisquare[12].method = 'Chi squared power calculation'
pwr_chisquare[12].note = 'N is the number of observations'
pwr_chisquare[13].w = 0.005
pwr_chisquare[13].N = 30000
pwr_chisquare[13].df = 4
pwr_chisquare[13].sig_level = 0.05
pwr_chisquare[13].power = 0.0904799545200348
pwr_chisquare[13].method = 'Chi squared power calculation'
pwr_chisquare[13].note = 'N is the number of observations'
pwr_chisquare[14].w = 0.1
pwr_chisquare[14].N = 30000
pwr_chisquare[14].df = 4
pwr_chisquare[14].sig_level = 0.05
pwr_chisquare[14].power = 1
pwr_chisquare[14].method = 'Chi squared power calculation'
pwr_chisquare[14].note = 'N is the number of observations'
pwr_chisquare[15].w = 1
pwr_chisquare[15].N = 30000
pwr_chisquare[15].df = 4
pwr_chisquare[15].sig_level = 0.05
pwr_chisquare[15].power = 1
pwr_chisquare[15].method = 'Chi squared power calculation'
pwr_chisquare[15].note = 'N is the number of observations'
