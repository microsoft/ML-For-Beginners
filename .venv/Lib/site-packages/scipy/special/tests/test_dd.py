# Tests for a few of the "double-double" C functions defined in cephes/dd_*.

import pytest
from numpy.testing import assert_allclose
from scipy.special._test_internal import _dd_exp, _dd_log, _dd_expm1


# Each tuple in test_data contains:
#   (dd_func, xhi, xlo, expected_yhi, expected_ylo)
# The expected values were computed with mpmath, e.g.
#
#   import mpmath
#   mpmath.mp.dps = 100
#   xhi = 10.0
#   xlo = 0.0
#   x = mpmath.mpf(xhi) + mpmath.mpf(xlo)
#   y = mpmath.log(x)
#   expected_yhi = float(y)
#   expected_ylo = float(y - expected_yhi)
#
test_data = [
    (_dd_exp, -0.3333333333333333, -1.850371707708594e-17,
     0.7165313105737893, -2.0286948382455594e-17),
    (_dd_exp, 0.0, 0.0, 1.0, 0.0),
    (_dd_exp, 10.0, 0.0, 22026.465794806718, -1.3780134700517372e-12),
    (_dd_log, 0.03125, 0.0, -3.4657359027997265, -4.930038229799327e-18),
    (_dd_log, 10.0, 0.0, 2.302585092994046, -2.1707562233822494e-16),
    (_dd_expm1, -1.25, 0.0, -0.7134952031398099, -4.7031321153650186e-17),
    (_dd_expm1, -0.484375, 0.0, -0.3839178722093218, 7.609376052156984e-18),
    (_dd_expm1, -0.25, 0.0, -0.22119921692859512, -1.0231869534531498e-17),
    (_dd_expm1, -0.0625, 0.0, -0.06058693718652421, -7.077887227488846e-19),
    (_dd_expm1, 0.0, 0.0, 0.0, 0.0),
    (_dd_expm1, 0.0625, 3.5e-18, 0.06449445891785943, 1.4323095758164254e-18),
    (_dd_expm1, 0.25, 0.0, 0.2840254166877415, -2.133257464457841e-17),
    (_dd_expm1, 0.498046875, 0.0, 0.645504254608231, -9.198435524984236e-18),
    (_dd_expm1, 1.25, 0.0, 2.4903429574618414, -4.604261945372796e-17)
]


@pytest.mark.parametrize('dd_func, xhi, xlo, expected_yhi, expected_ylo',
                         test_data)
def test_dd(dd_func, xhi, xlo, expected_yhi, expected_ylo):
    yhi, ylo = dd_func(xhi, xlo)
    assert yhi == expected_yhi, (f"high double ({yhi}) does not equal the "
                                 f"expected value {expected_yhi}")
    assert_allclose(ylo, expected_ylo, rtol=5e-15)
