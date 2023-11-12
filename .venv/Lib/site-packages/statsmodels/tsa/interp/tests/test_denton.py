import numpy as np
from statsmodels.tsa.interp import dentonm


def test_denton_quarterly():
    # Data and results taken from IMF paper
    indicator = np.array([98.2, 100.8, 102.2, 100.8, 99.0, 101.6,
                          102.7, 101.5, 100.5, 103.0, 103.5, 101.5])
    benchmark = np.array([4000.,4161.4])
    x_imf = dentonm(indicator, benchmark, freq="aq")
    imf_stata = np.array([969.8, 998.4, 1018.3, 1013.4, 1007.2, 1042.9,
                                1060.3, 1051.0, 1040.6, 1066.5, 1071.7, 1051.0])
    np.testing.assert_almost_equal(imf_stata, x_imf, 1)


def test_denton_quarterly2():
    # Test denton vs stata. Higher precision than other test.
    zQ = np.array([50,100,150,100] * 5)
    Y = np.array([500,400,300,400,500])
    x_denton = dentonm(zQ, Y, freq="aq")
    x_stata = np.array([64.334796,127.80616,187.82379,120.03526,56.563894,
                    105.97568,147.50144,89.958987,40.547201,74.445963,
                    108.34473,76.66211,42.763347,94.14664,153.41596,
                    109.67405,58.290761,122.62556,190.41409,128.66959])
    np.testing.assert_almost_equal(x_denton, x_stata, 5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
