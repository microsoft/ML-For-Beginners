import pytest
from numpy import array

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.web import _generate_url, webdoc


class TestWeb:
    stable = 'https://www.statsmodels.org/stable/'
    devel = 'https://www.statsmodels.org/devel/'

    def test_string(self):
        url = _generate_url('arch', True)
        assert url == self.stable + 'search.html?q=' \
                                    'arch&check_keywords=yes&area=default'
        url = _generate_url('arch', False)
        assert url == self.devel + 'search.html?q=' \
                                   'arch&check_keywords=yes&area=default'
        url = _generate_url('dickey fuller', False)
        assert url == (self.devel +
                       'search.html?q='
                       'dickey+fuller&check_keywords=yes&area=default')

    def test_function(self):
        url = _generate_url(OLS, True)
        assert url == (self.stable
                       + 'generated/'
                         'statsmodels.regression.linear_model.OLS.html')
        url = _generate_url(OLS, False)
        assert url == (self.devel
                       + 'generated/'
                         'statsmodels.regression.linear_model.OLS.html')

    def test_nothing(self):
        url = _generate_url(None, True)
        assert url == 'https://www.statsmodels.org/stable/'
        url = _generate_url(None, False)
        assert url == 'https://www.statsmodels.org/devel/'

    def test_errors(self):
        with pytest.raises(ValueError):
            webdoc(array, True)
        with pytest.raises(ValueError):
            webdoc(1, False)
