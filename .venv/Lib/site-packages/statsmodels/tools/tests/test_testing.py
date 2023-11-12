import pytest
import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch, \
    MarginTableTestBunch, Holder


@pytest.mark.parametrize('attribute, bunch_type',
                         (('params_table', ParamsTableTestBunch),
                          ('margins_table', MarginTableTestBunch)))
def check_params_table_classes(attribute, bunch_type):
    table = np.empty((10, 4))
    bunch = bunch_type(**{attribute: table})
    assert attribute in bunch


def test_bad_table():
    table = np.empty((10, 4))
    with pytest.raises(AttributeError):
        ParamsTableTestBunch(margins_table=table)


def test_holder():
    holder = Holder()
    holder.new_attr = 1
    assert hasattr(holder, 'new_attr')
    assert getattr(holder, 'new_attr') == 1
