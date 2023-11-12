"""
Tests of save / load / remove_data state space functionality.
"""
import pickle
import os
import tempfile

import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
                                        dynamic_factor)
from numpy.testing import assert_allclose

current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data


@pytest.fixture()
def temp_filename():
    fd, filename = tempfile.mkstemp()
    yield filename
    try:
        os.close(fd)
        os.unlink(filename)
    except Exception:
        print("Couldn't close or delete file "
              "{filename}.".format(filename=filename))


def test_sarimax(temp_filename):
    mod = sarimax.SARIMAX(macrodata['realgdp'].values, order=(4, 1, 0))
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename)
    res2 = sarimax.SARIMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


# GH7527
@pytest.mark.parametrize("order", ((4, 1, 0), (0, 1, 4), (0, 2, 0)))
def test_sarimax_save_remove_data(temp_filename, order):
    mod = sarimax.SARIMAX(macrodata["realgdp"].values, order=order)
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename, remove_data=True)
    res2 = sarimax.SARIMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_sarimax_pickle():
    mod = sarimax.SARIMAX(macrodata['realgdp'].values, order=(4, 1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(mod.start_params)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)


def test_structural(temp_filename):
    mod = structural.UnobservedComponents(
        macrodata['realgdp'].values, 'llevel')
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename)
    res2 = structural.UnobservedComponentsResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_structural_pickle():
    mod = structural.UnobservedComponents(
        macrodata['realgdp'].values, 'llevel')
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(pkl_mod.start_params)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)


def test_dynamic_factor(temp_filename):
    mod = dynamic_factor.DynamicFactor(
        macrodata[['realgdp', 'realcons']].diff().iloc[1:].values, k_factors=1,
        factor_order=1)
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename)
    res2 = dynamic_factor.DynamicFactorResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_dynamic_factor_pickle(temp_filename):
    mod = varmax.VARMAX(
        macrodata[['realgdp', 'realcons']].diff().iloc[1:].values,
        order=(1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(mod.start_params)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)

    res.summary()
    res.save(temp_filename)
    res2 = varmax.VARMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_varmax(temp_filename):
    mod = varmax.VARMAX(
        macrodata[['realgdp', 'realcons']].diff().iloc[1:].values,
        order=(1, 0))
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename)
    res2 = varmax.VARMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_varmax_pickle(temp_filename):
    mod = varmax.VARMAX(
        macrodata[['realgdp', 'realcons']].diff().iloc[1:].values,
        order=(1, 0))
    res = mod.smooth(mod.start_params)

    res.summary()
    res.save(temp_filename)
    res2 = varmax.VARMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_existing_pickle():
    pkl_file = os.path.join(current_path, 'results', 'sm-0.9-sarimax.pkl')
    loaded = sarimax.SARIMAXResults.load(pkl_file)
    assert isinstance(loaded, sarimax.SARIMAXResultsWrapper)
