import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
)
import pytest

import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
    dt_s_tup_to_string,
    load_results_jmulti,
    sublists,
)
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
    VECM,
    select_coint_rank,
    select_order,
)

pytestmark = pytest.mark.filterwarnings("ignore:in the future np.array_split")


class DataSet:
    """
    A class for representing the data in a data module.

    Parameters
    ----------
    data_module : module
        A module contained in the statsmodels/datasets directory.
    n_seasons : list
        A list of integers. Each int represents a number of seasons to test the
        model with.
    first_season : list
        A list of integers. Each int corresponds to the int with the same index
        in the n_seasons list and declares to which season the first data entry
        belongs.
    variable_names : list
        A list of strings. Each string names a variable.
    """

    def __init__(self, data_module, n_seasons, first_season, variable_names):
        self.data_module = data_module
        self.seasons = n_seasons
        self.first_season = first_season
        self.dt_s_list = [
            (det, s, self.first_season[i])
            for det in deterministic_terms_list
            for i, s in enumerate(self.seasons)
        ]
        self.variable_names = variable_names

    def __str__(self):
        return self.data_module.__str__()


atol = 0.0005  # absolute tolerance
rtol = 0  # relative tolerance
datasets = []
data = {}
results_ref = {}
results_sm = {}
results_sm_exog = {}
results_sm_exog_coint = {}
coint_rank = 1

debug_mode = False
dont_test_se_t_p = False
deterministic_terms_list = ["nc", "co", "colo", "ci", "cili"]

all_tests = [
    "Gamma",
    "alpha",
    "beta",
    "C",
    "det_coint",
    "Sigma_u",
    "VAR repr. A",
    "VAR to VEC representation",
    "log_like",
    "fc",
    "granger",
    "inst. causality",
    "impulse-response",
    "lag order",
    "test_norm",
    "whiteness",
    "summary",
    "exceptions",
    "select_coint_rank",
]
to_test = all_tests  # ["beta"]


def load_data(dataset, data_dict):
    dtset = dataset.data_module.load_pandas()
    variables = dataset.variable_names
    loaded = dtset.data[variables].astype(float).values
    data_dict[dataset] = loaded.reshape((-1, len(variables)))


def load_results_statsmodels(dataset):
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)

    for dt_s_tup in dataset.dt_s_list:
        model = VECM(
            data[dataset],
            k_ar_diff=3,
            coint_rank=coint_rank,
            deterministic=dt_s_tup[0],
            seasons=dt_s_tup[1],
            first_season=dt_s_tup[2],
        )
        results_per_deterministic_terms[dt_s_tup] = model.fit(method="ml")
    return results_per_deterministic_terms


def load_results_statsmodels_exog(dataset):
    """
    Load data with seasonal terms in `exog`.

    Same as load_results_statsmodels() except that the seasonal term is
    provided to :class:`VECM`'s `__init__()` method via the `eoxg` parameter.
    This is to check whether the same results are produced no matter whether
    `exog` or `seasons` is being used.

    Parameters
    ----------
    dataset : DataSet
    """
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    endog = data[dataset]
    for dt_s_tup in dataset.dt_s_list:
        det_string = dt_s_tup[0]
        seasons = dt_s_tup[1]
        first_season = dt_s_tup[2]
        if seasons == 0:
            exog = None
        else:
            exog = seasonal_dummies(
                seasons, len(data[dataset]), first_season, centered=True
            )
            if "lo" in dt_s_tup[0]:
                exog = np.hstack(
                    (exog, 1 + np.arange(len(endog)).reshape(-1, 1))
                )
                # remove "lo" since it's now already in exog.
                det_string = det_string[:-2]
        model = VECM(
            endog,
            exog,
            k_ar_diff=3,
            coint_rank=coint_rank,
            deterministic=det_string,
        )
        results_per_deterministic_terms[dt_s_tup] = model.fit(method="ml")
    return results_per_deterministic_terms


def load_results_statsmodels_exog_coint(dataset):
    """
    Load data with deterministic terms in `exog_coint`.

    Same as load_results_statsmodels() except that deterministic terms inside
    the cointegration relation are provided to :class:`VECM`'s `__init__()`
    method via the `eoxg_coint` parameter. This is to check whether the same
    results are produced no matter whether `exog_coint` or the `deterministic`
    argument is being used.

    Parameters
    ----------
    dataset : DataSet
    """
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    endog = data[dataset]
    for dt_s_tup in dataset.dt_s_list:
        det_string = dt_s_tup[0]

        if "ci" not in det_string and "li" not in det_string:
            exog_coint = None
        else:
            exog_coint = []
            if "li" in det_string:
                exog_coint.append(1 + np.arange(len(endog)).reshape(-1, 1))
                det_string = det_string[:-2]
            if "ci" in det_string:
                exog_coint.append(np.ones(len(endog)).reshape(-1, 1))
                det_string = det_string[:-2]
            # reversing (such that constant is first and linear is second)
            exog_coint = exog_coint[::-1]
            exog_coint = np.hstack(exog_coint)
        model = VECM(
            endog,
            exog=None,
            exog_coint=exog_coint,
            k_ar_diff=3,
            coint_rank=coint_rank,
            deterministic=det_string,
            seasons=dt_s_tup[1],
            first_season=dt_s_tup[2],
        )
        results_per_deterministic_terms[dt_s_tup] = model.fit(method="ml")
    return results_per_deterministic_terms


def build_err_msg(ds, dt_s, parameter_str):
    dt = dt_s_tup_to_string(dt_s)
    seasons = dt_s[1]
    err_msg = "Error in " + parameter_str + " for:\n"
    err_msg += "- Dataset: " + ds.__str__() + "\n"
    err_msg += "- Deterministic terms: "
    err_msg += dt_s[0] if dt != "n" else "no det. terms"
    if seasons > 0:
        err_msg += ", seasons: " + str(seasons)
    return err_msg


def setup():
    datasets.append(
        DataSet(e6, [0, 4], [0, 1], ["Dp", "R"]),
        # DataSet(...) TODO: append more data sets for more test cases.
    )

    for ds in datasets:
        load_data(ds, data)
        results_ref[ds] = load_results_jmulti(ds)
        results_sm[ds] = load_results_statsmodels(ds)
        results_sm_exog[ds] = load_results_statsmodels_exog(ds)
        results_sm_exog_coint[ds] = load_results_statsmodels_exog_coint(ds)


setup()


def test_ml_gamma():
    if debug_mode:
        if "Gamma" not in to_test:  # pragma: no cover
            return
        print("\n\nGAMMA", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            # estimated parameter vector
            err_msg = build_err_msg(ds, dt, "Gamma")
            obtained = results_sm[ds][dt].gamma
            obtained_exog = results_sm_exog[ds][dt].gamma
            obtained_exog_coint = results_sm_exog_coint[ds][dt].gamma
            desired = results_ref[ds][dt]["est"]["Gamma"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint,
                    obtained,
                    "WITH EXOG_COINT: " + err_msg,
                )

            if debug_mode and dont_test_se_t_p:  # pragma: no cover
                continue
            # standard errors
            obt = results_sm[ds][dt].stderr_gamma
            obt_exog = results_sm_exog[ds][dt].stderr_gamma
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_gamma
            des = results_ref[ds][dt]["se"]["Gamma"]
            assert_allclose(
                obt, des, rtol, atol, False, "STANDARD ERRORS\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )
            # t-values
            obt = results_sm[ds][dt].tvalues_gamma
            obt_exog = results_sm_exog[ds][dt].tvalues_gamma
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_gamma
            des = results_ref[ds][dt]["t"]["Gamma"]
            assert_allclose(
                obt, des, rtol, atol, False, "t-VALUES\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )
            # p-values
            obt = results_sm[ds][dt].pvalues_gamma
            obt_exog = results_sm_exog[ds][dt].pvalues_gamma
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_gamma
            des = results_ref[ds][dt]["p"]["Gamma"]
            assert_allclose(
                obt, des, rtol, atol, False, "p-VALUES\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )


def test_ml_alpha():
    if debug_mode:
        if "alpha" not in to_test:  # pragma: no cover
            return
        print("\n\nALPHA", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "alpha")
            obtained = results_sm[ds][dt].alpha
            obtained_exog = results_sm_exog[ds][dt].alpha
            obtained_exog_coint = results_sm_exog_coint[ds][dt].alpha
            desired = results_ref[ds][dt]["est"]["alpha"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint,
                    obtained,
                    "WITH EXOG_COINT: " + err_msg,
                )

            if debug_mode and dont_test_se_t_p:  # pragma: no cover
                continue
            # standard errors
            obt = results_sm[ds][dt].stderr_alpha
            obt_exog = results_sm_exog[ds][dt].stderr_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_alpha
            des = results_ref[ds][dt]["se"]["alpha"]
            assert_allclose(
                obt, des, rtol, atol, False, "STANDARD ERRORS\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )
            # t-values
            obt = results_sm[ds][dt].tvalues_alpha
            obt_exog = results_sm_exog[ds][dt].tvalues_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_alpha
            des = results_ref[ds][dt]["t"]["alpha"]
            assert_allclose(
                obt, des, rtol, atol, False, "t-VALUES\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )
            # p-values
            obt = results_sm[ds][dt].pvalues_alpha
            obt_exog = results_sm_exog[ds][dt].pvalues_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_alpha
            des = results_ref[ds][dt]["p"]["alpha"]
            assert_allclose(
                obt, des, rtol, atol, False, "p-VALUES\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )


def test_ml_beta():
    if debug_mode:
        if "beta" not in to_test:  # pragma: no cover
            return
        print("\n\nBETA", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "beta")
            desired = results_ref[ds][dt]["est"]["beta"]
            rows = desired.shape[0]
            # - first coint_rank rows in JMulTi output have se=t_val=p_val=0
            # - beta includes deterministic terms in cointegration relation in
            #   sm, so we compare only the elements belonging to beta.
            obtained = results_sm[ds][dt].beta[coint_rank:rows]
            obtained_exog = results_sm_exog[ds][dt].beta[coint_rank:rows]
            obtained_exog_coint = results_sm_exog_coint[ds][dt].beta[
                coint_rank:rows
            ]
            desired = desired[coint_rank:]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint,
                    obtained,
                    "WITH EXOG_COINT: " + err_msg,
                )

            if debug_mode and dont_test_se_t_p:  # pragma: no cover
                continue
            # standard errors
            obt = results_sm[ds][dt].stderr_beta[coint_rank:rows]
            obt_exog = results_sm_exog[ds][dt].stderr_beta[coint_rank:rows]
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_beta[
                coint_rank:rows
            ]
            des = results_ref[ds][dt]["se"]["beta"][coint_rank:]
            assert_allclose(
                obt, des, rtol, atol, False, "STANDARD ERRORS\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )
            # t-values
            obt = results_sm[ds][dt].tvalues_beta[coint_rank:rows]
            obt_exog = results_sm_exog[ds][dt].tvalues_beta[coint_rank:rows]
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_beta[
                coint_rank:rows
            ]
            des = results_ref[ds][dt]["t"]["beta"][coint_rank:]
            assert_allclose(
                obt, des, rtol, atol, False, "t-VALUES\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )
            # p-values
            obt = results_sm[ds][dt].pvalues_beta[coint_rank:rows]
            obt_exog = results_sm_exog[ds][dt].pvalues_beta[coint_rank:rows]
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_beta[
                coint_rank:rows
            ]
            des = results_ref[ds][dt]["p"]["beta"][coint_rank:]
            assert_allclose(
                obt, des, rtol, atol, False, "p-VALUES\n" + err_msg
            )
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg
                )


def test_ml_c():  # test deterministic terms outside coint relation
    if debug_mode:
        if "C" not in to_test:  # pragma: no cover
            return
        print("\n\nDET_COEF", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            C_obt = results_sm[ds][dt].det_coef
            C_obt_exog = results_sm_exog[ds][dt].det_coef
            C_obt_exog_coint = results_sm_exog_coint[ds][dt].det_coef
            se_C_obt = results_sm[ds][dt].stderr_det_coef
            se_C_obt_exog = results_sm_exog[ds][dt].stderr_det_coef
            se_C_obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_det_coef
            t_C_obt = results_sm[ds][dt].tvalues_det_coef
            t_C_obt_exog = results_sm_exog[ds][dt].tvalues_det_coef
            t_C_obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_det_coef
            p_C_obt = results_sm[ds][dt].pvalues_det_coef
            p_C_obt_exog = results_sm_exog[ds][dt].pvalues_det_coef
            p_C_obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_det_coef

            if "C" not in results_ref[ds][dt]["est"].keys():
                # case: there are no deterministic terms
                if (
                    C_obt.size == 0
                    and se_C_obt.size == 0
                    and t_C_obt.size == 0
                    and p_C_obt.size == 0
                ):
                    assert_(True)
                    continue

            desired = results_ref[ds][dt]["est"]["C"]
            dt_string = dt_s_tup_to_string(dt)
            if "co" in dt_string:
                err_msg = build_err_msg(ds, dt, "CONST")
                const_obt = C_obt[:, :1]
                const_obt_exog = C_obt_exog[:, :1]
                const_obt_exog_coint = C_obt_exog_coint[:, :1]
                const_des = desired[:, :1]
                C_obt = C_obt[:, 1:]
                C_obt_exog = C_obt_exog[:, 1:]
                C_obt_exog_coint = C_obt_exog_coint[:, 1:]
                desired = desired[:, 1:]
                assert_allclose(
                    const_obt, const_des, rtol, atol, False, err_msg
                )
                if exog:
                    assert_equal(
                        const_obt_exog, const_obt, "WITH EXOG: " + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        const_obt_exog_coint,
                        const_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "s" in dt_string:
                err_msg = build_err_msg(ds, dt, "SEASONAL")
                if "lo" in dt_string:
                    seas_obt = C_obt[:, :-1]
                    seas_obt_exog = C_obt_exog[:, :-1]
                    seas_obt_exog_coint = C_obt_exog_coint[:, :-1]
                    seas_des = desired[:, :-1]
                else:
                    seas_obt = C_obt
                    seas_obt_exog = C_obt_exog
                    seas_obt_exog_coint = C_obt_exog_coint
                    seas_des = desired
                assert_allclose(seas_obt, seas_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(
                        seas_obt_exog, seas_obt, "WITH EXOG: " + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        seas_obt_exog_coint,
                        seas_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "lo" in dt_string:
                err_msg = build_err_msg(ds, dt, "LINEAR TREND")
                lt_obt = C_obt[:, -1:]
                lt_obt_exog = C_obt_exog[:, -1:]
                lt_obt_exog_coint = C_obt_exog_coint[:, -1:]
                lt_des = desired[:, -1:]
                assert_allclose(lt_obt, lt_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(lt_obt_exog, lt_obt, "WITH EXOG: " + err_msg)
                if exog_coint:
                    assert_equal(
                        lt_obt_exog_coint,
                        lt_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if debug_mode and dont_test_se_t_p:  # pragma: no cover
                continue
            # standard errors
            se_desired = results_ref[ds][dt]["se"]["C"]
            if "co" in dt_string:
                err_msg = build_err_msg(ds, dt, "SE CONST")
                se_const_obt = se_C_obt[:, 0][:, None]
                se_C_obt = se_C_obt[:, 1:]
                se_const_obt_exog = se_C_obt_exog[:, 0][:, None]
                se_C_obt_exog = se_C_obt_exog[:, 1:]
                se_const_obt_exog_coint = se_C_obt_exog_coint[:, 0][:, None]
                se_C_obt_exog_coint = se_C_obt_exog_coint[:, 1:]
                se_const_des = se_desired[:, 0][:, None]
                se_desired = se_desired[:, 1:]
                assert_allclose(
                    se_const_obt, se_const_des, rtol, atol, False, err_msg
                )
                if exog:
                    assert_equal(
                        se_const_obt_exog,
                        se_const_obt,
                        "WITH EXOG: " + err_msg,
                    )
                if exog_coint:
                    assert_equal(
                        se_const_obt_exog_coint,
                        se_const_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "s" in dt_string:
                err_msg = build_err_msg(ds, dt, "SE SEASONAL")
                if "lo" in dt_string:
                    se_seas_obt = se_C_obt[:, :-1]
                    se_seas_obt_exog = se_C_obt_exog[:, :-1]
                    se_seas_obt_exog_coint = se_C_obt_exog_coint[:, :-1]
                    se_seas_des = se_desired[:, :-1]
                else:
                    se_seas_obt = se_C_obt
                    se_seas_obt_exog = se_C_obt_exog
                    se_seas_obt_exog_coint = se_C_obt_exog_coint
                    se_seas_des = se_desired
                assert_allclose(
                    se_seas_obt, se_seas_des, rtol, atol, False, err_msg
                )
                if exog:
                    assert_equal(
                        se_seas_obt_exog, se_seas_obt, "WITH EXOG: " + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        se_seas_obt_exog_coint,
                        se_seas_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
                if "lo" in dt_string:
                    err_msg = build_err_msg(ds, dt, "SE LIN. TREND")
                    se_lt_obt = se_C_obt[:, -1:]
                    se_lt_obt_exog = se_C_obt_exog[:, -1:]
                    se_lt_obt_exog_coint = se_C_obt_exog_coint[:, -1:]
                    se_lt_des = se_desired[:, -1:]
                    assert_allclose(
                        se_lt_obt, se_lt_des, rtol, atol, False, err_msg
                    )
                    if exog:
                        assert_equal(
                            se_lt_obt_exog, se_lt_obt, "WITH EXOG: " + err_msg
                        )
                    if exog_coint:
                        assert_equal(
                            se_lt_obt_exog_coint,
                            se_lt_obt,
                            "WITH EXOG_COINT: " + err_msg,
                        )
            # t-values
            t_desired = results_ref[ds][dt]["t"]["C"]
            if "co" in dt_string:
                t_const_obt = t_C_obt[:, 0][:, None]
                t_C_obt = t_C_obt[:, 1:]
                t_const_obt_exog = t_C_obt_exog[:, 0][:, None]
                t_C_obt_exog = t_C_obt_exog[:, 1:]
                t_const_obt_exog_coint = t_C_obt_exog_coint[:, 0][:, None]
                t_C_obt_exog_coint = t_C_obt_exog_coint[:, 1:]
                t_const_des = t_desired[:, 0][:, None]
                t_desired = t_desired[:, 1:]
                assert_allclose(
                    t_const_obt,
                    t_const_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "T CONST"),
                )
                if exog:
                    assert_equal(
                        t_const_obt_exog, t_const_obt, "WITH EXOG: " + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        t_const_obt_exog_coint,
                        t_const_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "s" in dt_string:
                if "lo" in dt_string:
                    t_seas_obt = t_C_obt[:, :-1]
                    t_seas_obt_exog = t_C_obt_exog[:, :-1]
                    t_seas_obt_exog_coint = t_C_obt_exog_coint[:, :-1]
                    t_seas_des = t_desired[:, :-1]
                else:
                    t_seas_obt = t_C_obt
                    t_seas_obt_exog = t_C_obt_exog
                    t_seas_obt_exog_coint = t_C_obt_exog_coint
                    t_seas_des = t_desired
                assert_allclose(
                    t_seas_obt,
                    t_seas_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "T SEASONAL"),
                )
                if exog:
                    assert_equal(
                        t_seas_obt_exog, t_seas_obt, "WITH EXOG" + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        t_seas_obt_exog_coint,
                        t_seas_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )
            if "lo" in dt_string:
                t_lt_obt = t_C_obt[:, -1:]
                t_lt_obt_exog = t_C_obt_exog[:, -1:]
                t_lt_obt_exog_coint = t_C_obt_exog_coint[:, -1:]
                t_lt_des = t_desired[:, -1:]
                assert_allclose(
                    t_lt_obt,
                    t_lt_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "T LIN. TREND"),
                )
                if exog:
                    assert_equal(
                        t_lt_obt_exog, t_lt_obt, "WITH EXOG" + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        t_lt_obt_exog_coint,
                        t_lt_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )
            # p-values
            p_desired = results_ref[ds][dt]["p"]["C"]
            if "co" in dt_string:
                p_const_obt = p_C_obt[:, 0][:, None]
                p_C_obt = p_C_obt[:, 1:]
                p_const_obt_exog = p_C_obt_exog[:, 0][:, None]
                p_C_obt_exog = p_C_obt_exog[:, 1:]
                p_const_obt_exog_coint = p_C_obt_exog_coint[:, 0][:, None]
                p_C_obt_exo_cointg = p_C_obt_exog_coint[:, 1:]
                p_const_des = p_desired[:, 0][:, None]
                p_desired = p_desired[:, 1:]
                assert_allclose(
                    p_const_obt,
                    p_const_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "P CONST"),
                )
                if exog:
                    assert_equal(
                        p_const_obt, p_const_obt_exog, "WITH EXOG" + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        p_const_obt,
                        p_const_obt_exog_coint,
                        "WITH EXOG_COINT" + err_msg,
                    )
            if "s" in dt_string:
                if "lo" in dt_string:
                    p_seas_obt = p_C_obt[:, :-1]
                    p_seas_obt_exog = p_C_obt_exog[:, :-1]
                    p_seas_obt_exog_coint = p_C_obt_exog_coint[:, :-1]
                    p_seas_des = p_desired[:, :-1]
                else:
                    p_seas_obt = p_C_obt
                    p_seas_obt_exog = p_C_obt_exog
                    p_seas_obt_exog_coint = p_C_obt_exog_coint
                    p_seas_des = p_desired
                assert_allclose(
                    p_seas_obt,
                    p_seas_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "P SEASONAL"),
                )
                if exog:
                    assert_equal(
                        p_seas_obt_exog, p_seas_obt, "WITH EXOG" + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        p_seas_obt_exog_coint,
                        p_seas_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )
            if "lo" in dt_string:
                p_lt_obt = p_C_obt[:, -1:]
                p_lt_obt_exog = p_C_obt_exog[:, -1:]
                p_lt_obt_exog_coint = p_C_obt_exog_coint[:, -1:]
                p_lt_des = p_desired[:, -1:]
                assert_allclose(
                    p_lt_obt,
                    p_lt_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "P LIN. TREND"),
                )
                if exog:
                    assert_equal(
                        p_lt_obt_exog, p_lt_obt, "WITH EXOG" + err_msg
                    )
                if exog_coint:
                    assert_equal(
                        p_lt_obt_exog_coint,
                        p_lt_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )


def test_ml_det_terms_in_coint_relation():
    if debug_mode:
        if "det_coint" not in to_test:  # pragma: no cover
            return
        print("\n\nDET_COEF_COINT", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "det terms in coint relation")
            dt_string = dt_s_tup_to_string(dt)
            obtained = results_sm[ds][dt].det_coef_coint
            obtained_exog = results_sm_exog[ds][dt].det_coef_coint
            obtained_exog_coint = results_sm_exog_coint[ds][dt].det_coef_coint
            if "ci" not in dt_string and "li" not in dt_string:
                if obtained.size > 0:
                    assert_(
                        False,
                        build_err_msg(
                            ds,
                            dt,
                            "There should not be any det terms in "
                            + "cointegration for deterministic terms "
                            + dt_string,
                        ),
                    )
                else:
                    assert_(True)
                continue
            desired = results_ref[ds][dt]["est"]["det_coint"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg
                )
            # standard errors
            se_obtained = results_sm[ds][dt].stderr_det_coef_coint
            se_obtained_exog = results_sm_exog[ds][dt].stderr_det_coef_coint
            se_obtained_exog_coint = results_sm_exog_coint[ds][
                dt
            ].stderr_det_coef_coint
            se_desired = results_ref[ds][dt]["se"]["det_coint"]
            assert_allclose(
                se_obtained,
                se_desired,
                rtol,
                atol,
                False,
                "STANDARD ERRORS\n" + err_msg,
            )
            if exog:
                assert_equal(
                    se_obtained_exog, se_obtained, "WITH EXOG" + err_msg
                )
            if exog_coint:
                assert_equal(
                    se_obtained_exog_coint,
                    se_obtained,
                    "WITH EXOG_COINT" + err_msg,
                )
            # t-values
            t_obtained = results_sm[ds][dt].tvalues_det_coef_coint
            t_obtained_exog = results_sm_exog[ds][dt].tvalues_det_coef_coint
            t_obtained_exog_coint = results_sm_exog_coint[ds][
                dt
            ].tvalues_det_coef_coint
            t_desired = results_ref[ds][dt]["t"]["det_coint"]
            assert_allclose(
                t_obtained,
                t_desired,
                rtol,
                atol,
                False,
                "t-VALUES\n" + err_msg,
            )
            if exog:
                assert_equal(
                    t_obtained_exog, t_obtained, "WITH EXOG" + err_msg
                )
            if exog_coint:
                assert_equal(
                    t_obtained_exog_coint,
                    t_obtained,
                    "WITH EXOG_COINT" + err_msg,
                )
            # p-values
            p_obtained = results_sm[ds][dt].pvalues_det_coef_coint
            p_obtained_exog = results_sm_exog[ds][dt].pvalues_det_coef_coint
            p_obtained_exog_coint = results_sm_exog_coint[ds][
                dt
            ].pvalues_det_coef_coint
            p_desired = results_ref[ds][dt]["p"]["det_coint"]
            assert_allclose(
                p_obtained,
                p_desired,
                rtol,
                atol,
                False,
                "p-VALUES\n" + err_msg,
            )
            if exog:
                assert_equal(
                    p_obtained_exog, p_obtained, "WITH EXOG" + err_msg
                )
            if exog_coint:
                assert_equal(
                    p_obtained_exog_coint,
                    p_obtained,
                    "WITH EXOG_COINT" + err_msg,
                )


def test_ml_sigma():
    if debug_mode:
        if "Sigma_u" not in to_test:  # pragma: no cover
            return
        print("\n\nSIGMA_U", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "Sigma_u")
            obtained = results_sm[ds][dt].sigma_u
            obtained_exog = results_sm_exog[ds][dt].sigma_u
            obtained_exog_coint = results_sm_exog_coint[ds][dt].sigma_u
            desired = results_ref[ds][dt]["est"]["Sigma_u"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg
                )


def test_var_rep():
    if debug_mode:
        if "VAR repr. A" not in to_test:  # pragma: no cover
            return
        print("\n\nVAR REPRESENTATION", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "VAR repr. A")
            obtained = results_sm[ds][dt].var_rep
            obtained_exog = results_sm_exog[ds][dt].var_rep
            obtained_exog_coint = results_sm_exog_coint[ds][dt].var_rep
            p = obtained.shape[0]
            desired = np.hsplit(results_ref[ds][dt]["est"]["VAR A"], p)
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg
                )


def test_var_to_vecm():
    if debug_mode:
        if "VAR to VEC representation" not in to_test:  # pragma: no cover
            return
        print("\n\nVAR TO VEC", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "VAR to VEC representation")
            sigma_u = results_sm[ds][dt].sigma_u
            coefs = results_sm[ds][dt].var_rep
            intercept = np.zeros(len(sigma_u))
            # Note: _params_info k_trend, k_exog, ... is inferred with defaults
            var = VARProcess(coefs, intercept, sigma_u)
            vecm_results = var.to_vecm()
            obtained_pi = vecm_results["Pi"]
            obtained_gamma = vecm_results["Gamma"]

            desired_pi = np.dot(
                results_sm[ds][dt].alpha, results_sm[ds][dt].beta.T
            )
            desired_gamma = results_sm[ds][dt].gamma
            assert_allclose(
                obtained_pi, desired_pi, rtol, atol, False, err_msg + " Pi"
            )
            assert_allclose(
                obtained_gamma,
                desired_gamma,
                rtol,
                atol,
                False,
                err_msg + " Gamma",
            )


# Commented out since JMulTi shows the same det. terms for both VEC & VAR repr.
# def test_var_rep_det():
#     for ds in datasets:
#         for dt in dt_s_list:
#             if dt != "n":
#                 err_msg = build_err_msg(ds, dt, "VAR repr. deterministic")
#                 obtained = 0  # not implemented since the same values as VECM
#                 desired = results_ref[ds][dt]["est"]["VAR deterministic"]
#                 assert_allclose(obtained, desired, rtol, atol, False, err_msg)


def test_log_like():
    if debug_mode:
        if "log_like" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nLOG LIKELIHOOD", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "Log Likelihood")
            obtained = results_sm[ds][dt].llf
            obtained_exog = results_sm_exog[ds][dt].llf
            obtained_exog_coint = results_sm_exog_coint[ds][dt].llf
            # JMulTi's llf seems to have a bug (Stata and tsdyn suggest that
            # our code is correct). We use nobs to correct this inconsistency.
            nobs = results_sm[ds][dt].nobs
            desired = results_ref[ds][dt]["log_like"] * nobs / (nobs - 1)
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg
                )


def test_fc():
    if debug_mode:
        if "fc" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nFORECAST", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            STEPS = 5
            ALPHA = 0.05
            err_msg = build_err_msg(ds, dt, "FORECAST")
            # test point forecast functionality of predict method
            obtained = results_sm[ds][dt].predict(steps=STEPS)
            desired = results_ref[ds][dt]["fc"]["fc"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)

            # -----------------------------------------------------------------
            # with exog:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_fc = None
            if exog:  # build future values of exog and test:
                seasons = dt[1]
                exog_model = results_sm_exog[ds][dt].exog
                exog_seasons_fc = exog_model[-seasons:, : seasons - 1]
                exog_seasons_fc = np.pad(
                    exog_seasons_fc,
                    ((0, STEPS - exog_seasons_fc.shape[0]), (0, 0)),
                    "wrap",
                )
                # if linear trend in exog
                if exog_seasons_fc.shape[1] + 1 == exog_model.shape[1]:
                    exog_lt_fc = exog_model[-1, -1] + 1 + np.arange(STEPS)
                    exog_fc = np.column_stack((exog_seasons_fc, exog_lt_fc))
                else:
                    exog_fc = exog_seasons_fc
                obtained_exog = results_sm_exog[ds][dt].predict(
                    steps=STEPS, exog_fc=exog_fc
                )
                assert_allclose(
                    obtained_exog,
                    obtained,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG" + err_msg,
                )
            # test predict method with confidence interval calculation
            err_msg = build_err_msg(ds, dt, "FORECAST WITH INTERVALS")
            obtained_w_intervals = results_sm[ds][dt].predict(
                steps=STEPS, alpha=ALPHA
            )
            obtained_w_intervals_exog = results_sm_exog[ds][dt].predict(
                steps=STEPS, alpha=ALPHA, exog_fc=exog_fc
            )
            obt = obtained_w_intervals[0]  # forecast
            obt_l = obtained_w_intervals[1]  # lower bound
            obt_u = obtained_w_intervals[2]  # upper bound
            obt_exog = obtained_w_intervals_exog[0]
            obt_exog_l = obtained_w_intervals_exog[1]
            obt_exog_u = obtained_w_intervals_exog[2]
            des = results_ref[ds][dt]["fc"]["fc"]
            des_l = results_ref[ds][dt]["fc"]["lower"]
            des_u = results_ref[ds][dt]["fc"]["upper"]
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)
            if exog:
                assert_allclose(
                    obt_exog, obt, 1e-07, 0, False, "WITH EXOG" + err_msg
                )
                assert_allclose(
                    obt_exog_l, obt_l, 1e-07, 0, False, "WITH EXOG" + err_msg
                )
                assert_allclose(
                    obt_exog_u, obt_u, 1e-07, 0, False, "WITH EXOG" + err_msg
                )

            # -----------------------------------------------------------------
            # with exog_coint:
            exog_coint_model = results_sm_exog_coint[ds][dt].exog_coint
            exog_coint = exog_coint_model is not None
            exog_coint_fc = None
            if exog_coint:  # build future values of exog_coint and test:
                # const is in exog_coint in all tests that have exog_coint
                exog_coint_fc = np.ones(STEPS - 1)
                # if linear trend in exog_coint
                if exog_coint_model.shape[1] == 2:
                    exog_coint_fc = np.column_stack(
                        (
                            exog_coint_fc,
                            exog_coint_model[-1, -1]
                            + 1
                            + np.arange(STEPS - 1),
                        )
                    )
                obtained_exog_coint = results_sm_exog_coint[ds][dt].predict(
                    steps=STEPS, exog_coint_fc=exog_coint_fc
                )

                assert_allclose(
                    obtained_exog_coint,
                    obtained,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )
            # test predict method with confidence interval calculation
            err_msg = build_err_msg(ds, dt, "FORECAST WITH INTERVALS")
            obtained_w_intervals = results_sm[ds][dt].predict(
                steps=STEPS, alpha=ALPHA
            )
            obtained_w_intervals_exog_coint = results_sm_exog_coint[ds][
                dt
            ].predict(steps=STEPS, alpha=ALPHA, exog_coint_fc=exog_coint_fc)
            obt = obtained_w_intervals[0]  # forecast
            obt_l = obtained_w_intervals[1]  # lower bound
            obt_u = obtained_w_intervals[2]  # upper bound
            obt_exog_coint = obtained_w_intervals_exog_coint[0]
            obt_exog_coint_l = obtained_w_intervals_exog_coint[1]
            obt_exog_coint_u = obtained_w_intervals_exog_coint[2]
            des = results_ref[ds][dt]["fc"]["fc"]
            des_l = results_ref[ds][dt]["fc"]["lower"]
            des_u = results_ref[ds][dt]["fc"]["upper"]
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)
            if exog_coint:
                assert_allclose(
                    obt_exog_coint,
                    obt,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )
                assert_allclose(
                    obt_exog_coint_l,
                    obt_l,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )
                assert_allclose(
                    obt_exog_coint_u,
                    obt_u,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )


def test_granger_causality():
    if debug_mode:
        if "granger" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nGRANGER", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg_g_p = build_err_msg(ds, dt, "GRANGER CAUS. - p-VALUE")
            err_msg_g_t = build_err_msg(ds, dt, "GRANGER CAUS. - TEST STAT.")

            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ["y" + str(i + 1) for i in causing_ind]
                causing_key = tuple(ds.variable_names[i] for i in causing_ind)

                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_names = ["y" + str(i + 1) for i in caused_ind]
                caused_key = tuple(ds.variable_names[i] for i in caused_ind)

                granger_sm_ind = results_sm[ds][dt].test_granger_causality(
                    caused_ind, causing_ind
                )
                granger_sm_ind_exog = results_sm_exog[ds][
                    dt
                ].test_granger_causality(caused_ind, causing_ind)
                granger_sm_ind_exog_coint = results_sm_exog_coint[ds][
                    dt
                ].test_granger_causality(caused_ind, causing_ind)
                granger_sm_str = results_sm[ds][dt].test_granger_causality(
                    caused_names, causing_names
                )

                # call methods to assure they do not raise exceptions
                granger_sm_ind.summary()
                str(granger_sm_ind)  # __str__()
                assert_(granger_sm_ind == granger_sm_str)  # __eq__()

                # test test-statistic for Granger non-causality:
                g_t_obt = granger_sm_ind.test_statistic
                g_t_obt_exog = granger_sm_ind_exog.test_statistic
                g_t_obt_exog_coint = granger_sm_ind_exog_coint.test_statistic
                g_t_des = results_ref[ds][dt]["granger_caus"]["test_stat"][
                    (causing_key, caused_key)
                ]
                assert_allclose(
                    g_t_obt, g_t_des, rtol, atol, False, err_msg_g_t
                )
                if exog:
                    assert_allclose(
                        g_t_obt_exog,
                        g_t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG" + err_msg_g_t,
                    )
                if exog_coint:
                    assert_allclose(
                        g_t_obt_exog_coint,
                        g_t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG_COINT" + err_msg_g_t,
                    )
                # check whether string sequences as args work in the same way:
                g_t_obt_str = granger_sm_str.test_statistic
                assert_allclose(
                    g_t_obt_str,
                    g_t_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_g_t
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g. 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1 or len(caused_ind) == 1:
                    ci = (
                        causing_ind[0]
                        if len(causing_ind) == 1
                        else causing_ind
                    )
                    ce = caused_ind[0] if len(caused_ind) == 1 else caused_ind
                    granger_sm_single_ind = results_sm[ds][
                        dt
                    ].test_granger_causality(ce, ci)
                    g_t_obt_single = granger_sm_single_ind.test_statistic
                    assert_allclose(
                        g_t_obt_single,
                        g_t_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_g_t
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )

                # test p-value for Granger non-causality:
                g_p_obt = granger_sm_ind.pvalue
                g_p_des = results_ref[ds][dt]["granger_caus"]["p"][
                    (causing_key, caused_key)
                ]
                assert_allclose(
                    g_p_obt, g_p_des, rtol, atol, False, err_msg_g_p
                )
                # check whether string sequences as args work in the same way:
                g_p_obt_str = granger_sm_str.pvalue
                assert_allclose(
                    g_p_obt_str,
                    g_p_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_g_t
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g. 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1:
                    g_p_obt_single = granger_sm_single_ind.pvalue
                    assert_allclose(
                        g_p_obt_single,
                        g_p_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_g_t
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )


def test_inst_causality():  # test instantaneous causality
    if debug_mode:
        if "inst. causality" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nINST. CAUSALITY", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg_i_p = build_err_msg(ds, dt, "INSTANT. CAUS. - p-VALUE")
            err_msg_i_t = build_err_msg(ds, dt, "INSTANT. CAUS. - TEST STAT.")

            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ["y" + str(i + 1) for i in causing_ind]
                causing_key = tuple(ds.variable_names[i] for i in causing_ind)

                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_key = tuple(ds.variable_names[i] for i in caused_ind)
                inst_sm_ind = results_sm[ds][dt].test_inst_causality(
                    causing_ind
                )
                inst_sm_ind_exog = results_sm_exog[ds][dt].test_inst_causality(
                    causing_ind
                )
                inst_sm_ind_exog_coint = results_sm_exog_coint[ds][
                    dt
                ].test_inst_causality(causing_ind)
                inst_sm_str = results_sm[ds][dt].test_inst_causality(
                    causing_names
                )
                # call methods to assure they do not raise exceptions
                inst_sm_ind.summary()
                str(inst_sm_ind)  # __str__()
                assert_(inst_sm_ind == inst_sm_str)  # __eq__()
                # test test-statistic for instantaneous non-causality
                t_obt = inst_sm_ind.test_statistic
                t_obt_exog = inst_sm_ind_exog.test_statistic
                t_obt_exog_coint = inst_sm_ind_exog_coint.test_statistic
                t_des = results_ref[ds][dt]["inst_caus"]["test_stat"][
                    (causing_key, caused_key)
                ]
                assert_allclose(t_obt, t_des, rtol, atol, False, err_msg_i_t)
                if exog:
                    assert_allclose(
                        t_obt_exog,
                        t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG" + err_msg_i_t,
                    )
                if exog_coint:
                    assert_allclose(
                        t_obt_exog_coint,
                        t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG_COINT" + err_msg_i_t,
                    )
                # check whether string sequences as args work in the same way:
                t_obt_str = inst_sm_str.test_statistic
                assert_allclose(
                    t_obt_str,
                    t_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_i_t
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g. 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][
                        dt
                    ].test_inst_causality(causing_ind[0])
                    t_obt_single = inst_sm_single_ind.test_statistic
                    assert_allclose(
                        t_obt_single,
                        t_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_i_t
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )

                # test p-value for instantaneous non-causality
                p_obt = (
                    results_sm[ds][dt].test_inst_causality(causing_ind).pvalue
                )
                p_des = results_ref[ds][dt]["inst_caus"]["p"][
                    (causing_key, caused_key)
                ]
                assert_allclose(p_obt, p_des, rtol, atol, False, err_msg_i_p)
                # check whether string sequences as args work in the same way:
                p_obt_str = inst_sm_str.pvalue
                assert_allclose(
                    p_obt_str,
                    p_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_i_p
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g. 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][
                        dt
                    ].test_inst_causality(causing_ind[0])
                    p_obt_single = inst_sm_single_ind.pvalue
                    assert_allclose(
                        p_obt_single,
                        p_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_i_p
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )


def test_impulse_response():
    if debug_mode:
        if "impulse-response" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nIMPULSE-RESPONSE", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "IMULSE-RESPONSE")
            periods = 20
            obtained_all = results_sm[ds][dt].irf(periods=periods).irfs
            obtained_all_exog = (
                results_sm_exog[ds][dt].irf(periods=periods).irfs
            )
            obtained_all_exog_coint = (
                results_sm_exog_coint[ds][dt].irf(periods=periods).irfs
            )
            # flatten inner arrays to make them comparable to parsed results:
            obtained_all = obtained_all.reshape(periods + 1, -1)
            obtained_all_exog = obtained_all_exog.reshape(periods + 1, -1)
            obtained_all_exog_coint = obtained_all_exog_coint.reshape(
                periods + 1, -1
            )
            desired_all = results_ref[ds][dt]["ir"]
            assert_allclose(
                obtained_all, desired_all, rtol, atol, False, err_msg
            )
            if exog:
                assert_equal(
                    obtained_all_exog, obtained_all, "WITH EXOG" + err_msg
                )
            if exog_coint:
                assert_equal(
                    obtained_all_exog_coint,
                    obtained_all,
                    "WITH EXOG_COINT" + err_msg,
                )


def test_lag_order_selection():
    if debug_mode:
        if "lag order" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nLAG ORDER SELECTION", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            deterministic = dt[0]
            endog_tot = data[ds]

            trend = "n" if dt[0] == "nc" else dt[0]
            obtained_all = select_order(
                endog_tot, 10, deterministic=dt[0], seasons=dt[1]
            )
            deterministic_outside_exog = ""
            # "co" is not in exog in any test case
            if "co" in deterministic:
                deterministic_outside_exog += "co"
            # "lo" is is in exog only in test cases with seasons>0
            if "lo" in deterministic and dt[1] == 0:
                deterministic_outside_exog += "lo"

            exog_model = results_sm_exog[ds][dt].exog
            exog = exog_model is not None
            exog_coint_model = results_sm_exog_coint[ds][dt].exog_coint
            exog_coint = exog_coint_model is not None

            obtained_all_exog = select_order(
                endog_tot,
                10,
                deterministic_outside_exog,
                seasons=0,
                exog=exog_model,
            )
            # "ci" and "li" are always in exog_coint, so pass "n" as det. term
            obtained_all_exog_coint = select_order(
                endog_tot, 10, "n", seasons=dt[1], exog_coint=exog_coint_model
            )
            for ic in ["aic", "fpe", "hqic", "bic"]:
                err_msg = build_err_msg(
                    ds, dt, "LAG ORDER SELECTION - " + ic.upper()
                )
                obtained = getattr(obtained_all, ic)
                desired = results_ref[ds][dt]["lagorder"][ic]
                assert_allclose(obtained, desired, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(
                        getattr(obtained_all_exog, ic),
                        getattr(obtained_all, ic),
                        "WITH EXOG" + err_msg,
                    )
                if exog_coint:
                    assert_equal(
                        getattr(obtained_all_exog_coint, ic),
                        getattr(obtained_all, ic),
                        "WITH EXOG_COINT" + err_msg,
                    )
            # call methods to assure they do not raise exceptions
            obtained_all.summary()
            str(obtained_all)  # __str__()


def test_normality():
    if debug_mode:
        if "test_norm" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nTEST NON-NORMALITY", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            obtained = results_sm[ds][dt].test_normality(signif=0.05)
            obtained_exog = results_sm_exog[ds][dt].test_normality(signif=0.05)
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_normality(
                signif=0.05
            )
            err_msg = build_err_msg(ds, dt, "TEST NON-NORMALITY - STATISTIC")
            obt_statistic = obtained.test_statistic
            obt_statistic_exog = obtained_exog.test_statistic
            obt_statistic_exog_coint = obtained_exog_coint.test_statistic
            des_statistic = results_ref[ds][dt]["test_norm"][
                "joint_test_statistic"
            ]
            assert_allclose(
                obt_statistic, des_statistic, rtol, atol, False, err_msg
            )
            if exog:
                assert_equal(
                    obt_statistic_exog, obt_statistic, "WITH EXOG" + err_msg
                )
            if exog_coint:
                assert_equal(
                    obt_statistic_exog_coint,
                    obt_statistic,
                    "WITH EXOG_COINT" + err_msg,
                )
            err_msg = build_err_msg(ds, dt, "TEST NON-NORMALITY - P-VALUE")
            obt_pvalue = obtained.pvalue
            des_pvalue = results_ref[ds][dt]["test_norm"]["joint_pvalue"]
            assert_allclose(obt_pvalue, des_pvalue, rtol, atol, False, err_msg)
            # call methods to assure they do not raise exceptions
            obtained.summary()
            str(obtained)  # __str__()
            assert_(obtained == obtained_exog)  # __eq__()


def test_whiteness():
    if debug_mode:
        if "whiteness" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nTEST WHITENESS OF RESIDUALS", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            lags = results_ref[ds][dt]["whiteness"]["tested order"]

            obtained = results_sm[ds][dt].test_whiteness(nlags=lags)
            obtained_exog = results_sm_exog[ds][dt].test_whiteness(nlags=lags)
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_whiteness(
                nlags=lags
            )
            # test statistic
            err_msg = build_err_msg(
                ds, dt, "WHITENESS OF RESIDUALS - " "TEST STATISTIC"
            )
            desired = results_ref[ds][dt]["whiteness"]["test statistic"]
            assert_allclose(
                obtained.test_statistic, desired, rtol, atol, False, err_msg
            )
            if exog:
                assert_equal(
                    obtained_exog.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG" + err_msg,
                )
            if exog_coint:
                assert_equal(
                    obtained_exog_coint.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG_COINT" + err_msg,
                )
            # p-value
            err_msg = build_err_msg(
                ds, dt, "WHITENESS OF RESIDUALS - " "P-VALUE"
            )
            desired = results_ref[ds][dt]["whiteness"]["p-value"]
            assert_allclose(
                obtained.pvalue, desired, rtol, atol, False, err_msg
            )

            obtained = results_sm[ds][dt].test_whiteness(
                nlags=lags, adjusted=True
            )
            obtained_exog = results_sm_exog[ds][dt].test_whiteness(
                nlags=lags, adjusted=True
            )
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_whiteness(
                nlags=lags, adjusted=True
            )
            # test statistic (adjusted Portmanteau test)
            err_msg = build_err_msg(
                ds,
                dt,
                "WHITENESS OF RESIDUALS - " "TEST STATISTIC (ADJUSTED TEST)",
            )
            desired = results_ref[ds][dt]["whiteness"]["test statistic adj."]
            assert_allclose(
                obtained.test_statistic, desired, rtol, atol, False, err_msg
            )
            if exog:
                assert_equal(
                    obtained_exog.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG" + err_msg,
                )
            if exog_coint:
                assert_equal(
                    obtained_exog_coint.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG_COINT" + err_msg,
                )
            # p-value (adjusted Portmanteau test)
            err_msg = build_err_msg(
                ds, dt, "WHITENESS OF RESIDUALS - " "P-VALUE (ADJUSTED TEST)"
            )
            desired = results_ref[ds][dt]["whiteness"]["p-value adjusted"]
            assert_allclose(
                obtained.pvalue, desired, rtol, atol, False, err_msg
            )
            # call methods to assure they do not raise exceptions
            obtained.summary()
            str(obtained)  # __str__()
            assert_(obtained == obtained_exog)  # __eq__()


def test_summary():
    if debug_mode:
        if "summary" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nSUMMARY", end="")
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            # see if summary gets printed
            results_sm[ds][dt].summary(alpha=0.05)

            exog = results_sm_exog[ds][dt].exog is not None
            if exog is not None:
                results_sm_exog[ds][dt].summary(alpha=0.05)

            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            if exog_coint is not None:
                results_sm_exog_coint[ds][dt].summary(alpha=0.05)


def test_exceptions():
    if debug_mode:
        if "exceptions" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nEXCEPTIONS\n", end="")
    ds = datasets[0]
    dt = ds.dt_s_list[0]
    endog = data[datasets[0]]

    # select_coint_rank:
    # method argument cannot be "my_method"
    assert_raises(
        ValueError, select_coint_rank, endog, 0, 3, "my_method", 0.05
    )
    # det_order has to be -1, 0, or 1:
    assert_raises(ValueError, select_coint_rank, endog, 2, 3)
    assert_raises(ValueError, select_coint_rank, endog, 0.5, 3)
    # 0.025 is not possible (must be 0.1, 0.05, or 0.01)
    assert_raises(ValueError, select_coint_rank, endog, 0, 3, "trace", 0.025)

    # Granger_causality:
    # ### 0<signif<1
    # this means signif=0
    assert_raises(
        ValueError, results_sm[ds][dt].test_granger_causality, 0, None, 0
    )
    # ### caused must be int, str or iterable of int or str
    # 0.5 not int
    assert_raises(TypeError, results_sm[ds][dt].test_granger_causality, [0.5])
    # ### causing must be None, int, str or iterable of int or str
    # .5 not int
    assert_raises(TypeError, results_sm[ds][dt].test_granger_causality, 0, 0.5)

    # exceptions in VECM class
    # ### choose only one of the two: "co" and "ci"
    model = VECM(endog, k_ar_diff=1, deterministic="cico")
    assert_raises(ValueError, model.fit)
    # ### we analyze multiple time series
    univariate_data = endog[0]
    assert_raises(ValueError, VECM, univariate_data)
    # ### fit only allowed with known method
    model = VECM(endog, k_ar_diff=1, deterministic="n")
    assert_raises(ValueError, model.fit, "abc")  # no "abc" estim.-method

    # pass a shorter array than endog as exog_coint argument
    assert_raises(ValueError, VECM, endog, None, np.ones(len(endog) - 1))
    # forecasting: argument checks
    STEPS = 5
    # ### with exog
    exog = seasonal_dummies(4, len(endog), 2, centered=True)  # seasonal...
    exog = np.hstack((exog, 1 + np.arange(len(endog)).reshape(-1, 1)))  # & lin
    vecm_res = VECM(
        endog, exog, k_ar_diff=3, coint_rank=coint_rank, deterministic="co"
    ).fit()
    # ##### exog_fc not passed as argument:
    assert_raises_regex(ValueError, "exog_fc is None.*", vecm_res.predict)
    # ##### exog_fc is too short:
    exog_fc = np.ones(STEPS)
    assert_raises_regex(
        ValueError,
        ".*exog_fc must have at least steps elements.*",
        vecm_res.predict,
        5,
        None,
        exog_fc[:2],
    )  # [:2] shortens exog_fc
    # ##### exog_coint_fc (NOT exog_fc) is passed when there is no exog_coint
    assert_raises_regex(
        ValueError,
        ".*exog_coint attribute is None.*",
        vecm_res.predict,
        5,
        None,
        exog_fc,
        exog_fc,
    )  # passed as exog_coint_fc-argument!
    # ### with exog_coint
    exog_coint = []
    exog_coint.append(np.ones(len(endog)).reshape(-1, 1))
    exog_coint.append(1 + np.arange(len(endog)).reshape(-1, 1))
    exog_coint = np.hstack(exog_coint)
    vecm_res = VECM(
        endog, k_ar_diff=1, deterministic="n", exog_coint=exog_coint
    ).fit()
    # ##### exog_coint_fc not passed as argument:
    assert_raises_regex(
        ValueError, "exog_coint_fc is None.*", vecm_res.predict
    )
    # ##### exog_coint_fc is too short:
    exog_coint_fc = np.ones(STEPS)
    assert_raises_regex(
        ValueError,
        ".*exog_coint_fc must have at least steps elements.*",
        vecm_res.predict,
        5,
        None,
        None,
        exog_coint_fc[:2],
    )  # [:2] shortens
    # ##### exog_fc (NOT exog_coint_fc) is passed when there is no exog
    assert_raises_regex(
        ValueError,
        ".*exog attribute is None.*",
        vecm_res.predict,
        5,
        None,
        exog_coint_fc,
    )  # passed as exog_fc-argument!


def test_select_coint_rank():  # This is only a smoke test.
    if debug_mode:
        if "select_coint_rank" not in to_test:  # pragma: no cover
            return
        else:
            print("\n\nSELECT_COINT_RANK\n", end="")
    endog = data[datasets[0]]
    neqs = endog.shape[1]

    trace_result = select_coint_rank(endog, 0, 3, method="trace")
    rank = trace_result.rank
    r_1 = trace_result.r_1
    test_stats = trace_result.test_stats
    crit_vals = trace_result.crit_vals
    if rank > 0:
        assert_equal(r_1[0], r_1[1])
        for i in range(rank):
            assert_(test_stats[i] > crit_vals[i])
    if rank < neqs:
        assert_(test_stats[rank] < crit_vals[rank])

    maxeig_result = select_coint_rank(endog, 0, 3, method="maxeig", signif=0.1)
    rank = maxeig_result.rank
    r_1 = maxeig_result.r_1
    test_stats = maxeig_result.test_stats
    crit_vals = maxeig_result.crit_vals
    if maxeig_result.rank > 0:
        assert_equal(r_1[0], r_1[1] - 1)
        for i in range(rank):
            assert_(test_stats[i] > crit_vals[i])
    if rank < neqs:
        assert_(test_stats[rank] < crit_vals[rank])


def test_VECM_seasonal_forecast():
    # timing of seasonal dummies, VAR forecast horizon
    np.random.seed(964255)
    nobs = 200
    seasons = 6
    fact = np.cumsum(0.1 + np.random.randn(nobs, 2), 0)

    xx = np.random.randn(nobs + 2, 3)
    xx = xx[2:] + 0.6 * xx[1:-1] + 0.25 * xx[:-2]
    xx[:, :2] += fact[:, 0][:, None]
    xx[:, 2:] += fact[:, 1][:, None]
    # add large seasonal effect
    xx += 3 * np.log(0.1 + (np.arange(nobs)[:, None] % seasons))

    res0 = VECM(
        xx,
        k_ar_diff=0,
        coint_rank=2,
        deterministic="co",
        seasons=seasons,
        first_season=0,
    ).fit()
    res2 = VECM(
        xx,
        k_ar_diff=2,
        coint_rank=2,
        deterministic="co",
        seasons=seasons,
        first_season=0,
    ).fit()
    res4 = VECM(
        xx,
        k_ar_diff=4,
        coint_rank=2,
        deterministic="co",
        seasons=seasons,
        first_season=0,
    ).fit()

    # check that seasonal dummy are independent of number of lags
    assert_allclose(
        res2._delta_x.T[-2 * seasons :, -seasons:],
        res0._delta_x.T[-2 * seasons :, -seasons:],
        rtol=1e-10,
    )
    assert_allclose(
        res4._delta_x.T[-2 * seasons :, -seasons:],
        res0._delta_x.T[-2 * seasons :, -seasons:],
        rtol=1e-10,
    )

    # check location of smallest seasonal coefficient
    assert_array_equal(np.argmin(res0.det_coef, axis=1), [1, 1, 1])
    assert_array_equal(np.argmin(res2.det_coef, axis=1), [1, 1, 1])
    assert_array_equal(np.argmin(res4.det_coef, axis=1), [1, 1, 1])

    # predict 3 cycles, check location of dips
    dips_true = np.array([[4, 4, 4], [10, 10, 10], [16, 16, 16]])
    for res in [res0, res2, res4]:
        forecast = res.predict(steps=3 * seasons)
        dips = np.sort(np.argsort(forecast, axis=0)[:3], axis=0)
        assert_array_equal(dips, dips_true)

    # res2.plot_forecast(steps=18, alpha=0.1, n_last_obs=4*seasons)
