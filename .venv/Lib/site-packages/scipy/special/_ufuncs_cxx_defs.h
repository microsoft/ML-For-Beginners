#ifndef UFUNCS_PROTO_H
#define UFUNCS_PROTO_H 1
#include "_faddeeva.h"
npy_double faddeeva_dawsn(npy_double);
npy_cdouble faddeeva_dawsn_complex(npy_cdouble);
#include "ellint_carlson_wrap.hh"
npy_double fellint_RC(npy_double, npy_double);
npy_cdouble cellint_RC(npy_cdouble, npy_cdouble);
npy_double fellint_RD(npy_double, npy_double, npy_double);
npy_cdouble cellint_RD(npy_cdouble, npy_cdouble, npy_cdouble);
npy_double fellint_RF(npy_double, npy_double, npy_double);
npy_cdouble cellint_RF(npy_cdouble, npy_cdouble, npy_cdouble);
npy_double fellint_RG(npy_double, npy_double, npy_double);
npy_cdouble cellint_RG(npy_cdouble, npy_cdouble, npy_cdouble);
npy_double fellint_RJ(npy_double, npy_double, npy_double, npy_double);
npy_cdouble cellint_RJ(npy_cdouble, npy_cdouble, npy_cdouble, npy_cdouble);
npy_cdouble faddeeva_erf(npy_cdouble);
npy_cdouble faddeeva_erfc_complex(npy_cdouble);
npy_double faddeeva_erfcx(npy_double);
npy_cdouble faddeeva_erfcx_complex(npy_cdouble);
npy_double faddeeva_erfi(npy_double);
npy_cdouble faddeeva_erfi_complex(npy_cdouble);
#include "boost_special_functions.h"
npy_float erfinv_float(npy_float);
npy_double erfinv_double(npy_double);
#include "_logit.h"
npy_double expit(npy_double);
npy_float expitf(npy_float);
npy_longdouble expitl(npy_longdouble);
npy_double hyp1f1_double(npy_double, npy_double, npy_double);
npy_double log_expit(npy_double);
npy_float log_expitf(npy_float);
npy_longdouble log_expitl(npy_longdouble);
npy_double faddeeva_log_ndtr(npy_double);
npy_cdouble faddeeva_log_ndtr_complex(npy_cdouble);
npy_double logit(npy_double);
npy_float logitf(npy_float);
npy_longdouble logitl(npy_longdouble);
npy_cdouble faddeeva_ndtr(npy_cdouble);
npy_float powm1_float(npy_float, npy_float);
npy_double powm1_double(npy_double, npy_double);
npy_double faddeeva_voigt_profile(npy_double, npy_double, npy_double);
npy_cdouble faddeeva_w(npy_cdouble);
#include "_wright.h"
npy_cdouble wrightomega(npy_cdouble);
npy_double wrightomega_real(npy_double);
#endif
