/*

Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
Written by Mark Wiebe (mwwiebe@gmail.com)

Copyright (c) 2011 by Enthought, Inc.
Copyright (c) 2005-2011, NumPy Developers

All rights reserved.
See NUMPY_LICENSE.txt for the license.
*/

#pragma once

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API

#include <numpy/ndarraytypes.h>
#include "pandas/vendored/numpy/datetime/np_datetime.h"
#include "pandas/vendored/numpy/datetime/np_datetime_strings.h"
#include "pandas/datetime/date_conversions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  npy_datetime (*npy_datetimestruct_to_datetime)(NPY_DATETIMEUNIT,
                                                 const npy_datetimestruct *);
  int (*scaleNanosecToUnit)(npy_int64 *, NPY_DATETIMEUNIT);
  char *(*int64ToIso)(int64_t, NPY_DATETIMEUNIT, NPY_DATETIMEUNIT, size_t *);
  npy_datetime (*NpyDateTimeToEpoch)(npy_datetime, NPY_DATETIMEUNIT);
  char *(*PyDateTimeToIso)(PyObject *, NPY_DATETIMEUNIT, size_t *);
  npy_datetime (*PyDateTimeToEpoch)(PyObject *, NPY_DATETIMEUNIT);
  char *(*int64ToIsoDuration)(int64_t, size_t *);
  void (*pandas_datetime_to_datetimestruct)(npy_datetime, NPY_DATETIMEUNIT,
                                            npy_datetimestruct *);
  void (*pandas_timedelta_to_timedeltastruct)(npy_datetime, NPY_DATETIMEUNIT,
                                              pandas_timedeltastruct *);
  int (*convert_pydatetime_to_datetimestruct)(PyObject *, npy_datetimestruct *);
  int (*cmp_npy_datetimestruct)(const npy_datetimestruct *,
                                const npy_datetimestruct *);
  PyArray_DatetimeMetaData (*get_datetime_metadata_from_dtype)(PyArray_Descr *);
  int (*parse_iso_8601_datetime)(const char *, int, int, npy_datetimestruct *,
                                 NPY_DATETIMEUNIT *, int *, int *, const char *,
                                 int, FormatRequirement);
  int (*get_datetime_iso_8601_strlen)(int, NPY_DATETIMEUNIT);
  int (*make_iso_8601_datetime)(npy_datetimestruct *, char *, int, int,
                                NPY_DATETIMEUNIT);
  int (*make_iso_8601_timedelta)(pandas_timedeltastruct *, char *, size_t *);
} PandasDateTime_CAPI;

// The capsule name appears limited to module.attributename; see bpo-32414
// cpython has an open PR gh-6898 to fix, but hasn't had traction for years
#define PandasDateTime_CAPSULE_NAME "pandas._pandas_datetime_CAPI"

/* block used as part of public API */
#ifndef _PANDAS_DATETIME_IMPL
static PandasDateTime_CAPI *PandasDateTimeAPI = NULL;

#define PandasDateTime_IMPORT                                                  \
  PandasDateTimeAPI =                                                          \
      (PandasDateTime_CAPI *)PyCapsule_Import(PandasDateTime_CAPSULE_NAME, 0)

#define npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT, npy_datetimestruct)   \
  PandasDateTimeAPI->npy_datetimestruct_to_datetime((NPY_DATETIMEUNIT),        \
                                                    (npy_datetimestruct))
#define scaleNanosecToUnit(value, unit)                                        \
  PandasDateTimeAPI->scaleNanosecToUnit((value), (unit))
#define int64ToIso(value, valueUnit, base, len)                                \
  PandasDateTimeAPI->int64ToIso((value), (valueUnit), (base), (len))
#define NpyDateTimeToEpoch(dt, base)                                           \
  PandasDateTimeAPI->NpyDateTimeToEpoch((dt), (base))
#define PyDateTimeToIso(obj, base, len)                                        \
  PandasDateTimeAPI->PyDateTimeToIso((obj), (base), (len))
#define PyDateTimeToEpoch(dt, base)                                            \
  PandasDateTimeAPI->PyDateTimeToEpoch((dt), (base))
#define int64ToIsoDuration(value, len)                                         \
  PandasDateTimeAPI->int64ToIsoDuration((value), (len))
#define pandas_datetime_to_datetimestruct(dt, base, out)                       \
  PandasDateTimeAPI->pandas_datetime_to_datetimestruct((dt), (base), (out))
#define pandas_timedelta_to_timedeltastruct(td, base, out)                     \
  PandasDateTimeAPI->pandas_timedelta_to_timedeltastruct((td), (base), (out))
#define convert_pydatetime_to_datetimestruct(dtobj, out)                       \
  PandasDateTimeAPI->convert_pydatetime_to_datetimestruct((dtobj), (out))
#define cmp_npy_datetimestruct(a, b)                                           \
  PandasDateTimeAPI->cmp_npy_datetimestruct((a), (b))
#define get_datetime_metadata_from_dtype(dtype)                                \
  PandasDateTimeAPI->get_datetime_metadata_from_dtype((dtype))
#define parse_iso_8601_datetime(str, len, want_exc, out, out_bestunit,         \
                                out_local, out_tzoffset, format, format_len,   \
                                format_requirement)                            \
  PandasDateTimeAPI->parse_iso_8601_datetime(                                  \
      (str), (len), (want_exc), (out), (out_bestunit), (out_local),            \
      (out_tzoffset), (format), (format_len), (format_requirement))
#define get_datetime_iso_8601_strlen(local, base)                              \
  PandasDateTimeAPI->get_datetime_iso_8601_strlen((local), (base))
#define make_iso_8601_datetime(dts, outstr, outlen, utc, base)                 \
  PandasDateTimeAPI->make_iso_8601_datetime((dts), (outstr), (outlen), (utc),  \
                                            (base))
#define make_iso_8601_timedelta(tds, outstr, outlen)                           \
  PandasDateTimeAPI->make_iso_8601_timedelta((tds), (outstr), (outlen))
#endif /* !defined(_PANDAS_DATETIME_IMPL) */

#ifdef __cplusplus
}
#endif
