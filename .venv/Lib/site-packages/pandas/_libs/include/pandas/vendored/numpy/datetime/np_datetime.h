/*

Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

This file is derived from NumPy 1.7. See NUMPY_LICENSE.txt

*/

#pragma once

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API

#include <numpy/ndarraytypes.h>

typedef struct {
        npy_int64 days;
        npy_int32 hrs, min, sec, ms, us, ns, seconds, microseconds, nanoseconds;
} pandas_timedeltastruct;

static const npy_datetimestruct _AS_MIN_DTS = {
    1969, 12, 31, 23, 59, 50, 776627, 963145, 224193};
static const npy_datetimestruct _FS_MIN_DTS = {
    1969, 12, 31, 21, 26, 16, 627963, 145224, 193000};
static const npy_datetimestruct _PS_MIN_DTS = {
    1969, 9, 16, 5, 57, 7, 963145, 224193, 0};
static const npy_datetimestruct _NS_MIN_DTS = {
    1677, 9, 21, 0, 12, 43, 145224, 193000, 0};
static const npy_datetimestruct _US_MIN_DTS = {
    -290308, 12, 21, 19, 59, 05, 224193, 0, 0};
static const npy_datetimestruct _MS_MIN_DTS = {
    -292275055, 5, 16, 16, 47, 4, 193000, 0, 0};
static const npy_datetimestruct _S_MIN_DTS = {
    -292277022657, 1, 27, 8, 29, 53, 0, 0, 0};
static const npy_datetimestruct _M_MIN_DTS = {
    -17536621475646, 5, 4, 5, 53, 0, 0, 0, 0};

static const npy_datetimestruct _AS_MAX_DTS = {
    1970, 1, 1, 0, 0, 9, 223372, 36854, 775807};
static const npy_datetimestruct _FS_MAX_DTS = {
    1970, 1, 1, 2, 33, 43, 372036, 854775, 807000};
static const npy_datetimestruct _PS_MAX_DTS = {
    1970, 4, 17, 18, 2, 52, 36854, 775807, 0};
static const npy_datetimestruct _NS_MAX_DTS = {
    2262, 4, 11, 23, 47, 16, 854775, 807000, 0};
static const npy_datetimestruct _US_MAX_DTS = {
    294247, 1, 10, 4, 0, 54, 775807, 0, 0};
static const npy_datetimestruct _MS_MAX_DTS = {
    292278994, 8, 17, 7, 12, 55, 807000, 0, 0};
static const npy_datetimestruct _S_MAX_DTS = {
    292277026596, 12, 4, 15, 30, 7, 0, 0, 0};
static const npy_datetimestruct _M_MAX_DTS = {
    17536621479585, 8, 30, 18, 7, 0, 0, 0, 0};

// stuff pandas needs
// ----------------------------------------------------------------------------

PyObject *extract_utc_offset(PyObject *obj);

npy_datetime npy_datetimestruct_to_datetime(NPY_DATETIMEUNIT base,
                                            const npy_datetimestruct *dts);

void pandas_datetime_to_datetimestruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                       npy_datetimestruct *result);

void pandas_timedelta_to_timedeltastruct(npy_timedelta val,
                                         NPY_DATETIMEUNIT fr,
                                         pandas_timedeltastruct *result);

extern const int days_per_month_table[2][12];

// stuff numpy-derived code needs in header
// ----------------------------------------------------------------------------

int is_leapyear(npy_int64 year);

/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts);


/*
 * Compares two npy_datetimestruct objects chronologically
 */
int cmp_npy_datetimestruct(const npy_datetimestruct *a,
                           const npy_datetimestruct *b);


/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes);

/*
 * This function returns the DateTimeMetaData
 * contained within the provided datetime dtype.
 */
PyArray_DatetimeMetaData get_datetime_metadata_from_dtype(
        PyArray_Descr *dtype);
