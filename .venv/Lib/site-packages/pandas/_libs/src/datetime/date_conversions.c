/*
Copyright (c) 2020, PyData Development Team
All rights reserved.
Distributed under the terms of the BSD Simplified License.
The full license is in the LICENSE file, distributed with this software.
*/

// Conversion routines that are useful for serialization,
// but which don't interact with JSON objects directly

#include "pandas/datetime/date_conversions.h"
#include "pandas/vendored/numpy/datetime/np_datetime.h"
#include "pandas/vendored/numpy/datetime/np_datetime_strings.h"

/*
 * Function: scaleNanosecToUnit
 * -----------------------------
 *
 * Scales an integer value representing time in nanoseconds to provided unit.
 *
 * Mutates the provided value directly. Returns 0 on success, non-zero on error.
 */
int scaleNanosecToUnit(npy_int64 *value, NPY_DATETIMEUNIT unit) {
    switch (unit) {
    case NPY_FR_ns:
        break;
    case NPY_FR_us:
        *value /= 1000LL;
        break;
    case NPY_FR_ms:
        *value /= 1000000LL;
        break;
    case NPY_FR_s:
        *value /= 1000000000LL;
        break;
    default:
        return -1;
    }

    return 0;
}

/* Converts the int64_t representation of a datetime to ISO; mutates len */
char *int64ToIso(int64_t value,
                 NPY_DATETIMEUNIT valueUnit,
                 NPY_DATETIMEUNIT base,
                 size_t *len) {
    npy_datetimestruct dts;
    int ret_code;

    pandas_datetime_to_datetimestruct(value, valueUnit, &dts);

    *len = (size_t)get_datetime_iso_8601_strlen(0, base);
    char *result = PyObject_Malloc(*len);

    if (result == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    // datetime64 is always naive
    ret_code = make_iso_8601_datetime(&dts, result, *len, 0, base);
    if (ret_code != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Could not convert datetime value to string");
        PyObject_Free(result);
    }

    // Note that get_datetime_iso_8601_strlen just gives a generic size
    // for ISO string conversion, not the actual size used
    *len = strlen(result);
    return result;
}

npy_datetime NpyDateTimeToEpoch(npy_datetime dt, NPY_DATETIMEUNIT base) {
    scaleNanosecToUnit(&dt, base);
    return dt;
}

/* Converts the int64_t representation of a duration to ISO; mutates len */
char *int64ToIsoDuration(int64_t value, size_t *len) {
    pandas_timedeltastruct tds;
    int ret_code;

    pandas_timedelta_to_timedeltastruct(value, NPY_FR_ns, &tds);

    // Max theoretical length of ISO Duration with 64 bit day
    // as the largest unit is 70 characters + 1 for a null terminator
    char *result = PyObject_Malloc(71);
    if (result == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    ret_code = make_iso_8601_timedelta(&tds, result, len);
    if (ret_code == -1) {
        PyErr_SetString(PyExc_ValueError,
                        "Could not convert timedelta value to string");
        PyObject_Free(result);
        return NULL;
    }

    return result;
}
