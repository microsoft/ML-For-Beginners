/*
Copyright (c) 2020, PyData Development Team
All rights reserved.
Distributed under the terms of the BSD Simplified License.
The full license is in the LICENSE file, distributed with this software.
*/

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarraytypes.h>

// Scales value inplace from nanosecond resolution to unit resolution
int scaleNanosecToUnit(npy_int64 *value, NPY_DATETIMEUNIT unit);

// Converts an int64 object representing a date to ISO format
// up to precision `base` e.g. base="s" yields 2020-01-03T00:00:00Z
// while base="ns" yields "2020-01-01T00:00:00.000000000Z"
// len is mutated to save the length of the returned string
char *int64ToIso(int64_t value,
                 NPY_DATETIMEUNIT valueUnit,
                 NPY_DATETIMEUNIT base,
                 size_t *len);

// TODO(username): this function doesn't do a lot; should augment or
// replace with scaleNanosecToUnit
npy_datetime NpyDateTimeToEpoch(npy_datetime dt, NPY_DATETIMEUNIT base);

char *int64ToIsoDuration(int64_t value, size_t *len);
