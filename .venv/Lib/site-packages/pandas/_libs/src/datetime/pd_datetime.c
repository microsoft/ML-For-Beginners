/*

Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Copyright (c) 2005-2011, NumPy Developers
All rights reserved.

This file is derived from NumPy 1.7. See NUMPY_LICENSE.txt

*/

#define _PANDAS_DATETIME_IMPL

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "datetime.h"
#include "pandas/datetime/pd_datetime.h"


static void pandas_datetime_destructor(PyObject *op) {
  void *ptr = PyCapsule_GetPointer(op, PandasDateTime_CAPSULE_NAME);
  PyMem_Free(ptr);
}

/*
 *
 * Converts a Python datetime.datetime or datetime.date
 * object into a NumPy npy_datetimestruct.  Uses tzinfo (if present)
 * to convert to UTC time.
 *
 * The following implementation just asks for attributes, and thus
 * supports datetime duck typing. The tzinfo time zone conversion
 * requires this style of access as well.
 *
 * Returns -1 on error, 0 on success, and 1 (with no error set)
 * if obj doesn't have the needed date or datetime attributes.
 */
static int convert_pydatetime_to_datetimestruct(PyObject *dtobj,
                                         npy_datetimestruct *out) {
    // Assumes that obj is a valid datetime object
    PyObject *tmp;
    PyObject *obj = (PyObject*)dtobj;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    out->year = PyLong_AsLong(PyObject_GetAttrString(obj, "year"));
    out->month = PyLong_AsLong(PyObject_GetAttrString(obj, "month"));
    out->day = PyLong_AsLong(PyObject_GetAttrString(obj, "day"));

    // TODO(anyone): If we can get PyDateTime_IMPORT to work, we could use
    // PyDateTime_Check here, and less verbose attribute lookups.

    /* Check for time attributes (if not there, return success as a date) */
    if (!PyObject_HasAttrString(obj, "hour") ||
        !PyObject_HasAttrString(obj, "minute") ||
        !PyObject_HasAttrString(obj, "second") ||
        !PyObject_HasAttrString(obj, "microsecond")) {
        return 0;
    }

    out->hour = PyLong_AsLong(PyObject_GetAttrString(obj, "hour"));
    out->min = PyLong_AsLong(PyObject_GetAttrString(obj, "minute"));
    out->sec = PyLong_AsLong(PyObject_GetAttrString(obj, "second"));
    out->us = PyLong_AsLong(PyObject_GetAttrString(obj, "microsecond"));

    if (PyObject_HasAttrString(obj, "tzinfo")) {
        PyObject *offset = extract_utc_offset(obj);
        /* Apply the time zone offset if datetime obj is tz-aware */
        if (offset != NULL) {
            if (offset == Py_None) {
                Py_DECREF(offset);
                return 0;
            }
            PyObject *tmp_int;
            int seconds_offset, minutes_offset;
            /*
             * The timedelta should have a function "total_seconds"
             * which contains the value we want.
             */
            tmp = PyObject_CallMethod(offset, "total_seconds", "");
            Py_DECREF(offset);
            if (tmp == NULL) {
                return -1;
            }
            tmp_int = PyNumber_Long(tmp);
            if (tmp_int == NULL) {
                Py_DECREF(tmp);
                return -1;
            }
            seconds_offset = PyLong_AsLong(tmp_int);
            if (seconds_offset == -1 && PyErr_Occurred()) {
                Py_DECREF(tmp_int);
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp_int);
            Py_DECREF(tmp);

            /* Convert to a minutes offset and apply it */
            minutes_offset = seconds_offset / 60;

            add_minutes_to_datetimestruct(out, -minutes_offset);
        }
    }

    return 0;
}

// Converts a Python object representing a Date / Datetime to ISO format
// up to precision `base` e.g. base="s" yields 2020-01-03T00:00:00Z
// while base="ns" yields "2020-01-01T00:00:00.000000000Z"
// len is mutated to save the length of the returned string
static char *PyDateTimeToIso(PyObject *obj, NPY_DATETIMEUNIT base,
                      size_t *len) {
    npy_datetimestruct dts;
    int ret;

    ret = convert_pydatetime_to_datetimestruct(obj, &dts);
    if (ret != 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                            "Could not convert PyDateTime to numpy datetime");
        }
        return NULL;
    }

    *len = (size_t)get_datetime_iso_8601_strlen(0, base);
    char *result = PyObject_Malloc(*len);
    // Check to see if PyDateTime has a timezone.
    // Don't convert to UTC if it doesn't.
    int is_tz_aware = 0;
    if (PyObject_HasAttrString(obj, "tzinfo")) {
        PyObject *offset = extract_utc_offset(obj);
        if (offset == NULL) {
            PyObject_Free(result);
            return NULL;
        }
        is_tz_aware = offset != Py_None;
        Py_DECREF(offset);
    }
    ret = make_iso_8601_datetime(&dts, result, *len, is_tz_aware, base);

    if (ret != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Could not convert datetime value to string");
        PyObject_Free(result);
        return NULL;
    }

    // Note that get_datetime_iso_8601_strlen just gives a generic size
    // for ISO string conversion, not the actual size used
    *len = strlen(result);
    return result;
}

// Convert a Python Date/Datetime to Unix epoch with resolution base
static npy_datetime PyDateTimeToEpoch(PyObject *dt, NPY_DATETIMEUNIT base) {
    npy_datetimestruct dts;
    int ret;

    ret = convert_pydatetime_to_datetimestruct(dt, &dts);
    if (ret != 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                            "Could not convert PyDateTime to numpy datetime");
        }
        // TODO(username): is setting errMsg required?
        // ((JSONObjectEncoder *)tc->encoder)->errorMsg = "";
        // return NULL;
    }

    npy_datetime npy_dt = npy_datetimestruct_to_datetime(NPY_FR_ns, &dts);
    return NpyDateTimeToEpoch(npy_dt, base);
}

static int pandas_datetime_exec(PyObject *module) {
  PyDateTime_IMPORT;
  PandasDateTime_CAPI *capi = PyMem_Malloc(sizeof(PandasDateTime_CAPI));
  if (capi == NULL) {
    PyErr_NoMemory();
    return -1;
  }
  capi->npy_datetimestruct_to_datetime = npy_datetimestruct_to_datetime;
  capi->scaleNanosecToUnit = scaleNanosecToUnit;
  capi->int64ToIso = int64ToIso;
  capi->NpyDateTimeToEpoch = NpyDateTimeToEpoch;
  capi->PyDateTimeToIso = PyDateTimeToIso;
  capi->PyDateTimeToEpoch = PyDateTimeToEpoch;
  capi->int64ToIsoDuration = int64ToIsoDuration;
  capi->pandas_datetime_to_datetimestruct = pandas_datetime_to_datetimestruct;
  capi->pandas_timedelta_to_timedeltastruct =
      pandas_timedelta_to_timedeltastruct;
  capi->convert_pydatetime_to_datetimestruct =
      convert_pydatetime_to_datetimestruct;
  capi->cmp_npy_datetimestruct = cmp_npy_datetimestruct;
  capi->get_datetime_metadata_from_dtype = get_datetime_metadata_from_dtype;
  capi->parse_iso_8601_datetime = parse_iso_8601_datetime;
  capi->get_datetime_iso_8601_strlen = get_datetime_iso_8601_strlen;
  capi->make_iso_8601_datetime = make_iso_8601_datetime;
  capi->make_iso_8601_timedelta = make_iso_8601_timedelta;

  PyObject *capsule = PyCapsule_New(capi, PandasDateTime_CAPSULE_NAME,
                                    pandas_datetime_destructor);
  if (capsule == NULL) {
    PyMem_Free(capi);
    return -1;
  }

  // Monkeypatch the top level pandas module to have an attribute for the
  // C-API. This is required because Python capsules do not support setting
  // this attribute on anything but the top level package. Ideally not
  // done when cpython gh-6898 gets implemented
  PyObject *pandas = PyImport_ImportModule("pandas");
  if (!pandas) {
    PyErr_SetString(PyExc_ImportError,
                    "pd_datetime.c could not import module pandas");
    Py_DECREF(capsule);
    return -1;
  }

  if (PyModule_AddObject(pandas, "_pandas_datetime_CAPI", capsule) < 0) {
    Py_DECREF(capsule);
    return -1;
  }

  return 0;
}

static PyModuleDef_Slot pandas_datetime_slots[] = {
    {Py_mod_exec, pandas_datetime_exec}, {0, NULL}};

static struct PyModuleDef pandas_datetimemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pandas._libs.pandas_datetime",

    .m_doc = "Internal module with datetime support for other extensions",
    .m_size = 0,
    .m_methods = NULL,
    .m_slots = pandas_datetime_slots};

PyMODINIT_FUNC PyInit_pandas_datetime(void) {
  PyDateTime_IMPORT;
  return PyModuleDef_Init(&pandas_datetimemodule);
}
