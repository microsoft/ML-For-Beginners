/*

Copyright (c) 2023, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

*/
#define _PANDAS_PARSER_IMPL

#include "pandas/parser/pd_parser.h"
#include "pandas/parser/io.h"

static int to_double(char *item, double *p_value, char sci, char decimal,
                            int *maybe_int) {
  char *p_end = NULL;
  int error = 0;

  /* Switch to precise xstrtod GH 31364 */
  *p_value =
      precise_xstrtod(item, &p_end, decimal, sci, '\0', 1, &error, maybe_int);

  return (error == 0) && (!*p_end);
}

static int floatify(PyObject *str, double *result, int *maybe_int) {
  int status;
  char *data;
  PyObject *tmp = NULL;
  const char sci = 'E';
  const char dec = '.';

  if (PyBytes_Check(str)) {
    data = PyBytes_AS_STRING(str);
  } else if (PyUnicode_Check(str)) {
    tmp = PyUnicode_AsUTF8String(str);
    if (tmp == NULL) {
      return -1;
    }
    data = PyBytes_AS_STRING(tmp);
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid object type");
    return -1;
  }

  status = to_double(data, result, sci, dec, maybe_int);

  if (!status) {
    /* handle inf/-inf infinity/-infinity */
    if (strlen(data) == 3) {
      if (0 == strcasecmp(data, "inf")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else if (strlen(data) == 4) {
      if (0 == strcasecmp(data, "-inf")) {
        *result = -HUGE_VAL;
        *maybe_int = 0;
      } else if (0 == strcasecmp(data, "+inf")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else if (strlen(data) == 8) {
      if (0 == strcasecmp(data, "infinity")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else if (strlen(data) == 9) {
      if (0 == strcasecmp(data, "-infinity")) {
        *result = -HUGE_VAL;
        *maybe_int = 0;
      } else if (0 == strcasecmp(data, "+infinity")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else {
      goto parsingerror;
    }
  }

  Py_XDECREF(tmp);
  return 0;

parsingerror:
  PyErr_Format(PyExc_ValueError, "Unable to parse string \"%s\"", data);
  Py_XDECREF(tmp);
  return -1;
}


static void pandas_parser_destructor(PyObject *op) {
  void *ptr = PyCapsule_GetPointer(op, PandasParser_CAPSULE_NAME);
  PyMem_Free(ptr);
}

static int pandas_parser_exec(PyObject *module) {
  PandasParser_CAPI *capi = PyMem_Malloc(sizeof(PandasParser_CAPI));
  if (capi == NULL) {
    PyErr_NoMemory();
    return -1;
  }

  capi->to_double = to_double;
  capi->floatify = floatify;
  capi->new_rd_source = new_rd_source;
  capi->del_rd_source = del_rd_source;
  capi->buffer_rd_bytes = buffer_rd_bytes;
  capi->uint_state_init = uint_state_init;
  capi->uint64_conflict = uint64_conflict;
  capi->coliter_setup = coliter_setup;
  capi->parser_new = parser_new;
  capi->parser_init = parser_init;
  capi->parser_free = parser_free;
  capi->parser_del = parser_del;
  capi->parser_add_skiprow = parser_add_skiprow;
  capi->parser_set_skipfirstnrows = parser_set_skipfirstnrows;
  capi->parser_set_default_options = parser_set_default_options;
  capi->parser_consume_rows = parser_consume_rows;
  capi->parser_trim_buffers = parser_trim_buffers;
  capi->tokenize_all_rows = tokenize_all_rows;
  capi->tokenize_nrows = tokenize_nrows;
  capi->str_to_int64 = str_to_int64;
  capi->str_to_uint64 = str_to_uint64;
  capi->xstrtod = xstrtod;
  capi->precise_xstrtod = precise_xstrtod;
  capi->round_trip = round_trip;
  capi->to_boolean = to_boolean;

  PyObject *capsule =
      PyCapsule_New(capi, PandasParser_CAPSULE_NAME, pandas_parser_destructor);
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
                    "pd_parser.c could not import module pandas");
    Py_DECREF(capsule);
    return -1;
  }

  if (PyModule_AddObject(pandas, "_pandas_parser_CAPI", capsule) < 0) {
    Py_DECREF(capsule);
    return -1;
  }

  return 0;
}

static PyModuleDef_Slot pandas_parser_slots[] = {
    {Py_mod_exec, pandas_parser_exec}, {0, NULL}};

static struct PyModuleDef pandas_parsermodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pandas._libs.pandas_parser",

    .m_doc = "Internal module with parser support for other extensions",
    .m_size = 0,
    .m_methods = NULL,
    .m_slots = pandas_parser_slots};

PyMODINIT_FUNC PyInit_pandas_parser(void) {
  return PyModuleDef_Init(&pandas_parsermodule);
}
