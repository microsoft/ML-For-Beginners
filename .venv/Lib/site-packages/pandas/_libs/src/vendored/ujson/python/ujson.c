/*
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the ESN Social Software AB nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights reserved.

Numeric decoder derived from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
* Copyright (c) 1988-1993 The Regents of the University of California.
* Copyright (c) 1994 Sun Microsystems, Inc.
*/

#include "pandas/vendored/ujson/python/version.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL UJSON_NUMPY
#include "numpy/arrayobject.h"

/* objToJSON */
PyObject *objToJSON(PyObject *self, PyObject *args, PyObject *kwargs);
void *initObjToJSON(void);

/* JSONToObj */
PyObject *JSONToObj(PyObject *self, PyObject *args, PyObject *kwargs);

#define ENCODER_HELP_TEXT                                                  \
    "Use ensure_ascii=false to output UTF-8. Pass in double_precision to " \
    "alter the maximum digit precision of doubles. Set "                   \
    "encode_html_chars=True to encode < > & as unicode escape sequences."

static PyMethodDef ujsonMethods[] = {
    {"ujson_dumps", (PyCFunction)objToJSON, METH_VARARGS | METH_KEYWORDS,
     "Converts arbitrary object recursively into JSON. " ENCODER_HELP_TEXT},
    {"ujson_loads", (PyCFunction)JSONToObj, METH_VARARGS | METH_KEYWORDS,
     "Converts JSON as string to dict object structure. Use precise_float=True "
     "to use high precision float decoder."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

typedef struct {
    PyObject *type_decimal;
    PyObject *type_dataframe;
    PyObject *type_series;
    PyObject *type_index;
    PyObject *type_nat;
    PyObject *type_na;
} modulestate;

#define modulestate(o) ((modulestate *)PyModule_GetState(o))

static int module_traverse(PyObject *m, visitproc visit, void *arg);
static int module_clear(PyObject *m);
static void module_free(void *module);

static struct PyModuleDef moduledef = {.m_base = PyModuleDef_HEAD_INIT,
                                       .m_name = "pandas._libs.json",
                                       .m_methods = ujsonMethods,
                                       .m_size = sizeof(modulestate),
                                       .m_traverse = module_traverse,
                                       .m_clear = module_clear,
                                       .m_free = module_free};

#ifndef PYPY_VERSION
/* Used in objToJSON.c */
int object_is_decimal_type(PyObject *obj) {
    PyObject *module = PyState_FindModule(&moduledef);
    if (module == NULL)
        return 0;
    modulestate *state = modulestate(module);
    if (state == NULL)
        return 0;
    PyObject *type_decimal = state->type_decimal;
    if (type_decimal == NULL) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_decimal);
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_dataframe_type(PyObject *obj) {
    PyObject *module = PyState_FindModule(&moduledef);
    if (module == NULL)
        return 0;
    modulestate *state = modulestate(module);
    if (state == NULL)
        return 0;
    PyObject *type_dataframe = state->type_dataframe;
    if (type_dataframe == NULL) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_dataframe);
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_series_type(PyObject *obj) {
    PyObject *module = PyState_FindModule(&moduledef);
    if (module == NULL)
        return 0;
    modulestate *state = modulestate(module);
    if (state == NULL)
        return 0;
    PyObject *type_series = state->type_series;
    if (type_series == NULL) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_series);
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_index_type(PyObject *obj) {
    PyObject *module = PyState_FindModule(&moduledef);
    if (module == NULL)
        return 0;
    modulestate *state = modulestate(module);
    if (state == NULL)
        return 0;
    PyObject *type_index = state->type_index;
    if (type_index == NULL) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_index);
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_nat_type(PyObject *obj) {
    PyObject *module = PyState_FindModule(&moduledef);
    if (module == NULL)
        return 0;
    modulestate *state = modulestate(module);
    if (state == NULL)
        return 0;
    PyObject *type_nat = state->type_nat;
    if (type_nat == NULL) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_nat);
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_na_type(PyObject *obj) {
    PyObject *module = PyState_FindModule(&moduledef);
    if (module == NULL)
        return 0;
    modulestate *state = modulestate(module);
    if (state == NULL)
        return 0;
    PyObject *type_na = state->type_na;
    if (type_na == NULL) {
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_na);
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}
#else
    /* Used in objToJSON.c */
int object_is_decimal_type(PyObject *obj) {
    PyObject *module = PyImport_ImportModule("decimal");
    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }
    PyObject *type_decimal = PyObject_GetAttrString(module, "Decimal");
    if (type_decimal == NULL) {
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_decimal);
    if (result == -1) {
        Py_DECREF(module);
        Py_DECREF(type_decimal);
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_dataframe_type(PyObject *obj) {
    PyObject *module = PyImport_ImportModule("pandas");
    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }
    PyObject *type_dataframe = PyObject_GetAttrString(module, "DataFrame");
    if (type_dataframe == NULL) {
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_dataframe);
    if (result == -1) {
        Py_DECREF(module);
        Py_DECREF(type_dataframe);
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_series_type(PyObject *obj) {
    PyObject *module = PyImport_ImportModule("pandas");
    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }
    PyObject *type_series = PyObject_GetAttrString(module, "Series");
    if (type_series == NULL) {
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_series);
    if (result == -1) {
        Py_DECREF(module);
        Py_DECREF(type_series);
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_index_type(PyObject *obj) {
    PyObject *module = PyImport_ImportModule("pandas");
    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }
    PyObject *type_index = PyObject_GetAttrString(module, "Index");
    if (type_index == NULL) {
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_index);
    if (result == -1) {
        Py_DECREF(module);
        Py_DECREF(type_index);
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_nat_type(PyObject *obj) {
    PyObject *module = PyImport_ImportModule("pandas._libs.tslibs.nattype");
    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }
    PyObject *type_nat = PyObject_GetAttrString(module, "NaTType");
    if (type_nat == NULL) {
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_nat);
    if (result == -1) {
        Py_DECREF(module);
        Py_DECREF(type_nat);
        PyErr_Clear();
        return 0;
    }
    return result;
}

int object_is_na_type(PyObject *obj) {
    PyObject *module = PyImport_ImportModule("pandas._libs.missing");
    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }
    PyObject *type_na = PyObject_GetAttrString(module, "NAType");
    if (type_na == NULL) {
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    int result = PyObject_IsInstance(obj, type_na);
    if (result == -1) {
        Py_DECREF(module);
        Py_DECREF(type_na);
        PyErr_Clear();
        return 0;
    }
    return result;
}

#endif

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(modulestate(m)->type_decimal);
    Py_VISIT(modulestate(m)->type_dataframe);
    Py_VISIT(modulestate(m)->type_series);
    Py_VISIT(modulestate(m)->type_index);
    Py_VISIT(modulestate(m)->type_nat);
    Py_VISIT(modulestate(m)->type_na);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(modulestate(m)->type_decimal);
    Py_CLEAR(modulestate(m)->type_dataframe);
    Py_CLEAR(modulestate(m)->type_series);
    Py_CLEAR(modulestate(m)->type_index);
    Py_CLEAR(modulestate(m)->type_nat);
    Py_CLEAR(modulestate(m)->type_na);
    return 0;
}

static void module_free(void *module) { module_clear((PyObject *)module); }

PyMODINIT_FUNC PyInit_json(void) {
    import_array()
    PyObject *module;

#ifndef PYPY_VERSION
    // This function is not supported in PyPy.
    if ((module = PyState_FindModule(&moduledef)) != NULL) {
        Py_INCREF(module);
        return module;
    }
#endif

    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;
    }

#ifndef PYPY_VERSION
    PyObject *mod_decimal = PyImport_ImportModule("decimal");
    if (mod_decimal) {
        PyObject *type_decimal = PyObject_GetAttrString(mod_decimal, "Decimal");
        assert(type_decimal != NULL);
        modulestate(module)->type_decimal = type_decimal;
        Py_DECREF(mod_decimal);
    }

    PyObject *mod_pandas = PyImport_ImportModule("pandas");
    if (mod_pandas) {
        PyObject *type_dataframe =
            PyObject_GetAttrString(mod_pandas, "DataFrame");
        assert(type_dataframe != NULL);
        modulestate(module)->type_dataframe = type_dataframe;

        PyObject *type_series = PyObject_GetAttrString(mod_pandas, "Series");
        assert(type_series != NULL);
        modulestate(module)->type_series = type_series;

        PyObject *type_index = PyObject_GetAttrString(mod_pandas, "Index");
        assert(type_index != NULL);
        modulestate(module)->type_index = type_index;

        Py_DECREF(mod_pandas);
    }

    PyObject *mod_nattype =
        PyImport_ImportModule("pandas._libs.tslibs.nattype");
    if (mod_nattype) {
        PyObject *type_nat = PyObject_GetAttrString(mod_nattype, "NaTType");
        assert(type_nat != NULL);
        modulestate(module)->type_nat = type_nat;

        Py_DECREF(mod_nattype);
    }

    PyObject *mod_natype = PyImport_ImportModule("pandas._libs.missing");
    if (mod_natype) {
        PyObject *type_na = PyObject_GetAttrString(mod_natype, "NAType");
        assert(type_na != NULL);
        modulestate(module)->type_na = type_na;

        Py_DECREF(mod_natype);
    } else {
        PyErr_Clear();
    }
#endif

    /* Not vendored for now
    JSONDecodeError = PyErr_NewException("ujson.JSONDecodeError",
    PyExc_ValueError, NULL); Py_XINCREF(JSONDecodeError); if
    (PyModule_AddObject(module, "JSONDecodeError", JSONDecodeError) < 0)
    {
      Py_XDECREF(JSONDecodeError);
      Py_CLEAR(JSONDecodeError);
      Py_DECREF(module);
      return NULL;
    }
    */

    return module;
}
