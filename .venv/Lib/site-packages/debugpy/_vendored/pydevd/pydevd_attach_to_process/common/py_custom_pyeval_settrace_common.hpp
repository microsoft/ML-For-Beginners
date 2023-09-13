#ifndef _PY_CUSTOM_PYEVAL_SETTRACE_COMMON_HPP_
#define _PY_CUSTOM_PYEVAL_SETTRACE_COMMON_HPP_

#include "python.h"
#include "py_utils.hpp"

struct InternalInitializeCustomPyEvalSetTrace {
    PyObject* pyNone;
    PyTuple_New* pyTuple_New;
    _PyObject_FastCallDict* pyObject_FastCallDict;
    PyEval_CallObjectWithKeywords* pyEval_CallObjectWithKeywords;
    PyUnicode_InternFromString* pyUnicode_InternFromString;  // Note: in Py2 will be PyString_InternFromString.
    PyTraceBack_Here* pyTraceBack_Here;
    PyEval_SetTrace* pyEval_SetTrace;
    bool isDebug;
    PyUnicode_AsUTF8* pyUnicode_AsUTF8;
    PyObject_Repr* pyObject_Repr;
};

/**
 * Helper information to access CPython internals.
 */
static InternalInitializeCustomPyEvalSetTrace *internalInitializeCustomPyEvalSetTrace = NULL;

/*
 * Cached interned string objects used for calling the profile and
 * trace functions.  Initialized by InternalTraceInit().
 */
static PyObject *InternalWhatstrings_37[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};


static int
InternalIsTraceInitialized()
{
    return internalInitializeCustomPyEvalSetTrace != NULL;
}



static int
InternalTraceInit(InternalInitializeCustomPyEvalSetTrace *p_internalInitializeSettrace_37)
{
    internalInitializeCustomPyEvalSetTrace = p_internalInitializeSettrace_37;
    static const char * const whatnames[8] = {
        "call", "exception", "line", "return",
        "c_call", "c_exception", "c_return",
        "opcode"
    };
    PyObject *name;
    int i;
    for (i = 0; i < 8; ++i) {
        if (InternalWhatstrings_37[i] == NULL) {
            name = internalInitializeCustomPyEvalSetTrace->pyUnicode_InternFromString(whatnames[i]);
            if (name == NULL)
                return -1;
            InternalWhatstrings_37[i] = name;
        }
    }
    return 0;
}

#endif //_PY_CUSTOM_PYEVAL_SETTRACE_COMMON_HPP_