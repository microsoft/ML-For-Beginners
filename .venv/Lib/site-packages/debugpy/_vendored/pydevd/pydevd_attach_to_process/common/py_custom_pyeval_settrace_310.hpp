#ifndef _PY_CUSTOM_PYEVAL_SETTRACE_310_HPP_
#define _PY_CUSTOM_PYEVAL_SETTRACE_310_HPP_

#include "python.h"
#include "py_utils.hpp"

static PyObject *
InternalCallTrampoline310(PyObject* callback,
                PyFrameObject310 *frame, int what, PyObject *arg)
{
    PyObject *result;
    PyObject *stack[3];

// Note: this is commented out from CPython (we shouldn't need it and it adds a reasonable overhead).
//     if (PyFrame_FastToLocalsWithError(frame) < 0) {
//         return NULL;
//     }
//
    stack[0] = (PyObject *)frame;
    stack[1] = InternalWhatstrings_37[what];
    stack[2] = (arg != NULL) ? arg : internalInitializeCustomPyEvalSetTrace->pyNone;
    
    
    // Helper to print info.
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)stack[0])));
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)stack[1])));
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)stack[2])));
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)callback)));

    result = internalInitializeCustomPyEvalSetTrace->pyObject_FastCallDict(callback, stack, 3, NULL);

// Note: this is commented out from CPython (we shouldn't need it and it adds a reasonable overhead).
//     PyFrame_LocalsToFast(frame, 1);

    if (result == NULL) {
        internalInitializeCustomPyEvalSetTrace->pyTraceBack_Here(frame);
    }

    return result;
}

// See: static int trace_trampoline(PyObject *self, PyFrameObject *frame, int what, PyObject *arg)
// in: https://github.com/python/cpython/blob/3.10/Python/sysmodule.c
static int
InternalTraceTrampoline310(PyObject *self, PyFrameObject *frameParam,
                 int what, PyObject *arg)
{
    PyObject *callback;
    PyObject *result;
    
    PyFrameObject310 *frame = reinterpret_cast<PyFrameObject310*>(frameParam);

    if (what == PyTrace_CALL){
        callback = self;
    } else {
        callback = frame->f_trace;
    }

    if (callback == NULL){
        return 0;
    }

    result = InternalCallTrampoline310(callback, frame, what, arg);
    if (result == NULL) {
        // Note: calling the original sys.settrace here.
        internalInitializeCustomPyEvalSetTrace->pyEval_SetTrace(NULL, NULL);
        PyObject *temp_f_trace = frame->f_trace;
        frame->f_trace = NULL;
        if(temp_f_trace != NULL){
            DecRef(temp_f_trace, internalInitializeCustomPyEvalSetTrace->isDebug);
        }
        return -1;
    }
    if (result != internalInitializeCustomPyEvalSetTrace->pyNone) {
        PyObject *tmp = frame->f_trace;
        frame->f_trace = result;
        DecRef(tmp, internalInitializeCustomPyEvalSetTrace->isDebug);
    }
    else {
        DecRef(result, internalInitializeCustomPyEvalSetTrace->isDebug);
    }
    return 0;
}

// Based on ceval.c (PyEval_SetTrace(Py_tracefunc func, PyObject *arg))
// https://github.com/python/cpython/blob/3.10/Python/ceval.c
template<typename T>
void InternalPySetTrace_Template310(T tstate, PyObjectHolder* traceFunc, bool isDebug)
{
    PyObject *temp = tstate->c_traceobj;

    PyObject *arg = traceFunc->ToPython();
    IncRef(arg);
    tstate->c_tracefunc = NULL;
    tstate->c_traceobj = NULL;
    
    // This is different (previously it was just: tstate->use_tracing, now
    // this flag is per-frame). 
    tstate->cframe->use_tracing = tstate->c_profilefunc != NULL;
    
    if(temp != NULL){
        DecRef(temp, isDebug);
    }
    tstate->c_tracefunc = InternalTraceTrampoline310;
    tstate->c_traceobj = arg;
    /* Flag that tracing or profiling is turned on */
    tstate->cframe->use_tracing = ((InternalTraceTrampoline310 != NULL)
                           || (tstate->c_profilefunc != NULL));

};


#endif //_PY_CUSTOM_PYEVAL_SETTRACE_310_HPP_