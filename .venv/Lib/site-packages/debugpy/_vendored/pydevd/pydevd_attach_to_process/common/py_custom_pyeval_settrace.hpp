#ifndef _PY_CUSTOM_PYEVAL_SETTRACE_HPP_
#define _PY_CUSTOM_PYEVAL_SETTRACE_HPP_

#include "python.h"
#include "py_utils.hpp"
#include "py_custom_pyeval_settrace_common.hpp"
#include "py_custom_pyeval_settrace_310.hpp"
#include "py_custom_pyeval_settrace_311.hpp"

// On Python 3.7 onwards the thread state is not kept in PyThread_set_key_value (rather
// it uses PyThread_tss_set using PyThread_tss_set(&_PyRuntime.gilstate.autoTSSkey, (void *)tstate)
// and we don't have access to that key from here (thus, we can't use the previous approach which
// made CPython think that the current thread had the thread state where we wanted to set the tracing).
//
// So, the solution implemented here is not faking that change and reimplementing PyEval_SetTrace.
// The implementation is mostly the same from the one in CPython, but we have one shortcoming:
//
// When CPython sets the tracing for a thread it increments _Py_TracingPossible (actually
// _PyRuntime.ceval.tracing_possible). This implementation has one issue: it only works on
// deltas when the tracing is set (so, a settrace(func) will increase the _Py_TracingPossible global value and a
// settrace(None) will decrease it, but when a thread dies it's kept as is and is not decreased).
// -- as we don't currently have access to _PyRuntime we have to create a thread, set the tracing
// and let it die so that the count is increased (this is really hacky, but better than having
// to create a local copy of the whole _PyRuntime (defined in pystate.h with several inner structs)
// which would need to be kept up to date for each new CPython version just to increment that variable).



/**
 * This version is used in internalInitializeCustomPyEvalSetTrace->pyObject_FastCallDict on older
 * versions of CPython (pre 3.7).
 */
 static PyObject *
 PyObject_FastCallDictCustom(PyObject* callback, PyObject *stack[3], int ignoredStackSizeAlways3, void* ignored)
 {
     PyObject *args = internalInitializeCustomPyEvalSetTrace->pyTuple_New(3);
     PyObject *result;

      if (args == NULL) {
          return NULL;
      }

     IncRef(stack[0]);
     IncRef(stack[1]);
     IncRef(stack[2]);

    // I.e.: same thing as: PyTuple_SET_ITEM(args, 0, stack[0]);
    reinterpret_cast<PyTupleObject *>(args)->ob_item[0] = stack[0];
    reinterpret_cast<PyTupleObject *>(args)->ob_item[1] = stack[1];
    reinterpret_cast<PyTupleObject *>(args)->ob_item[2] = stack[2];

     /* call the Python-level function */
     result = internalInitializeCustomPyEvalSetTrace->pyEval_CallObjectWithKeywords(callback, args, (PyObject*)NULL);

    /* cleanup */
    DecRef(args, internalInitializeCustomPyEvalSetTrace->isDebug);
    return result;
}

static PyObject *
InternalCallTrampoline(PyObject* callback,
                PyFrameObjectBaseUpTo39 *frame, int what, PyObject *arg)
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
    
    
    // Helpers to print info.
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)stack[0])));
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)stack[1])));
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)stack[2])));
    // printf("%s\n", internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8(internalInitializeCustomPyEvalSetTrace->pyObject_Repr((PyObject *)callback)));

    // call the Python-level function
    // result = _PyObject_FastCall(callback, stack, 3);
    //
    // Note that _PyObject_FastCall is actually a define:
    // #define _PyObject_FastCall(func, args, nargs) _PyObject_FastCallDict((func), (args), (nargs), NULL)

    result = internalInitializeCustomPyEvalSetTrace->pyObject_FastCallDict(callback, stack, 3, NULL);


// Note: this is commented out from CPython (we shouldn't need it and it adds a reasonable overhead).
//     PyFrame_LocalsToFast(frame, 1);

    if (result == NULL) {
        internalInitializeCustomPyEvalSetTrace->pyTraceBack_Here(frame);
    }

    return result;
}

static int
InternalTraceTrampoline(PyObject *self, PyFrameObject *frameParam,
                 int what, PyObject *arg)
{
    PyObject *callback;
    PyObject *result;
    
    PyFrameObjectBaseUpTo39 *frame = reinterpret_cast<PyFrameObjectBaseUpTo39*>(frameParam);

    if (what == PyTrace_CALL){
        callback = self;
    } else {
        callback = frame->f_trace;
    }

    if (callback == NULL){
        return 0;
    }

    result = InternalCallTrampoline(callback, frame, what, arg);
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
template<typename T>
void InternalPySetTrace_Template(T tstate, PyObjectHolder* traceFunc, bool isDebug)
{
    PyObject *temp = tstate->c_traceobj;

    // We can't increase _Py_TracingPossible. Everything else should be equal to CPython.
    // runtime->ceval.tracing_possible += (func != NULL) - (tstate->c_tracefunc != NULL);

    PyObject *arg = traceFunc->ToPython();
    IncRef(arg);
    tstate->c_tracefunc = NULL;
    tstate->c_traceobj = NULL;
    /* Must make sure that profiling is not ignored if 'temp' is freed */
    tstate->use_tracing = tstate->c_profilefunc != NULL;
    if(temp != NULL){
        DecRef(temp, isDebug);
    }
    tstate->c_tracefunc = InternalTraceTrampoline;
    tstate->c_traceobj = arg;
    /* Flag that tracing or profiling is turned on */
    tstate->use_tracing = ((InternalTraceTrampoline != NULL)
                           || (tstate->c_profilefunc != NULL));

};


void InternalPySetTrace(PyThreadState* curThread, PyObjectHolder* traceFunc, bool isDebug, PythonVersion version)
{
    if (PyThreadState_25_27::IsFor(version)) {
        InternalPySetTrace_Template<PyThreadState_25_27*>(reinterpret_cast<PyThreadState_25_27*>(curThread), traceFunc, isDebug);
    } else if (PyThreadState_30_33::IsFor(version)) {
        InternalPySetTrace_Template<PyThreadState_30_33*>(reinterpret_cast<PyThreadState_30_33*>(curThread), traceFunc, isDebug);
    } else if (PyThreadState_34_36::IsFor(version)) {
        InternalPySetTrace_Template<PyThreadState_34_36*>(reinterpret_cast<PyThreadState_34_36*>(curThread), traceFunc, isDebug);
    } else if (PyThreadState_37_38::IsFor(version)) {
        InternalPySetTrace_Template<PyThreadState_37_38*>(reinterpret_cast<PyThreadState_37_38*>(curThread), traceFunc, isDebug);
    } else if (PyThreadState_39::IsFor(version)) {
        InternalPySetTrace_Template<PyThreadState_39*>(reinterpret_cast<PyThreadState_39*>(curThread), traceFunc, isDebug);
    } else if (PyThreadState_310::IsFor(version)) {
        // 3.10 has other changes on the actual algorithm (use_tracing is per-frame now), so, we have a full new version for it.
        InternalPySetTrace_Template310<PyThreadState_310*>(reinterpret_cast<PyThreadState_310*>(curThread), traceFunc, isDebug);
    } else if (PyThreadState_311::IsFor(version)) {
        InternalPySetTrace_Template311<PyThreadState_311*>(reinterpret_cast<PyThreadState_311*>(curThread), traceFunc, isDebug);
    } else {
        printf("Unable to set trace to target thread with Python version: %d", version);
    }
}


#endif //_PY_CUSTOM_PYEVAL_SETTRACE_HPP_