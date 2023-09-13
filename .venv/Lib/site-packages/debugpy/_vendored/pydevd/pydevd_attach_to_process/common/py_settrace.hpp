#ifndef _PY_SETTRACE_HPP_
#define _PY_SETTRACE_HPP_

#include "ref_utils.hpp"
#include "py_utils.hpp"
#include "python.h"
#include "py_custom_pyeval_settrace.hpp"
#include <unordered_set>


#ifdef _WIN32

typedef HMODULE MODULE_TYPE;
#else // LINUX -----------------------------------------------------------------

typedef void* MODULE_TYPE;
typedef ssize_t SSIZE_T;
typedef unsigned int DWORD;

#endif

DWORD GetPythonThreadId(PythonVersion version, PyThreadState* curThread) {
    DWORD threadId = 0;
    if (PyThreadState_25_27::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_25_27*)curThread)->thread_id;
    } else if (PyThreadState_30_33::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_30_33*)curThread)->thread_id;
    } else if (PyThreadState_34_36::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_34_36*)curThread)->thread_id;
    } else if (PyThreadState_37_38::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_37_38*)curThread)->thread_id;
    } else if (PyThreadState_39::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_39*)curThread)->thread_id;
    } else if (PyThreadState_310::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_310*)curThread)->thread_id;
    } else if (PyThreadState_311::IsFor(version)) {
        threadId = (DWORD)((PyThreadState_311*)curThread)->thread_id;
    }
    return threadId;
}


/**
 * This function may be called to set a tracing function to existing python threads.
 */
int InternalSetSysTraceFunc(
    MODULE_TYPE module,
    bool isDebug,
    bool showDebugInfo,
    PyObjectHolder* traceFunc,
    PyObjectHolder* setTraceFunc,
    unsigned int threadId,
    PyObjectHolder* pyNone)
{

    if(showDebugInfo){
        PRINT("InternalSetSysTraceFunc started.");
    }

    DEFINE_PROC(isInit, Py_IsInitialized*, "Py_IsInitialized", 100);
    if (!isInit()) {
        PRINT("Py_IsInitialized returned false.");
        return 110;
    }

    auto version = GetPythonVersion(module);

    // found initialized Python runtime, gather and check the APIs we need.

    DEFINE_PROC(interpHead, PyInterpreterState_Head*, "PyInterpreterState_Head", 120);
    DEFINE_PROC(gilEnsure, PyGILState_Ensure*, "PyGILState_Ensure", 130);
    DEFINE_PROC(gilRelease, PyGILState_Release*, "PyGILState_Release", 140);
    DEFINE_PROC(threadHead, PyInterpreterState_ThreadHead*, "PyInterpreterState_ThreadHead", 150);
    DEFINE_PROC(threadNext, PyThreadState_Next*, "PyThreadState_Next", 160);
    DEFINE_PROC(threadSwap, PyThreadState_Swap*, "PyThreadState_Swap", 170);
    DEFINE_PROC(call, PyObject_CallFunctionObjArgs*, "PyObject_CallFunctionObjArgs", 180);

    PyInt_FromLong* intFromLong;

    if (version >= PythonVersion_30) {
        DEFINE_PROC(intFromLongPy3, PyInt_FromLong*, "PyLong_FromLong", 190);
        intFromLong = intFromLongPy3;
    } else {
        DEFINE_PROC(intFromLongPy2, PyInt_FromLong*, "PyInt_FromLong", 200);
        intFromLong = intFromLongPy2;
    }

    DEFINE_PROC(pyGetAttr, PyObject_GetAttrString*, "PyObject_GetAttrString", 250);
    DEFINE_PROC(pyHasAttr, PyObject_HasAttrString*, "PyObject_HasAttrString", 260);
    DEFINE_PROC_NO_CHECK(PyCFrame_Type, PyTypeObject*, "PyCFrame_Type", 300);  // optional

    DEFINE_PROC_NO_CHECK(curPythonThread, PyThreadState**, "_PyThreadState_Current", 310);  // optional
    DEFINE_PROC_NO_CHECK(getPythonThread, _PyThreadState_UncheckedGet*, "_PyThreadState_UncheckedGet", 320);  // optional

    if (curPythonThread == nullptr && getPythonThread == nullptr) {
        // we're missing some APIs, we cannot attach.
        PRINT("Error, missing Python threading API!!");
        return 330;
    }

    auto head = interpHead();
    if (head == nullptr) {
        // this interpreter is loaded but not initialized.
        PRINT("Interpreter not initialized!");
        return 340;
    }

    GilHolder gilLock(gilEnsure, gilRelease);   // acquire and hold the GIL until done...

    int retVal = 0;
    // find what index is holding onto the thread state...
    auto curPyThread = getPythonThread ? getPythonThread() : *curPythonThread;

    if(curPyThread == nullptr){
        PRINT("Getting the current python thread returned nullptr.");
        return 345;
    }


    // We do what PyEval_SetTrace does, but for any target thread.
    PyUnicode_InternFromString* pyUnicode_InternFromString;
    if (version >= PythonVersion_30) {
        DEFINE_PROC(unicodeFromString, PyUnicode_InternFromString*, "PyUnicode_InternFromString", 520);
        pyUnicode_InternFromString = unicodeFromString;
    } else {
        DEFINE_PROC(stringFromString, PyUnicode_InternFromString*, "PyString_InternFromString", 525);
        pyUnicode_InternFromString = stringFromString;
    }

    DEFINE_PROC_NO_CHECK(pyObject_FastCallDict, _PyObject_FastCallDict*, "_PyObject_FastCallDict", 530);
    DEFINE_PROC(pyTuple_New, PyTuple_New*, "PyTuple_New", 531);
    DEFINE_PROC(pyEval_CallObjectWithKeywords, PyEval_CallObjectWithKeywords*, "PyEval_CallObjectWithKeywords", 532);

    if(pyObject_FastCallDict == nullptr) {
        DEFINE_PROC_NO_CHECK(pyObject_VectorcallDict, _PyObject_FastCallDict*, "PyObject_VectorcallDict", 533);
        pyObject_FastCallDict = pyObject_VectorcallDict;
    }
                                                                                                         
    if(pyObject_FastCallDict == nullptr) {
        // we have to use PyObject_FastCallDictCustom for older versions of CPython (pre 3.7).
        pyObject_FastCallDict = reinterpret_cast<_PyObject_FastCallDict*>(&PyObject_FastCallDictCustom);
    }


    DEFINE_PROC(pyTraceBack_Here, PyTraceBack_Here*, "PyTraceBack_Here", 540);
    DEFINE_PROC(pyEval_SetTrace, PyEval_SetTrace*, "PyEval_SetTrace", 550);
    
    // These are defined mostly for printing info while debugging, so, if they're not there, don't bother reporting.
    DEFINE_PROC_NO_CHECK(pyObject_Repr, PyObject_Repr*, "PyObject_Repr", 551);
    DEFINE_PROC_NO_CHECK(pyUnicode_AsUTF8, PyUnicode_AsUTF8*, "PyUnicode_AsUTF8", 552);


    bool found = false;
    for (PyThreadState* curThread = threadHead(head); curThread != nullptr; curThread = threadNext(curThread)) {
        if (GetPythonThreadId(version, curThread) != threadId) {
            continue;
        }
        found = true;

        if(showDebugInfo){
            printf("setting trace for thread: %d\n", threadId);
        }

        if(!InternalIsTraceInitialized())
        {
            InternalInitializeCustomPyEvalSetTrace *internalInitializeCustomPyEvalSetTrace = new InternalInitializeCustomPyEvalSetTrace();

            IncRef(pyNone->ToPython());
            internalInitializeCustomPyEvalSetTrace->pyNone = pyNone->ToPython();

            internalInitializeCustomPyEvalSetTrace->pyUnicode_InternFromString = pyUnicode_InternFromString;
            internalInitializeCustomPyEvalSetTrace->pyObject_FastCallDict = pyObject_FastCallDict;
            internalInitializeCustomPyEvalSetTrace->isDebug = isDebug;
            internalInitializeCustomPyEvalSetTrace->pyTraceBack_Here = pyTraceBack_Here;
            internalInitializeCustomPyEvalSetTrace->pyEval_SetTrace = pyEval_SetTrace;
            internalInitializeCustomPyEvalSetTrace->pyTuple_New = pyTuple_New;
            internalInitializeCustomPyEvalSetTrace->pyEval_CallObjectWithKeywords = pyEval_CallObjectWithKeywords;
            internalInitializeCustomPyEvalSetTrace->pyObject_Repr = pyObject_Repr;
            internalInitializeCustomPyEvalSetTrace->pyUnicode_AsUTF8 = pyUnicode_AsUTF8;

            InternalTraceInit(internalInitializeCustomPyEvalSetTrace);
        }
        InternalPySetTrace(curThread, traceFunc, isDebug, version);
        break;
    }
    if(!found) {
        retVal = 501;
    }

    return retVal;

}

#endif // _PY_SETTRACE_HPP_
