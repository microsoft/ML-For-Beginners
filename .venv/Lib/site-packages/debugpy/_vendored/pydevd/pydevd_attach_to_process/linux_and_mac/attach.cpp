// This is much simpler than the windows version because we're using gdb and
// we assume that gdb will call things in the correct thread already.

//compile with: g++ -shared -o attach_linux.so -fPIC -nostartfiles attach_linux.c


#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdbool.h>

#include "../common/python.h"
#include "../common/ref_utils.hpp"
#include "../common/py_utils.hpp"
#include "../common/py_settrace.hpp"
//#include <unistd.h> used for usleep

// Exported function: hello(): Just to print something and check that we've been
// able to connect.
extern "C" int hello(void);

int hello()
{
    printf("Hello world!\n");

    void *module = dlopen(nullptr, 0x2);

    void *hndl = dlsym (module, "PyGILState_Ensure");
    if(hndl == nullptr){
        printf("nullptr\n");

    }else{
        printf("Worked (found PyGILState_Ensure)!\n");
    }

    printf("%d", GetPythonVersion(module));


    return 2;
}


// Internal function to keep on the tracing
int _PYDEVD_ExecWithGILSetSysStrace(bool showDebugInfo, bool isDebug);

// Implementation details below
typedef PyObject* (PyImport_ImportModuleNoBlock) (const char *name);
typedef int (*PyEval_ThreadsInitialized)();
typedef unsigned long (*_PyEval_GetSwitchInterval)(void);
typedef void (*_PyEval_SetSwitchInterval)(unsigned long microseconds);

// isDebug is pretty important! Must be true on python debug builds (python_d)
// If this value is passed wrongly the program will crash.
extern "C" int DoAttach(bool isDebug, const char *command, bool showDebugInfo);

int DoAttach(bool isDebug, const char *command, bool showDebugInfo)
{
    void *module = dlopen(nullptr, 0x2);
    DEFINE_PROC(isInitFunc, Py_IsInitialized*, "Py_IsInitialized", 1);
    DEFINE_PROC(gilEnsure, PyGILState_Ensure*, "PyGILState_Ensure", 51);
    DEFINE_PROC(gilRelease, PyGILState_Release*, "PyGILState_Release", 51);


    if(!isInitFunc()){
        if(showDebugInfo){
            printf("Py_IsInitialized returned false.\n");
        }
        return 2;
    }

    PythonVersion version = GetPythonVersion(module);

    DEFINE_PROC(interpHead, PyInterpreterState_Head*, "PyInterpreterState_Head", 51);

    auto head = interpHead();
    if (head == nullptr) {
        // this interpreter is loaded but not initialized.
        if(showDebugInfo){
            printf("Interpreter not initialized!\n");
        }
        return 54;
    }

    // Note: unlike windows where we have to do many things to enable threading
    // to work to get the gil, here we'll be executing in an existing thread,
    // so, it's mostly a matter of getting the GIL and running it and we shouldn't
    // have any more problems.

    DEFINE_PROC(pyRun_SimpleString, PyRun_SimpleString*, "PyRun_SimpleString", 51);
    
    GilHolder gilLock(gilEnsure, gilRelease);   // acquire and hold the GIL until done...
    
    pyRun_SimpleString(command);
    return 0;
}


// This is the function which enables us to set the sys.settrace for all the threads
// which are already running.
extern "C" int AttachDebuggerTracing(bool showDebugInfo, void* pSetTraceFunc, void* pTraceFunc, unsigned int threadId, void* pPyNone);

int AttachDebuggerTracing(bool showDebugInfo, void* pSetTraceFunc, void* pTraceFunc, unsigned int threadId, void* pPyNone)
{
    void *module = dlopen(nullptr, 0x2);
    bool isDebug = false;
    PyObjectHolder traceFunc(isDebug, (PyObject*) pTraceFunc, true);
    PyObjectHolder setTraceFunc(isDebug, (PyObject*) pSetTraceFunc, true);
    PyObjectHolder pyNone(isDebug, reinterpret_cast<PyObject*>(pPyNone), true);
    return InternalSetSysTraceFunc(module, isDebug, showDebugInfo, &traceFunc, &setTraceFunc, threadId, &pyNone);
}

