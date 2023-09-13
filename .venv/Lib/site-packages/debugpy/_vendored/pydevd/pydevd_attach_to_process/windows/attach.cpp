/* ****************************************************************************
*
* Copyright (c) Microsoft Corporation.
*
* This source code is subject to terms and conditions of the Apache License, Version 2.0. A
* copy of the license can be found in the License.html file at the root of this distribution. If
* you cannot locate the Apache License, Version 2.0, please send an email to
* vspython@microsoft.com. By using this source code in any fashion, you are agreeing to be bound
* by the terms of the Apache License, Version 2.0.
*
* You must not remove this notice, or any other, from this software.
*
* Contributor: Fabio Zadrozny
*
* Based on PyDebugAttach.cpp from PVTS. Windows only.
*
* https://github.com/Microsoft/PTVS/blob/master/Python/Product/PyDebugAttach/PyDebugAttach.cpp
*
* Initially we did an attach completely based on shellcode which got the
* GIL called PyRun_SimpleString with the needed code and was done with it
* (so, none of this code was needed).
* Now, newer version of Python don't initialize threading by default, so,
* most of this code is done only to overcome this limitation (and as a plus,
* if there's no code running, we also pause the threads to make our code run).
*
* On Linux the approach is still the simpler one (using gdb), so, on newer
* versions of Python it may not work unless the user has some code running
* and threads are initialized.
* I.e.:
*
* The user may have to add the code below in the start of its script for
* a successful attach (if he doesn't already use threads).
*
* from threading import Thread
* Thread(target=str).start()
*
* -- this is the workaround for the fact that we can't get the gil
* if there aren't any threads (PyGILState_Ensure gives an error).
* ***************************************************************************/


// Access to std::cout and std::endl
#include <iostream>
#include <mutex>
// DECLDIR will perform an export for us
#define DLL_EXPORT

#include "attach.h"
#include "stdafx.h"

#include "../common/python.h"
#include "../common/ref_utils.hpp"
#include "../common/py_utils.hpp"
#include "../common/py_settrace.hpp"


#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "psapi.lib")

#include "py_win_helpers.hpp"
#include "run_code_in_memory.hpp"

// _Always_ is not defined for all versions, so make it a no-op if missing.
#ifndef _Always_
#define _Always_(x) x
#endif


typedef void (PyEval_Lock)(); // Acquire/Release lock
typedef void (PyThreadState_API)(PyThreadState *); // Acquire/Release lock
typedef PyObject* (Py_CompileString)(const char *str, const char *filename, int start);
typedef PyObject* (PyEval_EvalCode)(PyObject *co, PyObject *globals, PyObject *locals);
typedef PyObject* (PyDict_GetItemString)(PyObject *p, const char *key);
typedef PyObject* (PyEval_GetBuiltins)();
typedef int (PyDict_SetItemString)(PyObject *dp, const char *key, PyObject *item);
typedef int (PyEval_ThreadsInitialized)();
typedef int (Py_AddPendingCall)(int (*func)(void *), void*);
typedef PyObject* (PyString_FromString)(const char* s);
typedef void PyEval_SetTrace(Py_tracefunc func, PyObject *obj);
typedef PyObject* (PyErr_Print)();
typedef PyObject* (PyObject_SetAttrString)(PyObject *o, const char *attr_name, PyObject* value);
typedef PyObject* (PyBool_FromLong)(long v);
typedef unsigned long (_PyEval_GetSwitchInterval)(void);
typedef void (_PyEval_SetSwitchInterval)(unsigned long microseconds);
typedef PyGILState_STATE PyGILState_EnsureFunc(void);
typedef void PyGILState_ReleaseFunc(PyGILState_STATE);
typedef PyThreadState *PyThreadState_NewFunc(PyInterpreterState *interp);

typedef PyObject *PyList_New(Py_ssize_t len);
typedef int PyList_Append(PyObject *list, PyObject *item);



std::wstring GetCurrentModuleFilename() {
    HMODULE hModule = nullptr;
    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCTSTR)GetCurrentModuleFilename, &hModule) != 0) {
        wchar_t filename[MAX_PATH];
        GetModuleFileName(hModule, filename, MAX_PATH);
        return filename;
    }
    return std::wstring();
}


struct InitializeThreadingInfo {
    PyImport_ImportModule* pyImportMod;
    PyEval_Lock* initThreads;

    std::mutex mutex;
    HANDLE initedEvent;  // Note: only access with mutex locked (and check if not already nullptr).
    bool completed; // Note: only access with mutex locked
};


int AttachCallback(void *voidInitializeThreadingInfo) {
    // initialize us for threading, this will acquire the GIL if not already created, and is a nop if the GIL is created.
    // This leaves us in the proper state when we return back to the runtime whether the GIL was created or not before
    // we were called.
    InitializeThreadingInfo* initializeThreadingInfo = reinterpret_cast<InitializeThreadingInfo*>(voidInitializeThreadingInfo);
    initializeThreadingInfo->initThreads(); // Note: calling multiple times is ok.
    initializeThreadingInfo->pyImportMod("threading");

    initializeThreadingInfo->mutex.lock();
    if(initializeThreadingInfo->initedEvent != nullptr) {
        SetEvent(initializeThreadingInfo->initedEvent);
    }
    initializeThreadingInfo->completed = true;
    initializeThreadingInfo->mutex.unlock();
    return 0;
}


// create a custom heap for our unordered map.  This is necessary because if we suspend a thread while in a heap function
// then we could deadlock here.  We need to be VERY careful about what we do while the threads are suspended.
static HANDLE g_heap = 0;

template<typename T>
class PrivateHeapAllocator {
public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef T         value_type;

    template<class U>
    struct rebind {
        typedef PrivateHeapAllocator<U> other;
    };

    explicit PrivateHeapAllocator() {}

    PrivateHeapAllocator(PrivateHeapAllocator const&) {}

    ~PrivateHeapAllocator() {}

    template<typename U>
    PrivateHeapAllocator(PrivateHeapAllocator<U> const&) {}

    pointer allocate(size_type size, std::allocator<void>::const_pointer hint = 0) {
        UNREFERENCED_PARAMETER(hint);

        if (g_heap == nullptr) {
            g_heap = HeapCreate(0, 0, 0);
        }
        auto mem = HeapAlloc(g_heap, 0, size * sizeof(T));
        return static_cast<pointer>(mem);
    }

    void deallocate(pointer p, size_type n) {
        UNREFERENCED_PARAMETER(n);

        HeapFree(g_heap, 0, p);
    }

    size_type max_size() const {
        return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    void construct(pointer p, const T& t) {
        new(p) T(t);
    }

    void destroy(pointer p) {
        p->~T();
    }
};

typedef std::unordered_map<DWORD, HANDLE, std::hash<DWORD>, std::equal_to<DWORD>, PrivateHeapAllocator<std::pair<DWORD, HANDLE>>> ThreadMap;

void ResumeThreads(ThreadMap &suspendedThreads) {
    for (auto start = suspendedThreads.begin();  start != suspendedThreads.end(); start++) {
        ResumeThread((*start).second);
        CloseHandle((*start).second);
    }
    suspendedThreads.clear();
}

// Suspends all threads ensuring that they are not currently in a call to Py_AddPendingCall.
void SuspendThreads(ThreadMap &suspendedThreads, Py_AddPendingCall* addPendingCall, PyEval_ThreadsInitialized* threadsInited) {
    DWORD curThreadId = GetCurrentThreadId();
    DWORD curProcess = GetCurrentProcessId();
    // suspend all the threads in the process so we can do things safely...
    bool suspended;

    do {
        suspended = false;
        HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (h != INVALID_HANDLE_VALUE) {

            THREADENTRY32 te;
            te.dwSize = sizeof(te);
            if (Thread32First(h, &te)) {
                do {
                    if (te.dwSize >= FIELD_OFFSET(THREADENTRY32, th32OwnerProcessID) + sizeof(te.th32OwnerProcessID) && te.th32OwnerProcessID == curProcess) {


                        if (te.th32ThreadID != curThreadId && suspendedThreads.find(te.th32ThreadID) == suspendedThreads.end()) {
                            auto hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, te.th32ThreadID);
                            if (hThread != nullptr) {
                                SuspendThread(hThread);

                                bool addingPendingCall = false;

                                CONTEXT context;
                                memset(&context, 0x00, sizeof(CONTEXT));
                                context.ContextFlags = CONTEXT_ALL;
                                GetThreadContext(hThread, &context);

#if defined(_X86_)
                                if (context.Eip >= *(reinterpret_cast<DWORD*>(addPendingCall)) && context.Eip <= (*(reinterpret_cast<DWORD*>(addPendingCall))) + 0x100) {
                                    addingPendingCall = true;
                                }
#elif defined(_AMD64_)
                                if (context.Rip >= *(reinterpret_cast<DWORD64*>(addPendingCall)) && context.Rip <= *(reinterpret_cast<DWORD64*>(addPendingCall) + 0x100)) {
                                    addingPendingCall = true;
                                }
#endif

                                if (addingPendingCall) {
                                    // we appear to be adding a pending call via this thread - wait for this to finish so we can add our own pending call...
                                    ResumeThread(hThread);
                                    SwitchToThread();   // yield to the resumed thread if it's on our CPU...
                                    CloseHandle(hThread);
                                } else {
                                    suspendedThreads[te.th32ThreadID] = hThread;
                                }
                                suspended = true;
                            }
                        }
                    }

                    te.dwSize = sizeof(te);
                } while (Thread32Next(h, &te) && !threadsInited());
            }
            CloseHandle(h);
        }
    } while (suspended && !threadsInited());
}



extern "C"
{

    /**
     * The returned value signals the error that happened!
     *
     * Return codes:
     * 0 = all OK.
     * 1 = Py_IsInitialized not found
     * 2 = Py_IsInitialized returned false
     * 3 = Missing Python API
     * 4 = Interpreter not initialized
     * 5 = Python version unknown
     * 6 = Connect timeout
     **/
	int DoAttach(HMODULE module, bool isDebug, const char *command, bool showDebugInfo )
	{
        auto isInit = reinterpret_cast<Py_IsInitialized*>(GetProcAddress(module, "Py_IsInitialized"));

        if (isInit == nullptr) {
            std::cerr << "Py_IsInitialized not found. " << std::endl << std::flush;
            return 1;
        }
        if (!isInit()) {
            std::cerr << "Py_IsInitialized returned false. " << std::endl << std::flush;
            return 2;
        }

        auto version = GetPythonVersion(module);

        // found initialized Python runtime, gather and check the APIs we need for a successful attach...
        DEFINE_PROC(addPendingCall, Py_AddPendingCall*, "Py_AddPendingCall", -100);
        DEFINE_PROC(interpHead, PyInterpreterState_Head*, "PyInterpreterState_Head", -110);
        DEFINE_PROC(gilEnsure, PyGILState_Ensure*, "PyGILState_Ensure", -120);
        DEFINE_PROC(gilRelease, PyGILState_Release*, "PyGILState_Release", -130);
        DEFINE_PROC(threadHead, PyInterpreterState_ThreadHead*, "PyInterpreterState_ThreadHead", -140);
        DEFINE_PROC(initThreads, PyEval_Lock*, "PyEval_InitThreads", -150);
        DEFINE_PROC(releaseLock, PyEval_Lock*, "PyEval_ReleaseLock", -160);
        DEFINE_PROC(threadsInited, PyEval_ThreadsInitialized*, "PyEval_ThreadsInitialized", -170);
        DEFINE_PROC(threadNext, PyThreadState_Next*, "PyThreadState_Next", -180);
        DEFINE_PROC(pyImportMod, PyImport_ImportModule*, "PyImport_ImportModule", -190);
        DEFINE_PROC(pyNone, PyObject*, "_Py_NoneStruct", -2000);
        DEFINE_PROC(pyRun_SimpleString, PyRun_SimpleString*, "PyRun_SimpleString", -210);

        // Either _PyThreadState_Current or _PyThreadState_UncheckedGet are required
        DEFINE_PROC_NO_CHECK(curPythonThread, PyThreadState**, "_PyThreadState_Current", -220);  // optional
        DEFINE_PROC_NO_CHECK(getPythonThread, _PyThreadState_UncheckedGet*, "_PyThreadState_UncheckedGet", -230);  // optional

        if (curPythonThread == nullptr && getPythonThread == nullptr) {
            // we're missing some APIs, we cannot attach.
            std::cerr << "Error, missing Python threading API!!" << std::endl << std::flush;
            return -240;
        }

        // Either _Py_CheckInterval or _PyEval_[GS]etSwitchInterval are useful, but not required
        DEFINE_PROC_NO_CHECK(intervalCheck, int*, "_Py_CheckInterval", -250);  // optional
        DEFINE_PROC_NO_CHECK(getSwitchInterval, _PyEval_GetSwitchInterval*, "_PyEval_GetSwitchInterval", -260);  // optional
        DEFINE_PROC_NO_CHECK(setSwitchInterval, _PyEval_SetSwitchInterval*, "_PyEval_SetSwitchInterval", -270);  // optional

        auto head = interpHead();
        if (head == nullptr) {
            // this interpreter is loaded but not initialized.
            std::cerr << "Interpreter not initialized! " << std::endl << std::flush;
            return 4;
        }

        // check that we're a supported version
        if (version == PythonVersion_Unknown) {
            std::cerr << "Python version unknown! " << std::endl << std::flush;
            return 5;
        } else if (version == PythonVersion_25 || version == PythonVersion_26 ||
                   version == PythonVersion_30 || version == PythonVersion_31 || version == PythonVersion_32) {
            std::cerr << "Python version unsupported! " << std::endl << std::flush;
            return 5;
        }


        // We always try to initialize threading and import the threading module in the main thread in the code
        // below...
        //
        // We need to initialize multiple threading support but we need to do so safely, so we call
        // Py_AddPendingCall and have our callback then initialize multi threading.  This is completely safe on 2.7
        // and up. Unfortunately that doesn't work if we're not actively running code on the main thread (blocked on a lock
        // or reading input).
        //
        // Another option is to make sure no code is running - if there is no active thread then we can safely call
        // PyEval_InitThreads and we're in business.  But to know this is safe we need to first suspend all the other
        // threads in the process and then inspect if any code is running (note that this is still not ideal because
        // this thread will be the thread head for Python, but still better than not attach at all).
        //
        // Finally if code is running after we've suspended the threads then we can go ahead and do Py_AddPendingCall
        // on down-level interpreters as long as we're sure no one else is making a call to Py_AddPendingCall at the same
        // time.
        //
        // Therefore our strategy becomes: Make the Py_AddPendingCall on interpreters and wait for it. If it doesn't
        // call after a timeout, suspend all threads - if a threads is in Py_AddPendingCall resume and try again.  Once we've got all of the threads
        // stopped and not in Py_AddPendingCall (which calls no functions its self, you can see this and it's size in the
        // debugger) then see if we have a current thread. If not go ahead and initialize multiple threading (it's now safe,
        // no Python code is running).

        InitializeThreadingInfo *initializeThreadingInfo = new InitializeThreadingInfo();
        initializeThreadingInfo->pyImportMod = pyImportMod;
        initializeThreadingInfo->initThreads = initThreads;
        initializeThreadingInfo->initedEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);

        // Add the call to initialize threading.
        addPendingCall(&AttachCallback, initializeThreadingInfo);

        ::WaitForSingleObject(initializeThreadingInfo->initedEvent, 5000);

        // Whether this completed or not, release the event handle as we won't use it anymore.
        initializeThreadingInfo->mutex.lock();
        CloseHandle(initializeThreadingInfo->initedEvent);
        bool completed = initializeThreadingInfo->completed;
        initializeThreadingInfo->initedEvent = nullptr;
        initializeThreadingInfo->mutex.unlock();

        if(completed) {
            // Note that this structure will leak if addPendingCall did not complete in the timeout
            // (we can't release now because it's possible that it'll still be called).
            delete initializeThreadingInfo;
            if (showDebugInfo) {
                std::cout << "addPendingCall to initialize threads/import threading completed. " << std::endl << std::flush;
            }
        } else {
            if (showDebugInfo) {
                std::cout << "addPendingCall to initialize threads/import threading did NOT complete. " << std::endl << std::flush;
            }
        }

        if (threadsInited()) {
            // Note that since Python 3.7, threads are *always* initialized!
            if (showDebugInfo) {
                std::cout << "Threads initialized! " << std::endl << std::flush;
            }

        } else {
            int saveIntervalCheck;
            unsigned long saveLongIntervalCheck;
            if (intervalCheck != nullptr) {
                // not available on 3.2
                saveIntervalCheck = *intervalCheck;
                *intervalCheck = -1;    // lower the interval check so pending calls are processed faster
                saveLongIntervalCheck = 0; // prevent compiler warning
            } else if (getSwitchInterval != nullptr && setSwitchInterval != nullptr) {
                saveLongIntervalCheck = getSwitchInterval();
                setSwitchInterval(0);
                saveIntervalCheck = 0; // prevent compiler warning
            }
            else {
                saveIntervalCheck = 0; // prevent compiler warning
                saveLongIntervalCheck = 0; // prevent compiler warning
            }

            // If threads weren't initialized in our pending call, instead of giving a timeout, try
            // to initialize it in this thread.
            for(int attempts = 0; !threadsInited() && attempts < 20; attempts++) {
                if(attempts > 0){
                    // If we haven't been able to do it in the first time, wait a bit before retrying.
                    Sleep(10);
                }

                ThreadMap suspendedThreads;
                if (showDebugInfo) {
                    std::cout << "SuspendThreads(suspendedThreads, addPendingCall, threadsInited);" << std::endl << std::flush;
                }
                SuspendThreads(suspendedThreads, addPendingCall, threadsInited);

                if(!threadsInited()){ // Check again with threads suspended.
                    if (showDebugInfo) {
                        std::cout << "ENTERED if (!threadsInited()) {" << std::endl << std::flush;
                    }
                    auto curPyThread = getPythonThread ? getPythonThread() : *curPythonThread;

                    if (curPyThread == nullptr) {
                        if (showDebugInfo) {
                            std::cout << "ENTERED if (curPyThread == nullptr) {" << std::endl << std::flush;
                        }
                         // no threads are currently running, it is safe to initialize multi threading.
                         PyGILState_STATE gilState;
                         if (version >= PythonVersion_34) {
                             // in 3.4 due to http://bugs.python.org/issue20891,
                             // we need to create our thread state manually
                             // before we can call PyGILState_Ensure() before we
                             // can call PyEval_InitThreads().

                             // Don't require this function unless we need it.
                             auto threadNew = (PyThreadState_NewFunc*)GetProcAddress(module, "PyThreadState_New");
                             if (threadNew != nullptr) {
                                 threadNew(head);
                             }
                         }

                         if (version >= PythonVersion_32) {
                             // in 3.2 due to the new GIL and later we can't call Py_InitThreads
                             // without a thread being initialized.
                             // So we use PyGilState_Ensure here to first
                             // initialize the current thread, and then we use
                             // Py_InitThreads to bring up multi-threading.
                             // Some context here: http://bugs.python.org/issue11329
                             // http://pytools.codeplex.com/workitem/834
                            gilState = gilEnsure();
                        }
                        else {
                            gilState = PyGILState_LOCKED; // prevent compiler warning
                         }

                        if (showDebugInfo) {
                            std::cout << "Called initThreads()" << std::endl << std::flush;
                        }
                        // Initialize threads in our secondary thread (this is NOT ideal because
                        // this thread will be the thread head), but is still better than not being
                        // able to attach if the main thread is not actually running any code.
                        initThreads();

                         if (version >= PythonVersion_32) {
                             // we will release the GIL here
                            gilRelease(gilState);
                         } else {
                             releaseLock();
                         }
                    }
                }
                ResumeThreads(suspendedThreads);
            }


            if (intervalCheck != nullptr) {
                *intervalCheck = saveIntervalCheck;
            } else if (setSwitchInterval != nullptr) {
                setSwitchInterval(saveLongIntervalCheck);
            }

        }

        if (g_heap != nullptr) {
            HeapDestroy(g_heap);
            g_heap = nullptr;
        }

        if (!threadsInited()) {
            std::cerr << "Unable to initialize threads in the given timeout! " << std::endl << std::flush;
            return 8;
        }

        GilHolder gilLock(gilEnsure, gilRelease);   // acquire and hold the GIL until done...

        pyRun_SimpleString(command);
        return 0;

    }




    // ======================================== Code related to setting tracing to existing threads.


    /**
     * This function is meant to be called to execute some arbitrary python code to be
     * run. It'll initialize threads as needed and then run the code with pyRun_SimpleString.
     *
     * @param command: the python code to be run
     * @param attachInfo: pointer to an int specifying whether we should show debug info (1) or not (0).
     **/
    DECLDIR int AttachAndRunPythonCode(const char *command, int *attachInfo )
    {

        int SHOW_DEBUG_INFO = 1;

        bool showDebugInfo = (*attachInfo & SHOW_DEBUG_INFO) != 0;

        if (showDebugInfo) {
            std::cout << "AttachAndRunPythonCode started (showing debug info). " << std::endl << std::flush;
        }

        ModuleInfo moduleInfo = GetPythonModule();
        if (moduleInfo.errorGettingModule != 0) {
            return moduleInfo.errorGettingModule;
        }
        HMODULE module = moduleInfo.module;
        int attached = DoAttach(module, moduleInfo.isDebug, command, showDebugInfo);

        if (attached != 0) {
            std::cerr << "Error when injecting code in target process. Error code (on windows): " << attached << std::endl << std::flush;
        }
        return attached;
    }


    DECLDIR int PrintDebugInfo() {
        PRINT("Getting debug info...");
        ModuleInfo moduleInfo = GetPythonModule();
        if (moduleInfo.errorGettingModule != 0) {
            PRINT("Error getting python module");
            return 0;
        }
        HMODULE module = moduleInfo.module;

        DEFINE_PROC(interpHead, PyInterpreterState_Head*, "PyInterpreterState_Head", 0);
        DEFINE_PROC(threadHead, PyInterpreterState_ThreadHead*, "PyInterpreterState_ThreadHead", 0);
        DEFINE_PROC(threadNext, PyThreadState_Next*, "PyThreadState_Next", 160);
        DEFINE_PROC(gilEnsure, PyGILState_Ensure*, "PyGILState_Ensure", 0);
        DEFINE_PROC(gilRelease, PyGILState_Release*, "PyGILState_Release", 0);

        auto head = interpHead();
        if (head == nullptr) {
            // this interpreter is loaded but not initialized.
            PRINT("Interpreter not initialized!");
            return 0;
        }

        auto version = GetPythonVersion(module);
        printf("Python version: %d\n", version);

        GilHolder gilLock(gilEnsure, gilRelease);   // acquire and hold the GIL until done...
        auto curThread = threadHead(head);
        if (curThread == nullptr) {
            PRINT("Thread head is NULL.")
            return 0;
        }

        for (auto curThread = threadHead(head); curThread != nullptr; curThread = threadNext(curThread)) {
            printf("Found thread id: %d\n", GetPythonThreadId(version, curThread));
        }

        PRINT("Finished getting debug info.")
        return 0;
    }


    /**
     * This function may be called to set a tracing function to existing python threads.
     **/
    DECLDIR int AttachDebuggerTracing(bool showDebugInfo, void* pSetTraceFunc, void* pTraceFunc, unsigned int threadId, void* pPyNone)
    {
        ModuleInfo moduleInfo = GetPythonModule();
        if (moduleInfo.errorGettingModule != 0) {
            return moduleInfo.errorGettingModule;
        }
        HMODULE module = moduleInfo.module;
        if (showDebugInfo) {
            std::cout << "Setting sys trace for existing threads." << std::endl << std::flush;
        }
        int attached = 0;
        PyObjectHolder traceFunc(moduleInfo.isDebug, reinterpret_cast<PyObject*>(pTraceFunc), true);
        PyObjectHolder setTraceFunc(moduleInfo.isDebug, reinterpret_cast<PyObject*>(pSetTraceFunc), true);
        PyObjectHolder pyNone(moduleInfo.isDebug, reinterpret_cast<PyObject*>(pPyNone), true);

        int temp = InternalSetSysTraceFunc(module, moduleInfo.isDebug, showDebugInfo, &traceFunc, &setTraceFunc, threadId, &pyNone);
        if (temp == 0) {
            // we've successfully attached the debugger
            return 0;
        } else {
           if (temp > attached) {
               //I.e.: the higher the value the more significant it is.
               attached = temp;
            }
        }

        if (showDebugInfo) {
            std::cout << "Setting sys trace for existing threads failed with code: " << attached << "." << std::endl << std::flush;
        }
        return attached;
    }

}

