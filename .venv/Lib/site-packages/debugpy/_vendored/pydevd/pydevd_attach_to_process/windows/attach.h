/* ****************************************************************************
 *
 * Copyright (c) Brainwy software Ltda.
 *
 * This source code is subject to terms and conditions of the Apache License, Version 2.0. A
 * copy of the license can be found in the License.html file at the root of this distribution. If
 * you cannot locate the Apache License, Version 2.0, please send an email to
 * vspython@microsoft.com. By using this source code in any fashion, you are agreeing to be bound
 * by the terms of the Apache License, Version 2.0.
 *
 * You must not remove this notice, or any other, from this software.
 *
 * ***************************************************************************/

#ifndef _ATTACH_DLL_H_
#define _ATTACH_DLL_H_

#if defined DLL_EXPORT
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR __declspec(dllimport)
#endif


extern "C"
{
    DECLDIR int AttachAndRunPythonCode(const char *command, int *result );
    
    /*
     * Helper to print debug information from the current process
     */
    DECLDIR int PrintDebugInfo();
    
    /*
    Could be used with ctypes (note that the threading should be initialized, so, 
    doing it in a thread as below is recommended):
    
    def check():
        
        import ctypes
        lib = ctypes.cdll.LoadLibrary(r'C:\...\attach_x86.dll')
        print 'result', lib.AttachDebuggerTracing(0)
        
    t = threading.Thread(target=check)
    t.start()
    t.join()
    */
    DECLDIR int AttachDebuggerTracing(
        bool showDebugInfo, 
        void* pSetTraceFunc, // Actually PyObject*, but we don't want to include it here.
        void* pTraceFunc,  // Actually PyObject*, but we don't want to include it here.
        unsigned int threadId,
        void* pPyNone  // Actually PyObject*, but we don't want to include it here.
    );
}

#endif