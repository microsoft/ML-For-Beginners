#include <iostream>
#include <thread>
#include "attach.h"
#include "stdafx.h"

#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <tchar.h>

typedef int (_RunCodeInMemoryInAttachedDll)();

HINSTANCE globalDllInstance = NULL;

class NotificationHelper {
public:
#pragma warning( push )
#pragma warning( disable : 4722 )
// disable c4722 here: Destructor never returns warning. Compiler sees ExitThread and assumes that
// there is a potential memory leak.
    ~NotificationHelper(){
        std::string eventName("_pydevd_pid_event_");
        eventName += std::to_string(GetCurrentProcessId());

        // When we finish we need to set the event that the caller is waiting for and
        // unload the dll (if we don't exit this dll we won't be able to reattach later on).
        auto event = CreateEventA(nullptr, false, false, eventName.c_str());
        if (event != nullptr) {
            SetEvent(event);
            CloseHandle(event);
        }
        FreeLibraryAndExitThread(globalDllInstance, 0);
    }
#pragma warning( pop )
};

DWORD WINAPI RunCodeInThread(LPVOID lpParam){
    NotificationHelper notificationHelper; // When we exit the scope the destructor should take care of the cleanup.
    
#ifdef BITS_32
    HMODULE attachModule = GetModuleHandleA("attach_x86.dll");
#else
    HMODULE attachModule = GetModuleHandleA("attach_amd64.dll");
#endif

    if (attachModule == nullptr) {
        std::cout << "Error: unable to get attach_x86.dll or attach_amd64.dll module handle." << std::endl;
        return 900;
    }

    _RunCodeInMemoryInAttachedDll* runCode = reinterpret_cast < _RunCodeInMemoryInAttachedDll* > (GetProcAddress(attachModule, "RunCodeInMemoryInAttachedDll"));
    if (runCode == nullptr) {
        std::cout << "Error: unable to GetProcAddress(attachModule, RunCodeInMemoryInAttachedDll) from attach_x86.dll or attach_amd64.dll." << std::endl;
        return 901;
    }

    runCode();
    return 0;
}

/**
 * When the dll is loaded we create a thread that will call 'RunCodeInMemoryInAttachedDll'
 * in the attach dll (when completed we unload this library for a reattach to work later on).
 */
BOOL WINAPI DllMain(
  _In_ HINSTANCE hinstDLL,
  _In_ DWORD     fdwReason,
  _In_ LPVOID    lpvReserved
){
    if(fdwReason == DLL_PROCESS_ATTACH){
        globalDllInstance = hinstDLL;
        DWORD threadId;
        CreateThread(nullptr, 0, &RunCodeInThread, nullptr, 0, &threadId);
    }
    else if(fdwReason == DLL_PROCESS_DETACH){
    }
    return true;
}
