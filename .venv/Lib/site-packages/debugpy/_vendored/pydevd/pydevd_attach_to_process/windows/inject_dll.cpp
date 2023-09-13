#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <tchar.h>
#include <tlhelp32.h>

#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "user32.lib")

// Helper to free data when we leave the scope.
class DataToFree {
public:
    HANDLE hProcess;
    HANDLE snapshotHandle;
    
    LPVOID remoteMemoryAddr;
    int remoteMemorySize;
    
    DataToFree(){
        this->hProcess = nullptr;
        this->snapshotHandle = nullptr;
        
        this->remoteMemoryAddr = nullptr;
        this->remoteMemorySize = 0;
    }
    
    ~DataToFree() {
        if(this->hProcess != nullptr){
            
            if(this->remoteMemoryAddr != nullptr && this->remoteMemorySize != 0){
                VirtualFreeEx(this->hProcess, this->remoteMemoryAddr, this->remoteMemorySize, MEM_RELEASE);
                this->remoteMemoryAddr = nullptr;
                this->remoteMemorySize = 0;
            }
            
            CloseHandle(this->hProcess);
            this->hProcess = nullptr;
        }

        if(this->snapshotHandle != nullptr){
            CloseHandle(this->snapshotHandle);
            this->snapshotHandle = nullptr;
        }
    }
};


/**
 * All we do here is load a dll in a remote program (in a remote thread).
 *
 * Arguments must be the pid and the dll name to run.
 *
 * i.e.: inject_dll.exe <pid> <dll path>
 */
int wmain( int argc, wchar_t *argv[ ], wchar_t *envp[ ] )
{
    std::cout << "Running executable to inject dll." << std::endl;
    
    // Helper to clear resources.
    DataToFree dataToFree;
    
    if(argc != 3){
        std::cout << "Expected 2 arguments (pid, dll name)." << std::endl;
        return 1;
    }
 
    const int pid = _wtoi(argv[1]);
    if(pid == 0){
        std::cout << "Invalid pid." << std::endl;
        return 2;
    }
    
    const int MAX_PATH_SIZE_PADDED = MAX_PATH + 1;
    char dllPath[MAX_PATH_SIZE_PADDED];
    memset(&dllPath[0], '\0', MAX_PATH_SIZE_PADDED);
    size_t pathLen = 0;
    wcstombs_s(&pathLen, dllPath, argv[2], MAX_PATH);
    
    const bool inheritable = false;
    const HANDLE hProcess = OpenProcess(PROCESS_VM_OPERATION | PROCESS_CREATE_THREAD | PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_QUERY_INFORMATION, inheritable, pid);
    if(hProcess == nullptr || hProcess == INVALID_HANDLE_VALUE){
        std::cout << "Unable to open process with pid: " << pid << ". Error code: " << GetLastError() << "." << std::endl;
        return 3;
    }
    dataToFree.hProcess = hProcess;
    std::cout << "OpenProcess with pid: " << pid << std::endl;
    
    const LPVOID remoteMemoryAddr = VirtualAllocEx(hProcess, nullptr, MAX_PATH_SIZE_PADDED, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if(remoteMemoryAddr == nullptr){
        std::cout << "Error. Unable to allocate memory in pid: " << pid << ". Error code: " << GetLastError() << "." << std::endl;
        return 4;
    }
    dataToFree.remoteMemorySize = MAX_PATH_SIZE_PADDED;
    dataToFree.remoteMemoryAddr = remoteMemoryAddr;
    
    std::cout << "VirtualAllocEx in pid: " << pid << std::endl;
    
    const bool written = WriteProcessMemory(hProcess, remoteMemoryAddr, dllPath, pathLen, nullptr);
    if(!written){
        std::cout << "Error. Unable to write to memory in pid: " << pid << ". Error code: " << GetLastError() << "." << std::endl;
        return 5;
    }
    std::cout << "WriteProcessMemory in pid: " << pid << std::endl;
    
    const LPVOID loadLibraryAddress = (LPVOID) GetProcAddress(GetModuleHandle("kernel32.dll"), "LoadLibraryA");
    if(loadLibraryAddress == nullptr){
        std::cout << "Error. Unable to get LoadLibraryA address. Error code: " << GetLastError() << "." << std::endl;
        return 6;
    }
    std::cout << "loadLibraryAddress: " << pid << std::endl;
    
    const HANDLE remoteThread = CreateRemoteThread(hProcess, nullptr, 0, (LPTHREAD_START_ROUTINE) loadLibraryAddress, remoteMemoryAddr, 0, nullptr);
    if (remoteThread == nullptr) {
        std::cout << "Error. Unable to CreateRemoteThread. Error code: " << GetLastError() << "." << std::endl;
        return 7;
    }
    
    // We wait for the load to finish before proceeding to get the function to actually do the attach.
    std::cout << "Waiting for LoadLibraryA to complete." << std::endl;
    DWORD result = WaitForSingleObject(remoteThread, 5 * 1000);
    
    if(result == WAIT_TIMEOUT) {
        std::cout << "WaitForSingleObject(LoadLibraryA thread) timed out." << std::endl;
        return 8;
        
    } else if(result == WAIT_FAILED) {
        std::cout << "WaitForSingleObject(LoadLibraryA thread) failed. Error code: " << GetLastError() << "." << std::endl;
        return 9;
    }
    
    std::cout << "Ok, finished dll injection." << std::endl;
    return 0;
}