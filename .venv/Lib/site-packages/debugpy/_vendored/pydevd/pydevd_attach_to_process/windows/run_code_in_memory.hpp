#ifndef _PY_RUN_CODE_IN_MEMORY_HPP_
#define _PY_RUN_CODE_IN_MEMORY_HPP_

#include <iostream>
#include "attach.h"
#include "stdafx.h"

#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <tchar.h>

#include "../common/python.h"
#include "../common/ref_utils.hpp"
#include "../common/py_utils.hpp"
#include "../common/py_settrace.hpp"

#include "py_win_helpers.hpp"

#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "advapi32.lib")

DECLDIR int AttachAndRunPythonCode(const char *command, int *attachInfo );

// NOTE: BUFSIZE must be the same from add_code_to_python_process.py
#define BUFSIZE 2048

// Helper to free data when we leave the scope.
class DataToFree {

public:

    HANDLE hMapFile;
    void* mapViewOfFile;
    char* codeToRun;

    DataToFree() {
        this->hMapFile = nullptr;
        this->mapViewOfFile = nullptr;
        this->codeToRun = nullptr;
    }

    ~DataToFree() {
        if (this->hMapFile != nullptr) {
            CloseHandle(this->hMapFile);
            this->hMapFile = nullptr;
        }
        if (this->mapViewOfFile != nullptr) {
            UnmapViewOfFile(this->mapViewOfFile);
            this->mapViewOfFile = nullptr;
        }
        if (this->codeToRun != nullptr) {
            delete this->codeToRun;
            this->codeToRun = nullptr;
        }
    }
};



extern "C"
{
    /**
     * This method will read the code to be executed from the named shared memory
     * and execute it.
     */
    DECLDIR int RunCodeInMemoryInAttachedDll() {
        // PRINT("Attempting to run Python code from named shared memory.")
        //get the code to be run (based on https://docs.microsoft.com/en-us/windows/win32/memory/creating-named-shared-memory).
        HANDLE hMapFile;
        char* mapViewOfFile;

        DataToFree dataToFree;

        std::string namedSharedMemoryName("__pydevd_pid_code_to_run__");
        namedSharedMemoryName += std::to_string(GetCurrentProcessId());

        hMapFile = OpenFileMappingA(
            FILE_MAP_ALL_ACCESS, // read/write access
            FALSE, // do not inherit the name
            namedSharedMemoryName.c_str()); // name of mapping object

        if (hMapFile == nullptr) {
            std::cout << "Error opening named shared memory (OpenFileMapping): " << GetLastError() + " name: " << namedSharedMemoryName << std::endl;
            return 1;
        } else {
            // PRINT("Opened named shared memory.")
        }

        dataToFree.hMapFile = hMapFile;

        mapViewOfFile = reinterpret_cast < char* > (MapViewOfFile(hMapFile, // handle to map object
            FILE_MAP_ALL_ACCESS, // read/write permission
            0,
            0,
            BUFSIZE));

        if (mapViewOfFile == nullptr) {
            std::cout << "Error mapping view of named shared memory (MapViewOfFile): " << GetLastError() << std::endl;
            return 1;
        } else {
            // PRINT("Mapped view of file.")
        }
        dataToFree.mapViewOfFile = mapViewOfFile;
        // std::cout << "Will run contents: " << mapViewOfFile << std::endl;

        dataToFree.codeToRun = new char[BUFSIZE];
        memmove(dataToFree.codeToRun, mapViewOfFile, BUFSIZE);

        int attachInfo = 0;
        return AttachAndRunPythonCode(dataToFree.codeToRun, &attachInfo);
    }
}

#endif