setlocal
@cd /d %~dp0

@set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
@echo Using vswhere at %VSWHERE%
@for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
@echo Using Visual C++ at %VSDIR%
                                 
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" x86 -vcvars_spectre_libs=spectre

cl -DUNICODE -D_UNICODE /EHsc /Zi /O1 /W3 /LD /MD /Qspectre attach.cpp /link /DEBUG /OPT:REF /OPT:ICF /GUARD:CF /out:attach_x86.dll
copy attach_x86.dll ..\attach_x86.dll /Y
copy attach_x86.pdb ..\attach_x86.pdb /Y

cl -DUNICODE -D_UNICODE /EHsc /Zi /O1 /W3 /LD /MD /D BITS_32 /Qspectre run_code_on_dllmain.cpp /link /DEBUG /OPT:REF /OPT:ICF /GUARD:CF /out:run_code_on_dllmain_x86.dll
copy run_code_on_dllmain_x86.dll ..\run_code_on_dllmain_x86.dll /Y
copy run_code_on_dllmain_x86.pdb ..\run_code_on_dllmain_x86.pdb /Y

cl /EHsc /Zi /O1 /W3 /Qspectre inject_dll.cpp /link /DEBUG  /OPT:REF /OPT:ICF /GUARD:CF /out:inject_dll_x86.exe
copy inject_dll_x86.exe ..\inject_dll_x86.exe /Y
copy inject_dll_x86.pdb ..\inject_dll_x86.pdb /Y

call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64 -vcvars_spectre_libs=spectre

cl -DUNICODE -D_UNICODE /EHsc /Zi /O1 /W3 /LD /MD /Qspectre attach.cpp /link /DEBUG /OPT:REF /OPT:ICF /GUARD:CF /out:attach_amd64.dll
copy attach_amd64.dll ..\attach_amd64.dll /Y
copy attach_amd64.pdb ..\attach_amd64.pdb /Y

cl -DUNICODE -D_UNICODE /EHsc /Zi /O1 /W3 /LD /MD /D BITS_64 /Qspectre run_code_on_dllmain.cpp /link /DEBUG /OPT:REF /OPT:ICF /GUARD:CF /out:run_code_on_dllmain_amd64.dll
copy run_code_on_dllmain_amd64.dll ..\run_code_on_dllmain_amd64.dll /Y
copy run_code_on_dllmain_amd64.pdb ..\run_code_on_dllmain_amd64.pdb /Y

cl /EHsc /Zi /O1 /W3 /Qspectre inject_dll.cpp /link /DEBUG  /OPT:REF /OPT:ICF /GUARD:CF /out:inject_dll_amd64.exe
copy inject_dll_amd64.exe ..\inject_dll_amd64.exe /Y
copy inject_dll_amd64.pdb ..\inject_dll_amd64.pdb /Y

del *.exe
del *.lib
del *.obj
del *.pdb
del *.dll
del *.exp